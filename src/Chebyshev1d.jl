# Fast Chebyshev approximation, integration, and inverse-CDF sampling
# for 1D densities. Julia port of chebyshev_utils.py from
# https://github.com/danleonte/Simulation-based-inference-via-telescoping-ratio-estimation-for-trawl-processes-paper/blob/main/src/utils/chebyshev_utils.py
#
# Various Julia libraries already support efficiently approximating 1D functions, i.e. f: R -> R, by Chebyshev polynomials. 
# This scripts targets posterior sampling for the TelescopingRatioEstimator, which requires sequentally
# approximating batches of 1D (conditional) denisities. For efficiency, the implementation has to
# i)  be GPU-compatible and GPU-oriented, hence non-adaptive; the degree of the approximating polynomial can not frequently change
# ii) allow for constructing approximations in parallel, and also generated independent samples in parallel.
#
# It is crucial to use vectorized bisection and not gradient-based methods, which require   
# different numbers of evaluations, thus preventing effective parallelism. By comparison,
# ApproxFun, FastChebInterp and FastTransforms.jl are adaptive or CPU oriented, or both. 
#
# Design notes
# ------------
#  * FFT-free to keep a clean, background agnostic implementation. 
#    The mapping from evaluated function values -> coefficients map 
#    is given by a fixed (deg+1)x(deg+1) matrix `D` 
#    (a DCT-I folded into a matmul); integration is a fixed
#    (deg+2)x(deg+1) matrix `Mint`. Both are precomputed once in `ChebPlan`.
#    This is the moral equivalent of static arguments from JAX.
#    A note on efficiency is in order. Although FFT is O(N log N) and matrix
#    multiplication is O(N^2), testing shows that in the intended usecase of the 
#    TelescopingRatioEstimator, the dominant step is sampling. 

#  * Sampling (inverse CDF by bisection) is ONE KernelAbstractions kernel:
#    each thread owns one sample and runs the whole Clenshaw + bisection loop
#    in registers. The same kernel compiles for the CPU backend (multithreaded;
#    start Julia with `-t auto`) and for CUDA (load CUDA.jl and pass CuArrays).
#    IMPORTANT: A previous version used a broadcast-based bisection, which on GPU was
#    20 - 100 times slow because of unnecesarily many kernel launches. See below.

#  * Dependencies: LinearAlgebra (stdlib) and KernelAbstractions
#    (`import Pkg; Pkg.add("KernelAbstractions")`). No FFTW/CUDA dependency;
#    CUDA support comes for free when CUDA.jl is loaded by the caller.
# =============================================================================
using LinearAlgebra: I
using KernelAbstractions
using KernelAbstractions: get_backend

"""
    chebnodes(degree, a, b)
 
The `degree + 1` Chebyshev second-kind nodes `cos(jπ/deg)` on `[a, b]`, 
ordered `j = 0, ...,  degree` from b to a.
"""
function chebnodes(degree::Integer, a::Real, b::Real)
    T = float(promote_type(typeof(a), typeof(b)))
    xstd = cos.((0:degree) .* (T(π) / degree))    # nodes in [-1, 1]
    off = T(0.5) * (a + b)
    scl = T(0.5) * (b - a)
    return off .+ scl .* xstd
end
 
"""
    chebcoeffmatrix(degree; T = Float64)
 
The `(degree+1) × (degree+1)` matrix `D` mapping function values at the
Chebyshev nodes of the second kind to first-kind Chebyshev coefficients: `c = D * fvals`.
Domain-independent: a DCT-I with endpoint half-weights folded in.
"""
function chebcoeffmatrix(degree::Integer; T::Type=Float64)
    n = degree
    D = Matrix{T}(undef, n + 1, n + 1)
    for k in 0:n, j in 0:n
        qk = (k == 0 || k == n) ? one(T) : T(2)     # coefficient-index weight
        sj = (j == 0 || j == n) ? T(0.5) : one(T)   # node-index weight
        D[k + 1, j + 1] = (qk / n) * sj * cos(T(k * j) * (T(π) / n))
    end
    return D
end
 
"""
    chebint_ab(coeff, a, b)
 
Indefinite integral of a Chebyshev series on `[a, b]`: coefficients (length
`length(coeff) + 1`) of an antiderivative, scaled by `(b - a) / 2`.
"""
function chebint_ab(coeff::AbstractVector{T}, a::Real, b::Real) where {T}
    L = length(coeff)
    out = zeros(T, L + 1)
    scale = T((b - a) / 2)
    out[2] = coeff[1] * scale                    # T_0 -> T_1
    if L > 1
        out[3] = coeff[2] * scale / 4            # T_1 -> T_2
    end
    @inbounds for j in 3:L                        # 0-based coeff index jj = 2 ... L-1
        jj = j - 1
        cj = coeff[j] * scale
        out[jj + 2] += cj / (2 * (jj + 1))
        out[jj]     -= cj / (2 * (jj - 1))
    end
    return out
end
 
"""
    chebintmatrix(degree, a, b; T = Float64)
 
The `(degree+2) × (degree+1)` matrix `Mint` with `chebint_ab(coeff, a, b) ==
Mint * coeff`, used for the batched/GPU integration path (`Mint * C`).
"""
function chebintmatrix(degree::Integer, a::Real, b::Real; T::Type=Float64)
    L = degree + 1
    basis = Matrix{T}(I, L, L)
    return reduce(hcat, (chebint_ab(view(basis, :, j), T(a), T(b)) for j in 1:L))
end
 
# -----------------------------------------------------------------------------
# Evaluation (Clenshaw)
# -----------------------------------------------------------------------------
 
"""
    chebval_ab(x, coeff, a, b)
 
Evaluate the Chebyshev series resepresented by `coeff` at `x` (scalar or array) on 
`[a, b]` via the Clenshaw recurrence, which is numerically stable and has complexity
linear in the polynomial degree, see 

https://en.wikipedia.org/wiki/Clenshaw_algorithm

This function is thus suitable for one coefficient vectorr, many evaluation points.
"""
function chebval_ab(x, coeff::AbstractVector, a::Real, b::Real)
    z = (2 .* x .- (a + b)) ./ (b - a)
    L = length(coeff)
    d = zero(z)
    dd = zero(z)
    @inbounds for k in L:-1:2
        ck = coeff[k]
        d, dd = (2 .* z .* d .- dd .+ ck), d
    end
    return z .* d .- dd .+ coeff[1]
end
 
# Scalar Clenshaw on one row of a pre-transposed coefficient matrix
# `Ct :: K × L` (envelope k = row k), at the mapped coordinate `zz ∈ [-1, 1]`.
# Plain scalar code: inlines into the KernelAbstractions kernel and compiles
# for both CPU and GPU.
@inline function _clenshaw_row(Ct::AbstractMatrix, k::Integer, zz)
    L = size(Ct, 2)
    d = zero(zz)
    dd = zero(zz)
    @inbounds for j in L:-1:2
        d, dd = muladd(2 * zz, d, Ct[k, j] - dd), d
    end
    @inbounds return muladd(zz, d, Ct[k, 1] - dd)
end
 
"""
    chebval_ab_batched(z, C, a, b)
 
Batched Clenshaw: evaluate envelope `k` (column `C[:, k]`) at point `z[k]`.
`z` length `K`, `C` size `(deg+1) × K`, result length `K`.
"""
function chebval_ab_batched(z::AbstractVector, C::AbstractMatrix, a::Real, b::Real)
    zz = (2 .* z .- (a + b)) ./ (b - a)
    Ct = permutedims(C)
    d = zero(zz)
    dd = zero(zz)
    L = size(Ct, 2)
    @inbounds for k in L:-1:2
        d, dd = (2 .* zz .* d .- dd .+ @view(Ct[:, k])), d
    end
    return zz .* d .- dd .+ @view(Ct[:, 1])
end

"""
    chebval_ab_batched(X::AbstractMatrix, C, a, b)
 
Many points per envelope: evaluate envelope `k` (column `C[:, k]`) at the points
`X[:, k]`. `X` size `M × K`, result `M × K`. The same Clenshaw algorithm as the
one-point method, with the coefficient rows broadcast across the `M` points. Very importantly,
still one broadcast per degree, not per point.
"""
function chebval_ab_batched(X::AbstractMatrix, C::AbstractMatrix, a::Real, b::Real)
    size(X, 2) == size(C, 2) ||
        throw(DimensionMismatch("one envelope per column: size(X,2)=$(size(X,2)), size(C,2)=$(size(C,2))"))
    zz = (2 .* X .- (a + b)) ./ (b - a)
    Ct = permutedims(C)
    d = zero(zz)
    dd = zero(zz)
    L = size(Ct, 2)
    @inbounds for k in L:-1:2
        d, dd = (2 .* zz .* d .- dd .+ transpose(@view(Ct[:, k]))), d
    end
    return zz .* d .- dd .+ transpose(@view(Ct[:, 1]))
end
 
# -----------------------------------------------------------------------------
# Inverse-CDF sampling: one fused Kernel Abstraction
# -----------------------------------------------------------------------------
#
# Thread i draws sample i. `Ct :: K × L` holds antiderivative coefficients,
# one envelope per ROW (transposed so that, at each Clenshaw step j, adjacent
# threads read adjacent memory — coalesced on GPU; 


#### I think numpy serializes the same way

# Each envelope owns `spe`  consecutive samples 
# (spe = length(u) ÷ K): thread i uses row (i-1) ÷ spe + 1.
# spe = length(u) recovers the shared-envelope case, spe = 1 the one-draw-per-
# envelope case, and anything in between is B envelopes with M draws each —
# the mode used by the coverage checks.

# --------------------------------------------------------------------------------
# Old version with broadcasting, where latency due to launching kernels dominates.
#
#     function invert_cdf_batched(CI, a, b, u; iters=...)          # RETIRED
#         Ct = permutedims(CI)
#         lower = fill!(similar(u, T, K), T(a))
#         upper = fill!(similar(u, T, K), T(b))
#         lo = _chebval_cols(lower, Ct, a, b)        # Clenshaw = deg+1 broadcasts
#         Z  = _chebval_cols(upper, Ct, a, b) .- lo
#         for _ in 1:iters                           # 24 sequential iterations
#             mid  = (lower .+ upper) ./ 2                       # launch
#             Fmid = (_chebval_cols(mid, Ct, a, b) .- lo) ./ Z .- u
#             #      deg+1 dependent broadcasts = deg+1 launches
#             upper = ifelse.(Fmid .> 0, mid, upper)             # launch
#             lower = ifelse.(Fmid .> 0, lower, mid)             # launch
#         end
#         return (lower .+ upper) ./ 2
#     end
# --------------------------------------------------------------------------------
 
"""
    default_bisection_iters(T)
 
Interval halvings needed to reach relative machine precision
53 for `Float64`, 24 for `Float32`. More iterations bring nothing.
"""
default_bisection_iters(::Type{T}) where {T} = 1 - exponent(eps(float(real(T))))
 
@kernel function _invertcdf_kernel!(out, @Const(Ct), @Const(u), a, b, iters, spe)
    i = @index(Global)
    @inbounds if i <= length(out)
        T = eltype(out)
        k = (i - 1) ÷ spe + 1                      # envelope owning sample i
        lo = _clenshaw_row(Ct, k, -one(T))         # zz(a) = -1
        Z  = _clenshaw_row(Ct, k,  one(T)) - lo    # zz(b) = +1
        uk = T(u[i])
        ab = a + b
        binv = one(T) / (b - a)
        lower = a
        upper = b
        for _ in 1:iters
            mid = (lower + upper) / 2
            zz = muladd(T(2), mid, -ab) * binv
            F = (_clenshaw_row(Ct, k, zz) - lo) / Z - uk   # CDF(mid) - u, nondecreasing
            if F > zero(T)
                upper = mid
            else
                lower = mid
            end
        end
        out[i] = (lower + upper) / 2
    end
end
 

function _invert_cdf_rows(Ct::AbstractMatrix, a::Real, b::Real, u::AbstractVector, iters::Int)
    backend = get_backend(Ct)
    get_backend(u) == backend ||
        throw(ArgumentError("coefficients and uniforms must be on the same device"))
    spe, r = divrem(length(u), size(Ct,1)) # samples for eahc envelope
    r == 0 || throw(DimensionMismatch("length(u)=$(length(u)) must be a multiple of the number of envelopes $(size(Ct, 1))"))
    T = float(promote_type(eltype(Ct), eltype(u)))
    out = similar(u, T)
    _invertcdf_kernel!(backend)(out, Ct, u, T(a), T(b), iters, spe; ndrange=length(u))
    KernelAbstractions.synchronize(backend)
    return out
end

########################################################################################
############     No longer needed since the batched versions are working    ############
########################################################################################
###"""
###    invert_cdf(ci, a, b, u; iters = default_bisection_iters(eltype(u)))
###
###Draw `length(u)` samples from ONE (unnormalised) CDF with antiderivative
###coefficients `ci`, by bisection at the uniforms `u`. Returns samples in `[a, b]`.
###"""
###function invert_cdf(ci::AbstractVector, a::Real, b::Real, u::AbstractVector;
###                    iters::Integer=default_bisection_iters(eltype(u)))
###    return _invert_cdf_rows(reshape(ci, 1, :), a, b, u, Int(iters))
###end
 
"""
    invert_cdf_batched(CI, a, b, u; iters = default_bisection_iters(eltype(CI)))
 
Batched inverse-CDF: one envelope per column of `CI` (`(deg+2) × K`), and
`length(u) ÷ K` uniforms per envelope. Envelope `k` owns the contiguous block
`u[(k-1)m+1 : km]`, so `vec(U)` of an `m × K` matrix of uniforms is already in
the right order. `length(u) == K` recovers one sample per envelope.
"""
function invert_cdf_batched(CI::AbstractMatrix, a::Real, b::Real, u::AbstractVector;
                            iters::Integer=default_bisection_iters(eltype(CI)))
    length(u) % size(CI, 2) == 0 ||
        throw(DimensionMismatch("uniforms per envelope must be constant: size(CI,2)=$(size(CI,2)), length(u)=$(length(u))"))
    return _invert_cdf_rows(permutedims(CI), a, b, u, Int(iters))
end
 
# -----------------------------------------------------------------------------
# ChebPlan: precomputed operators for a fixed degree and domain
# -----------------------------------------------------------------------------
 
"""
    ChebPlan(a, b; degree = 128, T = Float64)
 
Precomputes the second-kind `nodes`, coefficient matrix `D`, and integration matrix
`Mint` for a fixed `degree` on `[a, b]`. Build once, reuse for every envelope.
Move to GPU by adapting the array fields to `CuArray`, e.g.
`gpu_plan(p) = ChebPlan_on(CuArray, p)` below. This is the equivalent
of static args from JAX in python.
"""
struct ChebPlan{T,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    degree::Int
    a::T
    b::T
    nodes::VT
    D::MT
    Mint::MT
end
 
function ChebPlan(a::Real, b::Real; degree::Integer=128, T::Type=Float64)
    a = T(a)
    b = T(b)
    nodes = collect(chebnodes(degree, a, b))::Vector{T}
    D = chebcoeffmatrix(degree; T=T)
    Mint = chebintmatrix(degree, a, b; T=T)
    return ChebPlan{T,typeof(D),typeof(nodes)}(degree, a, b, nodes, D, Mint)
end
 
"""
    ChebPlan_on(ArrayT, plan)
 
Copy of `plan` with array fields converted by `ArrayT` (e.g. `CuArray`):
`gpu_plan = ChebPlan_on(CuArray, plan)`.
"""
function ChebPlan_on(::Type{AT}, p::ChebPlan) where {AT}
    nodes = AT(p.nodes)
    D = AT(p.D)
    Mint = AT(p.Mint)
    return ChebPlan{eltype(D),typeof(D),typeof(nodes)}(p.degree, p.a, p.b, nodes, D, Mint)
end
 
"""
    chebfit(plan, fvals)
 
Chebyshev coefficients from values at `plan.nodes`. `fvals`: vector (single
envelope) or `(degree+1) × K` matrix (K envelopes) — a single matmul.
"""
chebfit(plan::ChebPlan, fvals::AbstractVecOrMat) = plan.D * fvals
 
"""
    chebintegrate(plan, coeff)
 
Antiderivative coefficients. `coeff`: vector or `(degree+1) × K` matrix.
"""
chebintegrate(plan::ChebPlan, coeff::AbstractVecOrMat) = plan.Mint * coeff

"""
    chebdefinite(CI, a, b)

Per-envelope definite integrals over the intervall [a,b] using ANTIDERIVATIVE 
coefficients (columnts of CI). Split out of `chebintegral` for callers that alreadyy
hold `CI`  (e.g. because they also sample from it) and should not pay the price of
`M_int * C` twice when it's not needed.
"""
function chebdefinite(CI::AbstractMatrix, a::Real, b::Real)
    K = size(CI,2)
    T = float(eltype(CI))
    xb = fill!(similar(CI,T,K),T(b))
    xa = fill!(similar(CI,T,K),T(a))
    return chebval_ab_batched(xb,CI,a,b) .- chebval_ab_batched(xa, CI, a, b)
    
end

 
"""
    chebintegral(plan, coeff)
 
Definite integral over `[plan.a, plan.b]`. Vector `coeff` → scalar; matrix
`(deg+1) × K` → length-`K` vector of per-envelope integrals.
"""
function chebintegral(plan::ChebPlan, coeff::AbstractVector)
    ci = chebintegrate(plan, coeff)
    return chebval_ab(plan.b, ci, plan.a, plan.b) - chebval_ab(plan.a, ci, plan.a, plan.b)
end

chebintegral(plan::ChebPlan, coeff::AbstractMatrix)=
    chebdefinite(chebintegrate(plan,coeff),plan.a,plan.b)


"""
    cheblogq(x, C, Zi, a, b)
 
Normalised log-density of fitted polynomial densities: `log f(x) - log Zi`, with the
value from `chebval_ab_batched(x, C, a, b)` (one point per envelope when `x` is a
vector, many points per envelope when `x` is an `M × K` matrix) and `Zi` the
per-envelope definite integrals, precomputed with `chebdefinite` so nothing is
evaluated twice. Both terms are floored at `floatmin` before the log: fitted densities
can dip slightly negative between nodes in negligible-mass tails, and the floor turns
that into a very negative log rather than a NaN.

### (dan) actually it looks like a 'DomainError', not sure if a JAX equivalent exist. 
### docstrings can be amended later  

Any positive rescaling applied to an
envelope before fitting (e.g. a max-shift) enters value and integral alike and cancels.
"""
function cheblogq(x::AbstractVecOrMat, C::AbstractMatrix, Zi::AbstractVector, a::Real, b::Real)
    T = float(promote_type(eltype(C), eltype(x)))
    V = chebval_ab_batched(x, C, a, b)
    Z = x isa AbstractVector ? Zi : reshape(Zi, 1, :)
    return log.(max.(V, floatmin(T))) .- log.(max.(Z, floatmin(T)))
end

########################################################################################
############     No longer needed since the batched versions are working    ############
########################################################################################
#"""
#    chebsample(plan, fvals::AbstractVector, u; iters...)
# 
#Draw `length(u)` samples from the (unnormalised) density with values `fvals`
#at `plan.nodes`. Single envelope, many samples — the coordinate-1 case.
#"""
#function chebsample(plan::ChebPlan, fvals::AbstractVector, u::AbstractVector;
#                    iters::Integer=default_bisection_iters(eltype(u)))
#    ci = chebintegrate(plan, chebfit(plan, fvals))
#    return invert_cdf(ci, plan.a, plan.b, u; iters=iters)
#end
 
"""
    chebsample(plan, F::AbstractMatrix, u::AbstractVector; iters...)
 
Batched: one envelope per column of `F` (`(degree+1) × K`), one uniform per
envelope, one sample per envelope — the coordinate-`i` (`i ≥ 2`) case.
"""
function chebsample(plan::ChebPlan, F::AbstractMatrix, u::AbstractVector;
                    iters::Integer=default_bisection_iters(eltype(u)))
    size(F, 2) == length(u) ||
        throw(DimensionMismatch("one uniform per envelope: size(F,2)=$(size(F,2)), length(u)=$(length(u))"))
    CI = chebintegrate(plan, chebfit(plan, F))
    return invert_cdf_batched(CI, plan.a, plan.b, u; iters=iters)
end


"""
    chebsample(plan, F::AbstractMatrix, U::AbstractMatrix; iters...)
 
Grouped: one envelope per column of `F` (`(degree+1) × B`), `M` uniforms per
envelope (`U` is `M × B`, column `b` for envelope `b`), returns `M × B` samples.
The per-head coverage check case: many draws from each of many envelopes.
"""
function chebsample(plan::ChebPlan, F::AbstractMatrix, U::AbstractMatrix;
                    iters::Integer=default_bisection_iters(eltype(U)))
    size(F, 2) == size(U, 2) ||
        throw(DimensionMismatch("one envelope per column: size(F,2)=$(size(F,2)), size(U,2)=$(size(U,2))"))
    CI = chebintegrate(plan, chebfit(plan, F))
    return reshape(invert_cdf_batched(CI, plan.a, plan.b, vec(U); iters=iters), size(U))
end