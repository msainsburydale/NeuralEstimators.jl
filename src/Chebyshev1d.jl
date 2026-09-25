# Fast approximation (in the basis of Chebyshev polynomials), integration, and inverse-CDF sampling for 1D densities. 
# The code is a Julia port of chebyshev_utils.py from
# https://github.com/danleonte/Simulation-based-inference-via-telescoping-ratio-estimation-for-trawl-processes-paper/blob/main/src/utils/chebyshev_utils.py
# and further details can be found at
# https://arxiv.org/abs/2510.04042, https://arxiv.org/abs/1307.1223
# note the second paper also implements the 2D case.

# Various Julia libraries already support efficiently approximating 1D functions, i.e. f: R -> R, by Chebyshev polynomials. 
# This scripts targets posterior sampling for the TelescopingRatioEstimator, which requires sequentally
# approximating batches of 1D (conditional) denisities. For efficiency, the implementation has to
# i)  be GPU-friendly, hence non-adaptive, i.e., the degree of the approximating polynomial can not change frequently
# ii) allow for constructing approximate CDFs in parallel, hence also generate independent (approximate) samples (from multiple distributions, in parallel.)
#
# It is crucial to use vectorized bisection and not gradient-based methods for sampling by inversion.
# Indeed, gradient-based methods require different numbers of evaluations, thus preventing effective parallelism;
# they can also get stuck, as opposed to bisection. By comparison,
# ApproxFun, FastChebInterp and FastTransforms.jl are adaptive or CPU oriented, or both. 
#
# In this implementation, we avoid directly using FFT. Although computing the coefficients of an approximating polynomial
# of degree n can be done in O(n log n) with FFT, rather than O(n^2) in this implementation, empirically we note that 
# we only require relatively small n values, say 128 for most cases and 256 for most pathological densities. Further,
# the dominant cost is given by the inverse-CDF sampling from the approximating polynomials, and not coefficient fitting;
# nevertheless, using FFTs for coefficient fittings is retained for future development. The main use case would be 
# for very concentrated densities. Also note that the FFT (specifically) DCT machinery seems to require quite different CPU / GPU treatment.

# IMPORTANT: A previous version used a broadcast-based bisection, which on GPU was
# 20 - 100 times slow because of unnecesarily many kernel launches. See below.

# Dependencies: LinearAlgebra (stdlib) and KernelAbstractions
# (`import Pkg; Pkg.add("KernelAbstractions")`). No FFTW/CUDA dependency.
# =============================================================================
using LinearAlgebra: I
using KernelAbstractions
using KernelAbstractions: get_backend

"""
    chebnodes(degree, a, b)
 
Returns the `degree + 1` Chebyshev second-kind (Lobatto) nodes on `[a, b]`.
These are obtained by mapping 

`cos(jπ/deg)`,       `j = 0, ...,  degree`

from [`-1, 1`] -> `[a,b]`. The nodes (or knots) are ordered decreasingly from b to a.
"""
function chebnodes(degree::Integer, a::Real, b::Real)
    T = float(promote_type(typeof(a), typeof(b)))
    xstd = cos.((0:degree) .* (T(π) / degree))    # nodes in [-1, 1]
    # next map to [a,b]
    off = T(0.5) * (a + b)
    scl = T(0.5) * (b - a)
    return off .+ scl .* xstd
end

"""
    chebcoeffmatrix(degree; T = Float64)
 
Constructs the `(degree+1) × (degree+1)` matrix `D` that inputs the target function evaluated at the
Chebyshev knots of the second kind to polynomial coefficients in the basis of first-kind Chebyshev 
polynomials: 

`coeff = D * fvals`.

This transformation does not depend on the bounds `[a,b]`. Thus, given a btch of 'fvals' values, i.e.,
multiple target functions evaluated at the Chebyshev knots of the second kind, the result
can be obtained through a single matrix multiplication.

"""
function chebcoeffmatrix(degree::Integer; T::Type = Float64)
    n = degree
    D = Matrix{T}(undef, n + 1, n + 1)
    for k = 0:n, j = 0:n
        qk = (k == 0 || k == n) ? one(T) : T(2)     # coefficient weight
        sj = (j == 0 || j == n) ? T(0.5) : one(T)   # (endpoint) node weight
        D[k + 1, j + 1] = (qk / n) * sj * cos(T(k * j) * (T(π) / n))
    end
    return D
end

"""
    chebint_ab(coeff, a, b)
 
Returns the Chebyshev coefficients of the indefinite integral of a Chebyshev series on `[a, b]`.
The length of coefficients increases by 1.

By default, the integration constant is set to 0.
"""
function chebint_ab(coeff::AbstractVector{T}, a::Real, b::Real) where {T}
    L = length(coeff)
    out = zeros(T, L + 1)
    scale = T((b - a) / 2)                       # account for transformation from `[-1,1]` to `[a,b]`
    out[2] = coeff[1] * scale
    if L > 1
        out[3] = coeff[2] * scale / 4
    end
    # see   Differentiation and integration in https://en.wikipedia.org/wiki/Chebyshev_polynomials for the integration identity for T_n
    @inbounds for j = 3:L
        k = j - 1
        cj = coeff[j] * scale
        out[k + 2] += cj / (2 * (k + 1))
        out[k] -= cj / (2 * (k - 1))
    end
    return out
end

"""
    chebintmatrix(degree, a, b; T = Float64)
 
Construct the `(degree+2) × (degree+1)` matrix `Mint` which maps Chebyshev coefficients 
to these of the antiverivative:

`chebint_ab(coeff, a, b) == Mint * coeff`,

through a matrix multiplication; useful for batched integration on both CPU and GPU.
"""
function chebintmatrix(degree::Integer, a::Real, b::Real; T::Type = Float64)
    L = degree + 1
    basis = Matrix{T}(I, L, L)
    return reduce(hcat, (chebint_ab(view(basis, :, j), T(a), T(b)) for j = 1:L))
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
"""
function chebval_ab(x, coeff::AbstractVector, a::Real, b::Real)
    z = (2 .* x .- (a + b)) ./ (b - a)
    L = length(coeff)
    d = zero(z)
    dd = zero(z)
    @inbounds for k = L:-1:2
        ck = coeff[k]
        d, dd = (2 .* z .* d .- dd .+ ck), d
    end
    return z .* d .- dd .+ coeff[1]
end

# Scalar Clenshaw on one row of a pre-transposed coefficient matrix
# `Ct`. Recurrence uses scalars, hence can be used efficiently in the
# KernelAbstractions kernel used for inverse CDF sampling for both CPU and GPU.
@inline function _clenshaw_row(Ct::AbstractMatrix, k::Integer, zz)
    L = size(Ct, 2)
    d = zero(zz)
    dd = zero(zz)
    @inbounds for j = L:-1:2
        d, dd = muladd(2 * zz, d, Ct[k, j] - dd), d
    end
    @inbounds return muladd(zz, d, Ct[k, 1] - dd)
end

"""
    chebval_ab_batched(x, C, a, b)
 
Batched Clenshaw: evaluates multiple Chebyshev series, each of which is being evaluated at
one point. This is particularly helpful for the TRE posterior sampling, where we require
efficiently generating samples from multiple densities, in parallel. Specifically, we will
evaluate the (approximate) CDF like this during bisection.

Columns `C[:, k]` give polynomial coefficients for the kth approximation, vector x gives the points.
If `C` has size `(degree + 1) × K`, then `x` and the returned vector both have length `K`.
"""
function chebval_ab_batched(x::AbstractVector, C::AbstractMatrix, a::Real, b::Real)
    zz = (2 .* x .- (a + b)) ./ (b - a)
    Ct = permutedims(C)
    d = zero(zz)
    dd = zero(zz)
    L = size(Ct, 2)
    @inbounds for k = L:-1:2
        d, dd = (2 .* zz .* d .- dd .+ @view(Ct[:, k])), d
    end
    return zz .* d .- dd .+ @view(Ct[:, 1])
end

"""
    chebval_ab_batched(X::AbstractMatrix, C, a, b)
 
Batched Clenshaw evaluation at multiple points per series, see above for comparison.
Columns `C[:, k]` give polynomial coefficients, `X[:, k]` gives the points to evaluate this polynomial at.
If `X` is `M × K`, the result is `M × K`. The coefficient row broadcast across the `M` points.
Importantly, the reucrrency is evaluated simulatenously for both points and polynomials,
with coefficients broadcasting over the `M` points.
"""
function chebval_ab_batched(X::AbstractMatrix, C::AbstractMatrix, a::Real, b::Real)
    size(X, 2) == size(C, 2) ||
        throw(DimensionMismatch("one envelope per column: size(X,2)=$(size(X,2)), size(C,2)=$(size(C,2))"))
    zz = (2 .* X .- (a + b)) ./ (b - a)
    Ct = permutedims(C)
    d = zero(zz)
    dd = zero(zz)
    L = size(Ct, 2)
    @inbounds for k = L:-1:2
        d, dd = (2 .* zz .* d .- dd .+ transpose(@view(Ct[:, k]))), d
    end
    return zz .* d .- dd .+ transpose(@view(Ct[:, 1]))
end

# -----------------------------------------------------------------------------
# Inverse-CDF sampling: one fused Kernel Abstraction
# -----------------------------------------------------------------------------
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
 
Number of interval halvings needed to reach relative machine precision
53 for `Float64`, 24 for `Float32`. More iterations bring nothing.
"""
default_bisection_iters(::Type{T}) where {T} = 1 - exponent(eps(float(real(T))))

@kernel function _invertcdf_kernel!(out, @Const(Ct), @Const(u), a, b, iters, spe)
    i = @index(Global)
    @inbounds if i <= length(out)
        T = eltype(out)
        # target density i
        k = (i - 1) ÷ spe + 1
        lo = _clenshaw_row(Ct, k, -one(T))         # antiderivative at a
        Z = _clenshaw_row(Ct, k, one(T)) - lo    # total mass over interval [a,b], used as normalization constant
        uk = T(u[i])
        ab = a + b
        binv = one(T) / (b - a)
        lower = a
        upper = b
        for _ = 1:iters
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
    spe, r = divrem(length(u), size(Ct, 1)) # samples for each target density
    r == 0 || throw(DimensionMismatch("length(u)=$(length(u)) must be a multiple of the number of envelopes $(size(Ct, 1))"))
    T = float(promote_type(eltype(Ct), eltype(u)))
    out = similar(u, T)
    _invertcdf_kernel!(backend)(out, Ct, u, T(a), T(b), iters, spe; ndrange = length(u))
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
 
Batched inverse-CDF samplig: generates independent samples by inverting the (batched) CDF(s);

As before, columns contain coefficients corresponding to one target density.
The special case `length(u) == K` recovers one sample per density.

The functions (densities) are assumed to be non-negative on `[a,b]`.
"""
function invert_cdf_batched(CI::AbstractMatrix, a::Real, b::Real, u::AbstractVector;
    iters::Integer = default_bisection_iters(eltype(CI)))
    length(u) % size(CI, 2) == 0 ||
        throw(DimensionMismatch("uniforms per envelope must be constant: size(CI,2)=$(size(CI,2)), length(u)=$(length(u))"))
    return _invert_cdf_rows(permutedims(CI), a, b, u, Int(iters))
end

# -----------------------------------------------------------------------------
# ChebPlan: precomputed operators for a fixed degree and domain
# -----------------------------------------------------------------------------

"""
    ChebPlan(a, b; degree = 128, T = Float64)
 
Precompute the quantities required to repeatedly construct Chebyshev
approximations of fixed `degree` on `[a,b]`.

We store the Chebyshev nodes, value-to-coefficient matrix `D`,
and integration matrix `Mint`. These quantities depend only on the
polynomial degree and interval, and can therefore be constructed once and
reused for every density. Comparing to the original jax implementation,
this acts as a static argument.

Subsequent fitting and integration reduces to matrix multiplications.
"""
struct ChebPlan{T, MT <: AbstractMatrix{T}, VT <: AbstractVector{T}}
    degree::Int
    a::T
    b::T
    nodes::VT
    D::MT
    Mint::MT
end

function ChebPlan(a::Real, b::Real; degree::Integer = 128, T::Type = Float64)
    a = T(a)
    b = T(b)
    nodes = collect(chebnodes(degree, a, b))::Vector{T}
    D = chebcoeffmatrix(degree; T = T)
    Mint = chebintmatrix(degree, a, b; T = T)
    return ChebPlan{T, typeof(D), typeof(nodes)}(degree, a, b, nodes, D, Mint)
end

"""
    ChebPlan_on(ArrayT, plan)
 
Copy of `Chebplan` with array fields converted by `ArrayT` (e.g. `CuArray`):
`gpu_plan = ChebPlan_on(CuArray, plan)`. This keeps the implementation independent
of GPU.
"""
function ChebPlan_on(::Type{AT}, p::ChebPlan) where {AT}
    nodes = AT(p.nodes)
    D = AT(p.D)
    Mint = AT(p.Mint)
    return ChebPlan{eltype(D), typeof(D), typeof(nodes)}(p.degree, p.a, p.b, nodes, D, Mint)
end

"""
    chebfit(plan, fvals)
 
Fit the Chebyshev coefficients of the best approximating polynomials from the function evaluated 
at `plan.nodes`. 

`fvals` must be either a vector, representing one function (density) evaluations,
or a `(degree+1) × K` matrix stacked columnwise for K different functions.
"""
chebfit(plan::ChebPlan, fvals::AbstractVecOrMat) = plan.D * fvals

"""
    chebintegrate(plan, coeff)
 
Antiderivative coefficients. `coeff`: vector or `(degree+1) × K` matrix, as above.
"""
chebintegrate(plan::ChebPlan, coeff::AbstractVecOrMat) = plan.Mint * coeff

"""
    chebdefinite(CI, a, b)

Compute definite integrals over the interval [a,b].
"""
function chebdefinite(CI::AbstractMatrix, a::Real, b::Real)
    K = size(CI, 2)
    T = float(eltype(CI))
    xb = fill!(similar(CI, T, K), T(b))
    xa = fill!(similar(CI, T, K), T(a))
    return chebval_ab_batched(xb, CI, a, b) .- chebval_ab_batched(xa, CI, a, b)
end

"""
    chebintegral(plan, coeff)
 
Definite integral over `[plan.a, plan.b]`. 

For a coefficient vector, it returns scalar; for a matrix `(deg+1) × K`, it returns a length-`K` vector of integrals.
"""
function chebintegral(plan::ChebPlan, coeff::AbstractVector)
    ci = chebintegrate(plan, coeff)
    return chebval_ab(plan.b, ci, plan.a, plan.b) - chebval_ab(plan.a, ci, plan.a, plan.b)
end

chebintegral(plan::ChebPlan, coeff::AbstractMatrix) =
    chebdefinite(chebintegrate(plan, coeff), plan.a, plan.b)

"""
    cheblogq(x, C, Zi, a, b)
 
Returns the normalised log-density of fitted polynomial densities: 

`log f(x) - log Zi`, 

where `C` contains polynomial coefficients, as above, `Zi` the definite intgrals.
For vector `x`, one point is evaluated per density. For an `M × K` matrix,
column `k` contains `M` points evaluated under density `k`.

Note that any rescaling by a positive constant before passing the input to cheblogq
cancels out.
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
 
Fit approximations by Chebyshev polynomials and sample from multiple 1D target densities
in parallel.

The columns of F contain the target densities evaluated at the plan nodes. 
`u` contains the uniform random samples.

Returns one sample per target density.
"""
function chebsample(plan::ChebPlan, F::AbstractMatrix, u::AbstractVector;
    iters::Integer = default_bisection_iters(eltype(u)))
    size(F, 2) == length(u) ||
        throw(DimensionMismatch("one uniform per envelope: size(F,2)=$(size(F,2)), length(u)=$(length(u))"))
    CI = chebintegrate(plan, chebfit(plan, F))
    return invert_cdf_batched(CI, plan.a, plan.b, u; iters = iters)
end

"""
    chebsample(plan, F::AbstractMatrix, U::AbstractMatrix; iters...)
 
Fit approximations by Chebyshev polynomials and sample from multiple 1D target densities
in parallel. Matrix version of teh above.
"""
function chebsample(plan::ChebPlan, F::AbstractMatrix, U::AbstractMatrix;
    iters::Integer = default_bisection_iters(eltype(U)))
    size(F, 2) == size(U, 2) ||
        throw(DimensionMismatch("one envelope per column: size(F,2)=$(size(F,2)), size(U,2)=$(size(U,2))"))
    CI = chebintegrate(plan, chebfit(plan, F))
    return reshape(invert_cdf_batched(CI, plan.a, plan.b, vec(U); iters = iters), size(U))
end
