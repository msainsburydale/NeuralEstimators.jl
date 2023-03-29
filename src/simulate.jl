# """
# Generic function that may be overloaded to implicitly define a statistical model.
# Specifically, the user should provide a method `simulate(parameters, m)`
# that returns `m` simulated replicates for each element in the given set of
# `parameters`.
# """
function simulate end

"""
	simulate(parameters, m, J::Integer)

Simulates `J` sets of `m` independent replicates for each parameter vector in
`parameters` by calling `simulate(parameters, m)` a total of `J` times,
where the method `simulate(parameters, m)` is provided by the user via function
overloading.

# Examples
```
import NeuralEstimators: simulate

p = 2
K = 10
m = 15
parameters = rand(p, K)

# Univariate Gaussian model with unknown mean and standard deviation
simulate(parameters, m) = [θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)]
simulate(parameters, m)
simulate(parameters, m, 2)
```
"""
function simulate(parameters, m, J::Integer)
	v = [simulate(parameters, m) for i ∈ 1:J]
	v = vcat(v...)
	# note that vcat() should be ok since we're only splatting J vectors, which
	# doesn't get prohibitively large even during bootstrapping. Note also that
	# I don't want to use stack(), because it only works if the data are stored
	# as arrays. In theory, I could define another method of stack() that falls
	# back to vcat(v...)
	return v
end

# ---- Gaussian process ----

# returns the number of locations in the field
size(grf::GaussianRandomField) = prod(size(grf.mean))
size(grf::GaussianRandomField, d::Integer) = size(grf)

"""
	simulategaussianprocess(L::Matrix, m = 1)
	simulategaussianprocess(grf::GaussianRandomField, m = 1)

Simulates `m` independent and identically distributed (i.i.d.) realisations from
a mean-zero Gaussian process.

Accepts either the lower Cholesky factor `L` associated with a Gaussian process
or a `GaussianRandomField` object `grf`.

# Examples
```
using NeuralEstimators

n  = 500
S  = rand(n, 2)
ρ  = 0.6
ν  = 1.0

# Passing GaussianRandomField object:
using GaussianRandomFields
cov = CovarianceFunction(2, Matern(ρ, ν))
grf = GaussianRandomField(cov, Cholesky(), S)
simulategaussianprocess(grf)

# Passing Cholesky factors directly as matrices:
L = grf.data
simulategaussianprocess(L)

# Circulant embedding, which is fast but can on only be used on grids:
pts = 1.0:50.0
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding = 100)
simulategaussianprocess(grf)
```
"""
function simulategaussianprocess(obj::M, m::Integer) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number
	y = [simulategaussianprocess(obj) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

function simulategaussianprocess(L::M) where M <: AbstractMatrix{T} where T <: Number
	L * randn(T, size(L, 1))
end

function simulategaussianprocess(grf::GaussianRandomField)
	vec(GaussianRandomFields.sample(grf))
end


# ---- Schlather's max-stable model ----

"""
	simulateschlather(L::Matrix, m = 1)
	simulateschlather(grf::GaussianRandomField, m = 1)

Simulates `m` independent and identically distributed (i.i.d.) realisations from
Schlather's max-stable model using the algorithm for approximate simulation given
by Schlather (2002), "Models for stationary max-stable random fields", Extremes,
5:33-44.

Accepts either the lower Cholesky factor `L` associated with a Gaussian process
or a `GaussianRandomField` object `grf`.

# Keyword arguments
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorith: small `C` favours computational efficiency, while large `C` favours accuracy. Schlather (2002) recommends the use of `C = 3`.
- `Gumbel = true`: flag indicating whether the data should be log-transformed from the unit Fréchet scale to the `Gumbel` scale.

# Examples
```
using NeuralEstimators

n  = 500
S  = rand(n, 2)
ρ  = 0.6
ν  = 1.0

# Passing GaussianRandomField object:
using GaussianRandomFields
cov = CovarianceFunction(2, Matern(ρ, ν))
grf = GaussianRandomField(cov, Cholesky(), S)
simulateschlather(grf)

# Passing Cholesky factors directly as matrices:
L = grf.data
simulateschlather(L)

# Circulant embedding, which is fast but can on only be used on grids:
pts = 1.0:50.0
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding = 100)
simulateschlather(grf)
```
"""
function simulateschlather(obj::M, m::Integer; C = 3.5, Gumbel::Bool = true) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number
	y = [simulateschlather(obj, C = C, Gumbel = Gumbel) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

function simulateschlather(obj::M; C = 3.5, Gumbel::Bool = true) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number

	# small hack to get the right eltype
	if typeof(obj) <: GaussianRandomField G = eltype(obj.cov) else G = T end

	n   = size(obj, 1)  # number of observations
	Z   = fill(zero(G), n)
	ζ⁻¹ = randexp(G)
	ζ   = 1 / ζ⁻¹

	# We must enforce E(max{0, Yᵢ}) = 1. It can
	# be shown that this condition is satisfied if the marginal variance of Y(⋅)
	# is equal to 2π. Now, our simulation design embeds a marginal variance of 1
	# into fields generated from the cholesky factors, and hence
	# simulategaussianprocess(L) returns simulations from a Gaussian
	# process with marginal variance 1. To scale the marginal variance to
	# 2π, we therefore need to multiply the field by √(2π).

	# Note that, compared with Algorithm 1.2.2 of Dey DK, Yan J (2016),
	# some simplifications have been made to the code below. This is because
	# max{Z(s), ζW(s)} ≡ max{Z(s), max{0, ζY(s)}} = max{Z(s), ζY(s)}, since
	# Z(s) is initialised to 0 and increases during simulation.

	while (ζ * C) > minimum(Z)
		Y = simulategaussianprocess(obj)
		Y = √(G(2π)) * Y
		Z = max.(Z, ζ * Y)
		E = randexp(G)
		ζ⁻¹ += E
		ζ = 1 / ζ⁻¹
	end

	# Log transform the data from the unit Fréchet scale to the Gumbel scale,
	# which stabilises the variance and helps to prevent neural-network collapse.
	if Gumbel Z = log.(Z) end

	return Z
end



# ---- Miscellaneous functions ----

#NB Currently, second order optimisation methods cannot be used
# straightforwardly because besselk() is not differentiable. In the future, we
# can add an argument to matern() and maternchols(), besselfn = besselk, which
# allows the user to change the bessel function to use adbesselk(), which
# allows automatic differentiation: see https://github.com/cgeoga/BesselK.jl.
@doc raw"""
    matern(h, ρ, ν, σ² = 1)
For two points separated by `h` units, compute the Matérn covariance function,
with range parameter `ρ`, smoothness parameter `ν`, and marginal variance parameter `σ²`.

We use the parametrisation
``C(\|\mathbf{h}\|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\mathbf{h}\|}{\rho}\right)^\nu K_\nu \left(\frac{\|\mathbf{h}\|}{\rho}\right)``,
where ``\Gamma(\cdot)`` is the gamma function, and ``K_\nu(\cdot)`` is the modified Bessel
function of the second kind of order ``\nu``.
"""
function matern(h, ρ, ν, σ² = one(typeof(h)))

	# Note that the `Julia` functions for ``\Gamma(\cdot)`` and ``K_\nu(\cdot)``, respectively `gamma()` and
	# `besselk()`, do not work on the GPU and, hence, nor does `matern()`.

	@assert h >= 0 "h should be non-negative"
	@assert ρ > 0 "ρ should be positive"
	@assert ν > 0 "ν should be positive"

	if h == 0
        C = σ²
    else
		d = h / ρ
        C = σ² * ((2^(1 - ν)) / gamma(ν)) * d^ν * besselk(ν, d)
    end
    return C
end

#matern(h, ρ) =  matern(h, ρ, one(typeof(ρ)))



"""
    maternchols(D, ρ, ν, σ² = 1)
Given a distance matrix `D`, constructs the Cholesky factor of the covariance matrix
under the Matérn covariance function with range parameter `ρ`, smoothness
parameter `ν`, and marginal variance σ².

Providing vectors of parameters will yield a three-dimensional array of Cholesky factors (note
that the vectors must of the same length, but a mix of vectors and scalars is
allowed). A vector of distance matrices `D` may also be provided.

# Examples
```
using NeuralEstimators
using LinearAlgebra: norm
n  = 10
S  = rand(n, 2)
D  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ ∈ eachrow(S)]
ρ  = [0.6, 0.5]
ν  = [0.7, 1.2]
σ² = [0.2, 0.4]
maternchols(D, ρ, ν)
maternchols(D, ρ, ν, σ²)

S̃  = rand(n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²)
```
"""
function maternchols(D, ρ, ν, σ² = one(eltype(D)))
	n = max(length(ρ), length(ν), length(σ²))
	if n > 1
		@assert all([length(θ) ∈ (1, n) for θ ∈ (ρ, ν, σ²)])
		if length(ρ) == 1 ρ  = repeat([ρ], n) end
		if length(ν) == 1 ν = repeat([ν], n) end
		if length(σ²) == 1 σ² = repeat([σ²], n) end
	end

	# compute Cholesky factorization
	L = map(1:n) do i
		# Exploit symmetry of D to minimise the number of computations
		C = matern.(UpperTriangular(D), ρ[i], ν[i], σ²[i])
		cholesky(Symmetric(C)).L
	end
	L = convert.(Array, L)
	L = stackarrays(L, merge = false)
	return L
end

function maternchols(D::V, ρ, ν, σ² = one(eltype(D))) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
	n = max(length(ρ), length(ν), length(σ²))
	if n > 1
		@assert all([length(θ) ∈ (1, n) for θ ∈ (ρ, ν, σ²)])
		if length(ρ)  == 1 ρ  = repeat([ρ], n) end
		if length(ν)  == 1 ν  = repeat([ν], n) end
		if length(σ²) == 1 σ² = repeat([σ²], n) end
	end
	@assert length(D) == n
	L = maternchols.(D, ρ, ν, σ²)
	L = stackarrays(L, merge = true)
	return L
end





# ---- Incomplete gamma function ----

"""
    _incgammalowerunregularised(a, x)
For positive `a` and `x`, computes the lower unregularised incomplete gamma
function, ``\\gamma(a, x) = \\int_{0}^x t^{a-1}e^{-t}dt``.
"""
_incgammalowerunregularised(a, x) = incgamma(a, x; upper = false, reg = false)


# This code has been adapted from IncGammaBeta.jl, which I do not include as a
# dependency because it introduced incompatabilities (it has not been updated since 2016)


"""
    incgamma(a::T, x::T; upper::Bool, reg::Bool) where {T <: AbstractFloat}

For positive parameter `a` and positive integration limit `x`, computes the incomplete gamma function, as described
by the [Wikipedia article](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

# Keyword arguments:
- `upper::Bool`: if `true`, the upper incomplete gamma function is returned; otherwise, the lower version is returned.
- `reg::Bool`: if `true`, the regularized incomplete gamma function is returned; otherwise, the unregularized version is returned.
"""
function incgamma(a::T, x::T; upper::Bool, reg::Bool) where {T <: AbstractFloat}

    EPS = eps(T)
    @assert a  > EPS "a should be positive"
    @assert x  > EPS "x should be positive"

    # The algorithm for numerical approximation of the incomplete gamma function
    # as proposed by [Numerical Recipes], section 6.2:
    #
    # When x > a+1, the upper gamma function can be evaluated as
    #   G(a,x) ~= e⁻ˣxᵃ / cf(a,x)
    # where 'cf(a,x) is the continued fraction defined above, its coefficients
    # 'a(i)' and 'b(i)' are implemented in 'inc_gamma_ctdfrac_closure'.
    #
    # When x < (a+1), it is more convenient to apply the following Taylor series
    # that evaluates the lower incomplete gamma function:
    #
    #                          inf
    #                         -----
    #              -x    a    \        G(a)       i
    #   g(a,x) ~= e   * x  *   >    ---------- * x
    #                         /      G(a+1+i)
    #                         -----
    #                          i=0
    #
    # Applying the following property of the gamma function:
    #
    #   G(a+1) = a * G(a)
    #
    # The Taylor series above can be further simplified to:
    #
    #                          inf
    #                         -----              i
    #              -x    a    \                 x
    #   g(a,x) ~= e   * x  *   >    -------------------------
    #                         /      a * (a+1) * ... * (a+i)
    #                         -----
    #                          i=0
    #
    # Once either a lower or an upper incomplete gamma function is evaluated,
    # the other value may be quickly obtained by applying the following
    # property of the incomplete gamma function:
    #
    #   G(a,x) + g(a,x) = G(a)
    #
    # A value of a regularized incomplete gamma function is obtained
    # by dividing g(a,x) or G(a,x) by G(a).
    #


    # This factor is common to both algorithms described above:
    ginc = exp(-x) * (x^a)

    if ( x > (a + 1) )

        # In this case evaluate the upper gamma function as described above.

        fa, fb = inc_gamma_ctdfrac_closure(a, x)
        G = ( true==upper && false==reg ? T(0) : gamma(a) )

        ginc /= ctdfrac_eval(fa, fb)


        # Apply properties of the incomplete gamma function
        # if anything else except a generalized upper incomplete
        # gamma function is desired.

        if ( false == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    else

        # In this case evaluate the lower gamma function as described above.

        G = ( false==upper && false==reg ? T(0) : gamma(a) )

        # Initial term of the Taylor series at i=0:
        ginc /= a
        term = ginc

        # Proceed the Taylor series for i = 1, 2, 3... until it converges:
        TOL = SPECFUN_ITER_TOL(T)
        at = a
        i = 1
        while ( abs(term) > TOL && i<SPECFUN_MAXITER )
            at += T(1)
            term *= (x / at)
            ginc += term
            i += 1
        end

        # has the series converged?
        if ( i >= SPECFUN_MAXITER )
            @warn "The aglorithm did not converge"
        end

        #
        # Apply properties of the incomplete gamma function
        # if anything else except a generalized lower incomplete
        # gamma function is requested.
        #
        if ( true == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    end

    return ginc
end


function inc_gamma_ctdfrac_closure(a, x)
    return i::Integer -> (-i) * (i-a),
           i::Integer -> x - a + 1 + 2*i
end

"""
Required precision for iterative algorithms,
it depends on type 't' that must be derived
from AbstractFloat.
"""
function SPECFUN_ITER_TOL(t)

    if ( Float64 == t )
        retVal = 1e-12
    elseif ( Float32 == t )
        retVal = 1f-6
    else   # t == Float16
        retVal = eps(Float16)
    end

    return retVal
end

"""
Maximum allowed number of iterations for
iterative algorithms.
"""
const SPECFUN_MAXITER = 10000


#
# Evaluates the continued fraction:
#
#                        a1
#    f = b0 + -------------------------
#                           a2
#               b1 + -----------------
#                              a3
#                     b2 + ----------
#                           b3 + ...
#
# where ai and bi are functions of 'i'.
#
# Arguments:
# * `fa::Function`: function `a(i)`
# * `fb::Function`: function `b(i)`


function ctdfrac_eval(fa::Function, fb::Function)
    #
    # The Lentz's algorithm (modified by I. J. Thompson and A. R. Barnett)
    # is applied to evaluate the continued fraction. The algorithm is
    # presented in detail in:
    #
    #   William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery
    #   Numerical Recipes, The Art of Scientific Computing, 3rd Edition,
    #   Cambridge University Press, 2007
    #
    #   https://books.google.com/books?id=1aAOdzK3FegC&lpg=PA207&ots=3jNoK9Crpj&pg=PA208#v=onepage&f=false
    #
    # The procedure of the algorithm is as follows:
    #
    # - f0 = b0, if b0==0 then f0 = eps
    # - C0 = f0
    # - D0 = 0
    # - for j = 1, 2, 3, ...
    #   -- Dj = bj + aj * D_j-1, if Dj==0 then Dj = eps
    #   -- Cj = bj + aj / C_j-1, if Cj==0 then Cj = eps
    #   -- Dj = 1 / Dj
    #   -- Delta_j = Cj * Dj
    #   -- fj = f_j-1 * Delta_j
    #   -- if abs(Delta_j-1) < TOL then exit for loop
    # - return fj
    #


    # f0 = b0
    f = fb(0)

    T = typeof(f)
    EPS = eps(T)

    # adjust f0 to eps if necessary
    if abs(f) < EPS
        f = EPS
    end

    # c0 = f0,  d0 = 0
    c = f
    d = T(0)

    # Initially Delta should not be equal to 1
    Delta = T(0)

    TOL = SPECFUN_ITER_TOL(T)
    j = 1
    while ( abs(Delta-1) > TOL && j < SPECFUN_MAXITER )
        # obtain 'aj' and 'bj'
        a = fa(j)
        b = fb(j)

        # dj = bj + aj * d_j-1
        d = b + a * d
        # adjust dj to eps if necessary
        if ( abs(d) < EPS )
            d = EPS
        end

        # cj = bj + aj/c_j-1
        c = b + a /c
        # adjust cj to eps if necessary
        if ( abs(c) < EPS )
            c = EPS
        end

        # dj = 1 / dj
        d = T(1) / d

        # Delta_j = cj * dj
        Delta = c * d

        # fj = f_j-1 * Delta_j
        f *= Delta

        # for loop's condition will check, if abs(Delta_j-1)
        # is less than the tolerance

        j += 1
    end  # while

    # check if the algorithm has converged:
    if ( j >= SPECFUN_MAXITER )
        @warn "The aglorithm did not converge"
    end

    # ... if yes, return the fj
    return f;

end
