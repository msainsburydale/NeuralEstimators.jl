"""
Generic function that may be overloaded to implicitly define a statistical model.
Specifically, the user should provide a method `simulate(parameters, m)`
that returns `m` simulated replicates for each element in the given set of
`parameters`.
"""
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
function simulate(parameters::P, m, J::Integer) where P <: Union{Matrix, ParameterConfigurations}
	v = [simulate(parameters, m) for i ∈ 1:J]
	if typeof(v[1]) <: Tuple
		z = vcat([v[i][1] for i ∈ eachindex(v)]...)
		x = vcat([v[i][2] for i ∈ eachindex(v)]...)
		v = (z, x)
	else
		v = vcat(v...)
	end
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
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorithm: small `C` favours computational efficiency, while large `C` favours accuracy. Schlather (2002) recommends the use of `C = 3`.
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




"""
    maternchols(D, ρ, ν, σ² = 1; stack = true)
Given a distance matrix `D`, constructs the Cholesky factor of the covariance matrix
under the Matérn covariance function with range parameter `ρ`, smoothness
parameter `ν`, and marginal variance `σ²`.

Providing vectors of parameters will yield a three-dimensional array of Cholesky factors (note
that the vectors must of the same length, but a mix of vectors and scalars is
allowed). A vector of distance matrices `D` may also be provided.

If `stack = true`, the Cholesky factors will be "stacked" into a
three-dimensional array (this is only possible if all distance matrices in `D`
are the same size).

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
maternchols([D], ρ, ν)
maternchols(D, ρ, ν, σ²; stack = false)

S̃  = rand(n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²)
maternchols([D, D̃], ρ, ν, σ²; stack = false)

S̃  = rand(2n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²; stack = false)
```
"""
function maternchols(D, ρ, ν, σ² = one(eltype(D)); stack::Bool = true)

	K = max(length(ρ), length(ν), length(σ²))
	if K > 1
		@assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)])
		#TODO converting the parameters to be of length K below is not completely robust: if the parameters are a length-one vector, we will get a vector of a vector, which will cause an error. 
		if length(ρ)  == 1 ρ  = repeat([ρ], K) end
		if length(ν)  == 1 ν  = repeat([ν], K) end
		if length(σ²) == 1 σ² = repeat([σ²], K) end
	end

	# compute Cholesky factorization (exploit symmetry of D to minimise computations)
	# NB surprisingly, found that Folds.map() is slower than map()
	# TODO try FLoops and other parallel packages
	L = map(1:K) do k
		C = matern.(UpperTriangular(D), ρ[k], ν[k], σ²[k])
		L = cholesky(Symmetric(C)).L
		L = convert(Array, L) # convert from Triangular to Array so that stackarrays() can be used
		L
	end

	# Optionally convert from Vector of Matrices to 3D Array
	if stack
		L = stackarrays(L, merge = false)
	end

	return L
end

function maternchols(D::V, ρ, ν, σ² = one(nested_eltype(D)); stack::Bool = true) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	if stack
		@assert length(unique(size.(D))) == 1 "Converting the Cholesky factors from a vector of matrices to a three-dimenisonal array is only possible if the Cholesky factors (i.e., all matrices `D`) are the same size."
	end

	K = max(length(ρ), length(ν), length(σ²))
	if K > 1
		@assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)])
		#TODO converting the parameters to be of length K below is not completely robust: if the parameters are a length-one vector, we will get a vector of a vector, which will cause an error.
		if length(ρ)  == 1 ρ  = repeat([ρ], K) end
		if length(ν)  == 1 ν  = repeat([ν], K) end
		if length(σ²) == 1 σ² = repeat([σ²], K) end
	end
	@assert length(D) ∈ (1, K)

	# Compute the Cholesky factors
	L = maternchols.(D, ρ, ν, σ², stack = false)

	# L is currently a length-one Vector of Vectors: drop redundant outer vector
	L = stackarrays(L, merge = true)

	# Optionally convert from Vector of Matrices to 3D Array
	if stack
		L = stackarrays(L, merge = false)
	end
	return L
end
