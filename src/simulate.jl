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
`parameters` by calling `simulate(parameters, m)` a total of `J` times.
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

# Wrapper function that returns simulated data and the true parameter values
_simulate(params::P, m) where {P <: Union{AbstractMatrix, ParameterConfigurations}} = (simulate(params, m), _extractθ(params))

# ---- Gaussian process ----

"""
	simulategaussianprocess(L, σ, m)
	simulategaussianprocess(L)

Simulates `m` realisations from a Gau(0, 𝚺 + σ²𝐈) distribution, where 𝚺 ≡ LL'.

If `σ` and `m` are omitted, a single field without the nugget effect is returned.
"""
function simulategaussianprocess(L::M, σ::T, m::Integer) where M <: AbstractMatrix{T} where T <: Number
	n = size(L, 1)
	y = similar(L, n, m)
	for h ∈ 1:m
		y[:, h] = simulategaussianprocess(L, σ)
	end
	return y
end

function simulategaussianprocess(L::M, σ::T) where M <: AbstractMatrix{T} where T <: Number
	n = size(L, 1)
	return simulategaussianprocess(L) + σ * randn(T, n)
end

function simulategaussianprocess(L::M) where M <: AbstractMatrix{T} where T <: Number
	n = size(L, 1)
	y = randn(T, n)
	return L * y
end


# ---- Schlather's max-stable model ----

"""
	simulateschlather(L, m; C = 3.5, Gumbel = true)
	simulateschlather(L;    C = 3.5, Gumbel = true)

Given the lower Cholesky factor `L` associated with a Gaussian process,
simulates `m` independent and identically distributed (i.i.d.) realisations from
Schlather's max-stable model using the algorithm for approximate simulation given by Schlather (2002).

By default, the simulated data are log transformed from the unit Fréchet scale
to the `Gumbel` scale.

The accuracy of the algorithm is controlled with a tuning parameter, `C`, which
involves a trade-off between computational efficiency (favouring small `C`) and
accuracy (favouring large `C`). Schlather (2002) recommends the use of `C = 3`;
conservatively, we set the default to `C = 3.5`.


Schlather, M. (2002). Models for stationary max-stable random fields. Extremes, 5:33--44.
"""
function simulateschlather(L::M, m::Integer; C = 3.5, Gumbel::Bool = true) where M <: AbstractMatrix{T} where T <: Number
	n = size(L, 1)
	Z = similar(L, n, m)
	for h ∈ 1:m
		Z[:, h] = simulateschlather(L, C = C, Gumbel = Gumbel)
	end

	return Z
end

function simulateschlather(L::M; C = 3.5, Gumbel::Bool = true) where M <: AbstractMatrix{T} where T <: Number

	n = size(L, 1)  # number of observations

	Z   = fill(zero(T), n)
	ζ⁻¹ = randexp(T)
	ζ   = 1 / ζ⁻¹

	# A property of the model that must be enforced is E(max{0, Yᵢ}) = 1. It can
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
		Y = simulategaussianprocess(L)
		Y = √(2π)Y
		Z = max.(Z, ζ * Y)
		E = randexp(T)
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
Given a distance matrix `D`, constructs the Cholesky of the covariance matrix
under the Matérn covariance function with range parameter `ρ`, smoothness
parameter `ν`, and marginal variance σ².

Providing vectors of parameters will yield a three-dimensional array of
Cholesky factors (note that the vectors must of the same length, but a mix of
vectors and scalars is allowed). A vector of distance matrices `D` may also be
provided.

# Examples
```
using NeuralEstimators
using LinearAlgebra: norm
n  = 10
S  = rand(n, 2)
D  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ in eachrow(S)]
ρ  = [0.6, 0.5]
ν  = [0.7, 1.2]
σ² = [0.2, 0.4]
maternchols(D, ρ, ν)
maternchols(D, ρ, ν, σ²)

S̃  = rand(n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ in eachrow(S̃)]
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
	L = [cholesky(Symmetric(matern.(D, ρ[i], ν[i], σ²[i]))).L  for i ∈ 1:n]
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






"""
    _incgammalowerunregularised(a, x)
For positive `a` and `x`, computes the lower unregularised incomplete gamma
function, ``\\gamma(a, x) = \\int_{0}^x t^{a-1}e^{-t}dt``.
"""
_incgammalowerunregularised(a, x) = incgamma(a, x; upper = false, reg = false)
