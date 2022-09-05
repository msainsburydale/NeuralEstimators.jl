# TODO Finish the documentation for all of the simulateX functions..


#TODO Why do I need to force m to be an integer?
"""
	simulate(parameters, m::Integer, J::Integer)

Simulates `J` sets of `m` independent replicates for each parameter vector in
`parameters` by calling `simulate(parameters, m)` a total of `J` times.
"""
function simulate(parameters, m::Integer, J::Integer)
	v = [simulate(parameters, m) for i ∈ 1:J]
	v = vcat(v...) # should be ok since we're only splatting J vectors, which doesn't get prohibitively large even during bootstrapping. TODO No reason not to use stack, though.
	return v
end


# Wrapper function that returns simulated data and the true parameter values
_simulate(params::P, m) where {P <: Union{AbstractMatrix, ParameterConfigurations}} = (simulate(params, m), _extractθ(params))



# ---- Gaussian process ----


"""
	simulategaussianprocess(L::AbstractArray{T, 2}, σ::T, m::Integer)
	simulategaussianprocess(L::AbstractArray{T, 2})

Simulates `m` realisations from a Gau(0, 𝚺 + σ²𝐈) distribution, where 𝚺 ≡ LL'.

If `σ` and `m` are not provided, a single field without nugget variance is returned.
"""
function simulategaussianprocess(L::AbstractArray{T, 2}, σ::T, m::Integer) where T
	n = size(L, 1)
	y = similar(L, n, m)
	for h ∈ 1:m
		y[:, h] = simulategaussianprocess(L, σ)
	end
	return y
end

function simulategaussianprocess(L::AbstractArray{T, 2}, σ::T) where T
	n = size(L, 1)
	return simulategaussianprocess(L) + σ * randn(T, n)
end

function simulategaussianprocess(L::AbstractArray{T, 2}) where T
	n = size(L, 1)
	y = randn(T, n)
	return L * y
end


# ---- Schlather's max-stable model ----

"""
	simulateschlather(L::AbstractArray{T, 2}; C = 3.5)
	simulateschlather(L::AbstractArray{T, 2}, m::Integer; C = 3.5)

Simulates from Schlather's max-stable model.
"""
function simulateschlather(L::AbstractArray{T, 2}; C = 3.5) where T <: Number

	n = size(L, 1)  # number of spatial locations

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

	# Lenzi et al. used the log transform to stablise the variance, and this can
	# help avoid neural network collapse. Note that there is also a theoretical
	# justification for this transformation; it transforms from the data from
	# the unit Fréchet scale to the Gumbel scale, which is typically better behaved.
	Z = log.(Z) # TODO decide if this is what we want to do; can add an arguement transform::Bool = true.

	return Z
end

function simulateschlather(L::AbstractArray{T, 2}, m::Integer; C = 3.5) where T <: Number
	n = size(L, 1)
	Z = similar(L, n, m)
	for h ∈ 1:m
		Z[:, h] = simulateschlather(L, C = C)
	end

	return Z
end


# ---- Conditional extremes ----

a(h, z; λ, κ) = z * exp(-(h / λ)^κ)
b(h, z; β, λ, κ) = 1 + a(h, z, λ = λ, κ = κ)^β
delta(h; δ₁) = 1 + exp(-(h / δ₁)^2)

C̃(h, ρ, ν) = matern(h, ρ, ν)
σ̃₀(h, ρ, ν) = √(2 - 2 * C̃(h, ρ, ν))

Φ(q::T) where T <: Number = cdf(Normal(zero(T), one(T)), q)
t(ỹ₀₁, μ, τ, δ) = Fₛ⁻¹(Φ(ỹ₀₁), μ, τ, δ)


"""
	simulateconditionalextremes(θ::AbstractVector{T}, L::AbstractArray{T, 2}, h::AbstractVector{T}, s₀_idx::Integer, u::T) where T <: Number
	simulateconditionalextremes(θ::AbstractVector{T}, L::AbstractArray{T, 2}, h::AbstractVector{T}, s₀_idx::Integer, u::T, m::Integer) where T <: Number

Simulates from the spatial conditional extremes model for parameters.

# Examples
```
S = rand(Float32, 10, 2)
D = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ in eachrow(S)]
L = maternchols(D, 0.6f0, 0.5f0)
s₀ = S[1, :]'
h = map(norm, eachslice(S .- s₀, dims = 1))
s₀_idx = findfirst(x -> x == 0.0, h)
u = 0.7f0
simulateconditionalextremes(θ, L[:, :, 1], h, s₀_idx, u)
```
"""
function simulateconditionalextremes(
	θ::AbstractVector{T}, L::AbstractArray{T, 2}, h::AbstractVector{T}, s₀_idx::Integer, u::T, m::Integer
	) where T <: Number

	n = size(L, 1)
	Z = similar(L, n, m)
	for k ∈ 1:m
		Z[:, k] = simulateconditionalextremes(θ, L, h, s₀_idx, u)
	end

	return Z
end


function simulateconditionalextremes(
	θ::AbstractVector{T}, L::AbstractArray{T, 2}, h::AbstractVector{T}, s₀_idx::Integer, u::T
	) where T <: Number

	@assert length(θ) == 8
	@assert s₀_idx > 0
	@assert s₀_idx <= length(h)
	@assert size(L, 1) == size(L, 2)
	@assert size(L, 1) == length(h)

	# Parameters associated with a(.) and b(.):
	κ = θ[1]
	λ = θ[2]
	β = θ[3]
	# Covariance parameters associated with the Gaussian process
	ρ = θ[4]
	ν = θ[5]
	# Location and scale parameters for the residual process
	μ = θ[6]
	τ = θ[7]
	δ₁ = θ[8]

	# Construct the parameter δ used in the Subbotin distribution:
	δ = delta.(h, δ₁ = δ₁)

	# Observed datum at the conditioning site, Z₀:
	Z₀ = u + randexp(T)

	# Simulate a mean-zero Gaussian random field with unit marginal variance,
    # independently of Z₀. Note that Ỹ inherits the order of L. Therefore, we
	# can use s₀_idx to access s₀ in all subsequent vectors.
	Ỹ  = simulategaussianprocess(L)

	# Adjust the Gaussian process so that it is 0 at s₀
	Ỹ₀ = Ỹ .- Ỹ[s₀_idx]

	# Transform to unit variance:
	# σ̃₀ = sqrt.(2 .- 2 *  matern.(h, ρ, ν))
	Ỹ₀₁ = Ỹ₀ ./ σ̃₀.(h, ρ, ν)
	Ỹ₀₁[s₀_idx] = zero(T) # avoid pathology by setting Ỹ₀₁(s₀) = 0.

	# Probability integral transform from the standard Gaussian scale to the
	# standard uniform scale, and then inverse probability integral transform
	# from the standard uniform scale to the Subbotin scale:
    Y = t.(Ỹ₀₁, μ, τ, δ)

	# Apply the functions a(⋅) and b(⋅) to simulate data throughout the domain:
	Z = a.(h, Z₀, λ = λ, κ = κ) + b.(h, Z₀, β = β, λ = λ, κ = κ) .* Y

	# Variance stabilising transform
	Z = cbrt.(Z) # TODO decide if this is what we want to do; can add an arguement transform::Bool = true.

	return Z
end




# ---- Miscellaneous functions ----

#TODO replace besselk with https://github.com/cgeoga/BesselK.jl
@doc raw"""
    matern(h, ρ, ν, σ² = 1)
For two points separated by `h` units, compute the Matérn covariance function
with range `ρ`, smoothness `ν`, and marginal variance `σ²`.

We use the parametrisation
``C(\mathbf{h}) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\mathbf{h}\|}{\rho}\right) K_\nu \left(\frac{\|\mathbf{h}\|}{\rho}\right)``,
where ``\Gamma(\cdot)`` is the gamma function, and ``K_\nu(\cdot)`` is the modified Bessel
function of the second kind of order ``\nu``. This parameterisation is the same as used by the `R`
package `fields`, but differs to the parametrisation given by Wikipedia.

Note that the `Julia` functions for ``\Gamma(\cdot)`` and ``K_\nu(\cdot)``, respectively `gamma()` and
`besselk()`, do not work on the GPU and, hence, nor does `matern()`.
"""
function matern(h, ρ, ν, σ² = one(typeof(h)))

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

matern(h, ρ) =  matern(h, ρ, 1.0)


# TODO a bit weird that we're forcing σ = 1
"""
    maternchols(D, ρ, ν)
Given a distance matrix `D`, computes the covariance matrix under the
Matérn covariance function with range `ρ` and smoothness `ν`, and
returns the Cholesky factor of this covariance matrix.

Providing vectors for `ρ` and `ν` will yield a three-dimensional array of
Cholesky factors.
"""
function maternchols(D, ρ, ν)
	L = [cholesky(Symmetric(matern.(D, ρ[i], ν[i]))).L  for i ∈ eachindex(ρ)]
	L = convert.(Array, L) # TODO Would be better if stackarrays() could handle other classes. Maybe it would work if I remove the type from stackarrays()
	L = stackarrays(L, merge = false)
	return L
end

"""
    _incgammalowerunregularised(a, x)
For positive `a` and `x`, computes the lower unregularised incomplete gamma
function, ``\\gamma(a, x) = \\int_{0}^x t^{a-1}e^{-t}dt``.
"""
_incgammalowerunregularised(a, x) = incgamma(a, x; upper = false, reg = false)
