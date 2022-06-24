"""
	simulate(parameters::P, m::Integer, num_rep::Integer) where {P <: ParameterConfigurations}

Generic method that simulates `m` independent replicates for each parameter
configuration (by internally calling `simulate(parameters, m)`), repeated a
total of `num_rep` times.

See also [Data simulation](@ref).
"""
function simulate(parameters::P, m::Integer, num_rep::Integer) where {P <: ParameterConfigurations}
	v = [simulate(parameters, m) for i ∈ 1:num_rep]
	v = vcat(v...) # should be ok since we're only splatting num_rep vectors, which doesn't get prohibitively large even during bootstrapping. No reason not to use stack, though.
	return v
end

# Wrapper function that returns simulated data and the true parameter values
_simulate(params::P, m) where {P <: ParameterConfigurations} = (simulate(params, m), params.θ)



# ---- Gaussian process ----


"""
	simulategaussianprocess(L::AbstractArray{T, 2}, σ²::T, m::Integer)

Simulates `m` realisations from a Gau(0, 𝚺 + σ²𝐈) distribution, where 𝚺 ≡ LL'.
"""
function simulategaussianprocess(L::AbstractArray{T, 2}, σ²::T, m::Integer) where T
	n = size(L, 1)
	y = similar(L, n, 1, m)
	for h ∈ 1:m
		y[:, :, 1, h] = simulategaussianprocess(L, σ²)
	end
	return y
end

function simulategaussianprocess(L::AbstractArray{T, 2}, σ²::T) where T
	n = size(L, 1)
	return simulategaussianprocess(L) + sqrt(σ²) * randn(T, n, n)
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

Simulates from Schlather's max-stable model. Based on Algorithm 1.2.2 of Dey DK, Yan J (2016). Extreme value modeling and
risk analysis: methods and applications. CRC Press, Boca Raton, Florida.
"""
function simulateschlather(L::AbstractArray{T, 2}; C = 3.5) where T <: Number

	n = size(L, 1)  # number of spatial locations

	Z   = fill(zero(T), n) # TODO Why fill this with zeros? Just do undef.
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
	Z = similar(L, n, 1, m)
	for h ∈ 1:m
		Z[:, 1, h] = simulateschlather(L, C = C)
	end

	return Z
end


# ---- Conditional extremes ----

a(h, z; λ, κ) = z * exp(-(h / λ)^κ)
b(h, z; β, λ, κ) = 1 + a(h, z, λ = λ, κ = κ)^β
delta(h; δ₁) = 1 + exp(-(h / δ₁)^2)

C̃(h, ρ, ν) = matern(h, ρ, ν)
σ̃₀(h, ρ, ν) = √(2 - 2 * C̃(h, ρ, ν))

# TODO Finish this documentation.
"""
	simulateconditionalextremes(L::AbstractArray{T, 2}, h, s₀_idx, u; <keyword args>)
	simulateconditionalextremes(L::AbstractArray{T, 2}, h, s₀_idx, m::Integer; <keyword args>)


Simulates from the spatial conditional extremes model.
"""
function simulateconditionalextremes(
	L::AbstractArray{T, 2}, h, s₀_idx, u; # TODO should s₀ just be provided rather than s₀_idx? Also, is this the neatest treatment of the parameters? maybe u should be a keyword argument too.
	ρ, ν, κ, λ, β, μ, τ, δ₁
	) where T <: Number

	# Construct the parameter δ used in the Subbotin distribution:
	δ = delta.(h, δ₁ = δ₁)

	# Observed datum at the conditioning site, Z₀:
	Z₀ = u + randexp(T)

	# Simulate a mean-zero Gaussian random field with unit marginal variance,
    # independently of Z₀. Note that Ỹ inherits the order of L. Therefore, we
	# can use s₀_idx to access s₀ in all subsequent vectors.
	n  = size(L, 1)  # number of spatial locations
	y  = randn(T, n)
	Ỹ  = L * y

	# Adjust the Gaussian process so that it is 0 at s₀
	Ỹ₀ = Ỹ .- Ỹ[s₀_idx]

	# Transform to unit variance:
	# σ̃₀ = sqrt.(2 .- 2 *  matern.(h, ρ, ν))
	Ỹ₀₁ = Ỹ₀ ./ σ̃₀.(h, ρ, ν)
	Ỹ₀₁[s₀_idx] = zero(T) # avoid pathology by setting Ỹ₀₁(s₀) = 0.

	# Probability integral transform from the standard Gaussian scale to the
	# standard uniform scale, and then inverse probability integral transform
	# from the standard uniform scale to the Subbotin scale:
    Y = t.(Ỹ₀₁, μ, τ, δ) # = Fₛ⁻¹.(Φ.(Ỹ₀₁), μ, τ, δ)

	# Apply the functions a(⋅) and b(⋅) to simulate data throughout the domain:
	Z = a.(h, Z₀, λ = λ, κ = κ) + b.(h, Z₀, β = β, λ = λ, κ = κ) .* Y

	# Variance stabilising transform
	Z = cbrt.(Z) # TODO decide if this is what we want to do; can add an arguement transform::Bool = true.

	return Z
end

function simulateconditionalextremes(L::AbstractArray{T, 2}, h, s₀_idx, u, m::Integer; ρ, ν, κ, λ, β, μ, τ, δ₁) where T <: Number
	n = size(L, 1)
	Z = similar(L, n, 1, m)
	Threads.@threads for k ∈ 1:m
		Z[:, 1, k] = simulateconditionalextremes(L, h, s₀_idx, u, ρ = ρ, ν = ν, κ = κ, λ = λ, β = β, μ = μ, τ = τ, δ₁ = δ₁)
	end

	return Z
end

# ---- Intermeditate functions ----

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
function matern(h, ρ, ν, σ² = 1)

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

matern(h, ρ) =  matern(h, ρ, 1)


@doc raw"""
	fₛ(x, μ, τ, δ)
	Fₛ(q, μ, τ, δ)
	Fₛ⁻¹(p, μ, τ, δ)

The density, distribution, and quantile functions Subbotin (delta-Laplace)
distribution with location parameter `μ`, scale parameter `τ`, and shape
parameter `δ`:

```math
 f_S(y; \mu, \tau, \delta) = \frac{\delta}{2\tau \Gamma(1/\delta)} \exp{\left(-\left|\frac{y - \mu}{\tau}\right|^\delta\right)},\\
 F_S(y; \mu, \tau, \delta) = \frac{1}{2} + \textrm{sign}(y - \mu) \frac{1}{2 \Gamma(1/\delta)} \gamma\!\left(1/\delta, \left|\frac{y - \mu}{\tau}\right|^\delta\right),\\
 F_S^{-1}(p; \mu, \tau, \delta) = \text{sign}(p - 0.5)G^{-1}\left(2|p - 0.5|; \frac{1}{\delta}, \frac{1}{(k\tau)^\delta}\right)^{1/\delta} + \mu,
```

with ``\gamma(\cdot)`` and ``G^{-1}(\cdot)`` the unnormalised incomplete lower gamma function and quantile function of the Gamma distribution, respectively.

# Examples
```
p = [0.025, 0.05, 0.5, 0.9, 0.95, 0.975]

# Standard Gaussian:
μ = 0.0; τ = sqrt(2); δ = 2.0
Fₛ⁻¹.(p, μ, τ, δ)

# Standard Laplace:
μ = 0.0; τ = 1.0; δ = 1.0
Fₛ⁻¹.(p, μ, τ, δ)
```
"""
fₛ(x, μ, τ, δ)   = δ * exp(-(abs((x - μ)/τ)^δ)) / (2τ * gamma(1/δ))
Fₛ(q, μ, τ, δ)   = 0.5 + 0.5 * sign(q - μ) * (1 / gamma(1/δ)) * γ(1/δ, abs((q - μ)/τ)^δ)
Fₛ⁻¹(p, μ, τ, δ) = μ + sign(p - 0.5) * (τ^δ * quantile(Gamma(1/δ), 2 * abs(p - 0.5)))^(1/δ)


ϕ(y)   = pdf(Normal(0, 1), y)
Φ(q)   = cdf(Normal(0, 1), q)
Φ⁻¹(p) = quantile(Normal(0, 1), p)
t(ỹ₀₁, μ, τ, δ) = Fₛ⁻¹(Φ(ỹ₀₁), μ, τ, δ)
t⁻¹(y, μ, τ, δ) = Φ⁻¹(Fₛ(y, μ, τ, δ))
t′(y, μ, τ, δ)  = ϕ(y) / fₛ(y, μ, τ, δ) # NB this isn't used currently but it may be useful for unit testing
ln_t′_t⁻¹(y, μ, τ, δ) = log(ϕ(t⁻¹(y, μ, τ, δ))) - log(fₛ(y, μ, τ, δ))


# TODO add these in runtests.jl
# # Unit testing
# let
# 	# Check that the Subbotin pdf is consistent with the cdf using finite differences
# 	finite_diff(y, μ, τ, δ, ϵ = 0.000001) = (Fₛ(y + ϵ, μ, τ, δ) - Fₛ(y, μ, τ, δ)) / ϵ
# 	function finite_diff_check(y, μ, τ, δ)
# 		@test abs(finite_diff(y, μ, τ, δ) - fₛ(y, μ, τ, δ)) < 0.0001
# 	end
#
# 	finite_diff_check(-1, 0.1, 3, 1.2)
# 	finite_diff_check(0, 0.1, 3, 1.2)
# 	finite_diff_check(0.9, 0.1, 3, 1.2)
# 	finite_diff_check(3.3, 0.1, 3, 1.2)
#
# 	# Check that f⁻¹(f(y)) ≈ y
# 	μ = 0.5; τ = 1.3; δ = 2.4; y = 0.3
# 	@test abs(y - Fₛ⁻¹(Fₛ(y, μ, τ, δ), μ, τ, δ)) < 0.0001
# 	@test abs(y - t⁻¹(t(y, μ, τ, δ), μ, τ, δ)) < 0.0001
# end
