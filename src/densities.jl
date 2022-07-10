# ---- Helper functions for computing the MAP ----

# Scaled logistic function for constraining parameters
scaledlogistic(θ, Ω)    = scaledlogistic(θ, minimum(Ω), maximum(Ω))
scaledlogistic(θ, a, b) = a + (b - a) / (1 + exp(-θ))

# Inverse of scaledlogistic
scaledlogit(f, Ω)    = scaledlogit(f, minimum(Ω), maximum(Ω))
scaledlogit(f, a, b) = log((f - a) / (b - f))


# ---- Efficient gaussianloglikelihood ----

@doc raw"""
    gaussiandensity(y::A, L; logdensity::Bool = true) where {A <: AbstractArray{T, 1}} where T
	gaussiandensity(y::A, Σ; logdensity::Bool = true) where {A <: AbstractArray{T, N}} where {T, N}

Efficiently computes the density function for `y` ~ 𝑁(0, `Σ`), with `L` the lower Cholesky factor of the
covariance matrix `Σ`.

The second method assumes that the last dimension of `y` corresponds to the
replicates dimension, and it exploits the fact that we need to compute
the Cholesky factor `L` for these replicates once only.

The density function is
```math
|2\pi\mathbf{\Sigma}|^{-1/2} \exp{-\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}},
```
and the log-density is
```math
-\frac{1}{2}\ln{2\pi}  -\frac{1}{2}\ln{|\mathbf{\Sigma}|} -\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}.
```
"""
function gaussiandensity(y::A, L; logdensity::Bool = true) where {A <: AbstractArray{T, 1}} where T
	n = length(y)
	x = L \ y # solution to Lx = y. If we need non-zero μ in the future, use x = L \ (y - μ)
	l = -0.5n*log(2π) -logdet(L) -0.5x'x
    return logdensity ? l : exp(l)
end

function gaussiandensity(y::A, Σ; logdensity::Bool = true) where {A <: AbstractArray{T, N}} where {T, N}

	# Here, we use `Symmetric()` to indicate that Σ is positive-definite;
	# this can help to alleviate issues caused by rounding, as described at
	# https://discourse.julialang.org/t/is-this-a-bug-with-cholesky/16970/3.
	L  = cholesky(Symmetric(Σ)).L
	ll = mapslices(y -> gaussianlogdensity(vec(y), L), y, dims = 1:(N-1))
	return sum(ll)
end



# ---- Bivariate density function for Schlather's model ----

G(z₁, z₂, ψ)   = exp(-V(z₁, z₂, ψ))
G₁₂(z₁, z₂, ψ) = (V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) * exp(-V(z₁, z₂, ψ))
logG₁₂(z₁, z₂, ψ) = log(V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) - V(z₁, z₂, ψ)
f(z₁, z₂, ψ)   = z₁^2 - 2*z₁*z₂*ψ + z₂^2 # function to reduce code repetition
V(z₁, z₂, ψ)   = (1/z₁ + 1/z₂) * (1 - 0.5(1 - (z₁+z₂)^-1 * f(z₁, z₂, ψ)^0.5))
V₁(z₁, z₂, ψ)  = -0.5 * z₁^-2 + 0.5(ψ / z₁ - z₂/(z₁^2)) * f(z₁, z₂, ψ)^-0.5
V₂(z₁, z₂, ψ)  = V₁(z₂, z₁, ψ)
V₁₂(z₁, z₂, ψ) = -0.5(1 - ψ^2) * f(z₁, z₂, ψ)^-1.5

"""
	schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true)
The bivariate density function for Schlather's max-stable model, as given in
Raphaël Huser's PhD thesis (pg. 231-232) and Appendix C of the manuscript.
"""
schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true) = logdensity ? logG₁₂(z₁, z₂, ψ) : G₁₂(z₁, z₂, ψ)
_schlatherbivariatecdf(z₁, z₂, ψ) = G(z₁, z₂, ψ)


# ---- Subbotin (delta-Laplace) distribution ----

# TODO As written, there is no documentation for the function Fₛ and Fₛ⁻¹. Might
# need to define a Subbotin object, which would also be in line with Distributions.jl.
# See the following for a guide:
# https://github.com/JuliaStats/Distributions.jl/blob/6ab4c1f5bd1b5b6890bbb6afc9d3349dc90cad6a/src/univariate/continuous/normal.jl
# https://juliastats.org/Distributions.jl/stable/extends/
# struct Subbotin{T <: Real} <: ContinuousUnivariateDistribution
# 	μ::T
# 	τ::T
# 	δ::T
# 	Subbotin{T}(µ::T, τ::T, δ::T) where {T <: Real} = new{T}(µ, τ, δ)
# end
# const DeltaLaplace = Normal


@doc raw"""
	fₛ(x, μ, τ, δ)
	Fₛ(q, μ, τ, δ)
	Fₛ⁻¹(p, μ, τ, δ)

The density, distribution, and quantile functions of the Subbotin (delta-Laplace)
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
Fₛ(q, μ, τ, δ)   = 0.5 + 0.5 * sign(q - μ) * (1 / gamma(1/δ)) * _incgammalowerunregularised(1/δ, abs((q - μ)/τ)^δ)
Fₛ⁻¹(p, μ, τ, δ) = μ + sign(p - 0.5) * (τ^δ * quantile(Gamma(1/δ), 2 * abs(p - 0.5)))^(1/δ)
