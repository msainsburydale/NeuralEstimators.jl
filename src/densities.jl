# ---- Helper functions for computing the MAP ----

# Scaled logistic function for constraining parameters
scaledlogistic(θ, Ω)    = scaledlogistic(θ, minimum(Ω), maximum(Ω))
scaledlogistic(θ, a, b) = a + (b - a) / (1 + exp(-θ))

# Inverse of scaledlogistic
scaledlogit(f, Ω)    = scaledlogit(f, minimum(Ω), maximum(Ω))
scaledlogit(f, a, b) = log((f - a) / (b - f))


# ---- Efficient gaussianloglikelihood ----

# The density function is
# ```math
# |2\pi\mathbf{\Sigma}|^{-1/2} \exp{-\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}},
# ```
# and the log-density is
# ```math
# -\frac{n}{2}\ln{2\pi}  -\frac{1}{2}\ln{|\mathbf{\Sigma}|} -\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}.
# ```

@doc raw"""
    gaussiandensity(y::V, L; logdensity = true) where {V <: AbstractVector{T}} where T
	gaussiandensity(y::A, Σ; logdensity = true) where {A <: AbstractArray{T, N}} where {T, N}

Efficiently computes the density function for `y` ~ 𝑁(0, `Σ`), with `L` the
lower Cholesky factor of the covariance matrix `Σ`.

The method `gaussiandensity(y::A, Σ)` assumes that the last dimension of `y`
corresponds to the independent-replicates dimension, and it exploits the fact
that we need to compute the Cholesky factor `L` for these independent replicates
once only.
"""
function gaussiandensity(y::V, L; logdensity::Bool = true) where {V <: AbstractVector{T}} where T
	n = length(y)
	x = L \ y # solution to Lx = y. If we need non-zero μ in the future, use x = L \ (y - μ)
	l = -0.5n*log(2π) -logdet(L) -0.5dot(x, x)
    return logdensity ? l : exp(l)
end

function gaussiandensity(y::A, Σ; logdensity::Bool = true) where {A <: AbstractArray{T, N}} where {T, N}

	# Here, we use `Symmetric()` to indicate that Σ is positive-definite;
	# this can help to alleviate issues caused by rounding, as described at
	# https://discourse.julialang.org/t/is-this-a-bug-with-cholesky/16970/3.
	L  = cholesky(Symmetric(Σ)).L
	l = mapslices(y -> gaussiandensity(vec(y), L, logdensity = logdensity), y, dims = 1:(N-1))
	return logdensity ? sum(l) : prod(l)
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
	schlatherbivariatedensity(z₁, z₂, ψ; logdensity = true)
The bivariate density function for Schlather's max-stable model, as given in
Huser (2013, pg. 231--232).

Huser, R. (2013). Statistical Modeling and Inference for Spatio-Temporal Ex-
tremes. PhD thesis, Swiss Federal Institute of Technology, Lausanne, Switzerland.
"""
schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true) = logdensity ? logG₁₂(z₁, z₂, ψ) : G₁₂(z₁, z₂, ψ)
_schlatherbivariatecdf(z₁, z₂, ψ) = G(z₁, z₂, ψ)
