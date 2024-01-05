# ---- Helper functions for computing the MAP ----

# Scaled logistic function for constraining parameters
scaledlogistic(θ, Ω)    = scaledlogistic(θ, minimum(Ω), maximum(Ω))
scaledlogistic(θ, a, b) = a + (b - a) / (1 + exp(-θ))

# Inverse of scaledlogistic
scaledlogit(f, Ω)    = scaledlogit(f, minimum(Ω), maximum(Ω))
scaledlogit(f, a, b) = log((f - a) / (b - f))


# ---- Gaussian density ----

# The density function is
# ```math
# |2\pi\mathbf{\Sigma}|^{-1/2} \exp{-\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}},
# ```
# and the log-density is
# ```math
# -\frac{n}{2}\ln{2\pi}  -\frac{1}{2}\ln{|\mathbf{\Sigma}|} -\frac{1}{2}\mathbf{y}^\top \mathbf{\Sigma}^{-1}\mathbf{y}.
# ```

@doc raw"""
    gaussiandensity(y::V, L::LT) where {V <: AbstractVector, LT <: LowerTriangular}
	gaussiandensity(y::A, L::LT) where {A <: AbstractArray, LT <: LowerTriangular}
	gaussiandensity(y::A, Σ::M) where {A <: AbstractArray, M <: AbstractMatrix}

Efficiently computes the density function for `y` ~ 𝑁(0, `Σ`) for covariance
matrix `Σ`, and where `L` is lower Cholesky factor of `Σ`.

The method `gaussiandensity(y::A, L::LT)` assumes that the last dimension of `y`
contains independent and identically distributed (iid) replicates.

The log-density is returned if the keyword argument `logdensity` is true (default).
"""
function gaussiandensity(y::V, L::LT; logdensity::Bool = true) where {V <: AbstractVector{T}, LT <: LowerTriangular} where T
	n = length(y)
	x = L \ y # solution to Lx = y. If we need non-zero μ in the future, use x = L \ (y - μ)
	l = -0.5n*log(2π) -logdet(L) -0.5dot(x, x)
    return logdensity ? l : exp(l)
end

function gaussiandensity(y::A, L::LT; logdensity::Bool = true) where {A <: AbstractArray{T, N}, LT <: LowerTriangular} where {T, N}
	l = mapslices(y -> gaussiandensity(vec(y), L; logdensity = logdensity), y, dims = 1:(N-1))
	return logdensity ? sum(l) : prod(l)
end

function gaussiandensity(y::A, Σ::M; args...) where {A <: AbstractArray{T, N}, M <: AbstractMatrix{T}} where {T, N}
	L = cholesky(Symmetric(Σ)).L
	gaussiandensity(y, L; args...)
end

#TODO Add generalised-hyperbolic density once neural EM paper is finished.

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
The bivariate density function for Schlather's max-stable model.
"""
schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true) = logdensity ? logG₁₂(z₁, z₂, ψ) : G₁₂(z₁, z₂, ψ)
_schlatherbivariatecdf(z₁, z₂, ψ) = G(z₁, z₂, ψ)
