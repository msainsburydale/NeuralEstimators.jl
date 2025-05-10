# Scaled logistic function for constraining parameters
scaledlogistic(θ, Ω) = scaledlogistic(θ, minimum(Ω), maximum(Ω))
scaledlogistic(θ, a, b) = a + (b - a) / (1 + exp(-θ))

# Inverse of scaledlogistic
scaledlogit(f, Ω) = scaledlogit(f, minimum(Ω), maximum(Ω))
scaledlogit(f, a, b) = log((f - a) / (b - f))

@doc raw"""
    gaussiandensity(Z::V, L::LT) where {V <: AbstractVector, LT <: LowerTriangular}
	gaussiandensity(Z::A, L::LT) where {A <: AbstractArray, LT <: LowerTriangular}
	gaussiandensity(Z::A, Σ::M) where {A <: AbstractArray, M <: AbstractMatrix}
Efficiently computes the density function for `Z` ~ 𝑁(0, `Σ`), namely,  
```math
|2\pi\boldsymbol{\Sigma}|^{-1/2} \exp\{-\frac{1}{2}\boldsymbol{Z}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{Z}\},
```
for covariance matrix `Σ`, and where `L` is lower Cholesky factor of `Σ`.

The method `gaussiandensity(Z::A, L::LT)` assumes that the last dimension of `Z`
contains independent and identically distributed replicates.

If `logdensity = true` (default), the log-density is returned.
"""
function gaussiandensity(
    y::V,
    L::LT;
    logdensity::Bool = true
) where {V <: AbstractVector, LT <: LowerTriangular}
    n = length(y)
    x = L \ y # solution to Lx = y. If we need non-zero μ in the future, use x = L \ (y - μ)
    l = -0.5n*log(2π) - logdet(L) - 0.5dot(x, x)
    return logdensity ? l : exp(l)
end

function gaussiandensity(
    y::A,
    L::LT;
    logdensity::Bool = true
) where {A <: AbstractArray{T, N}, LT <: LowerTriangular} where {T, N}
    l = mapslices(
        y -> gaussiandensity(vec(y), L; logdensity = logdensity),
        y,
        dims = 1:(N - 1)
    )
    return logdensity ? sum(l) : prod(l)
end

function gaussiandensity(
    y::A,
    Σ::M;
    args...
) where {A <: AbstractArray, M <: AbstractMatrix}
    L = cholesky(Symmetric(Σ)).L
    gaussiandensity(y, L; args...)
end

"""
	schlatherbivariatedensity(z₁, z₂, ψ₁₂; logdensity = true)
The bivariate density function (see, e.g., [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/suppl/10.1080/00031305.2023.2249522?scroll=top), Sec. S6.2) for [Schlather's (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model, where `ψ₁₂` denotes the spatial correlation function evaluated at the locations of observations `z₁` and `z₂`.
"""
schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true) =
    logdensity ? logG₁₂(z₁, z₂, ψ) : G₁₂(z₁, z₂, ψ)
_schlatherbivariatecdf(z₁, z₂, ψ) = G(z₁, z₂, ψ)
G(z₁, z₂, ψ) = exp(-V(z₁, z₂, ψ))
G₁₂(z₁, z₂, ψ) = (V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) * exp(-V(z₁, z₂, ψ))
logG₁₂(z₁, z₂, ψ) = log(V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) - V(z₁, z₂, ψ)
f(z₁, z₂, ψ) = z₁^2 - 2*z₁*z₂*ψ + z₂^2
V(z₁, z₂, ψ) = (1/z₁ + 1/z₂) * (1 - 0.5(1 - (z₁+z₂)^-1 * f(z₁, z₂, ψ)^0.5))
V₁(z₁, z₂, ψ) = -0.5 * z₁^-2 + 0.5(ψ / z₁ - z₂/(z₁^2)) * f(z₁, z₂, ψ)^-0.5
V₂(z₁, z₂, ψ) = V₁(z₂, z₁, ψ)
V₁₂(z₁, z₂, ψ) = -0.5(1 - ψ^2) * f(z₁, z₂, ψ)^-1.5
