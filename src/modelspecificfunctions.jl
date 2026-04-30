"""
	simulategaussian(L::AbstractMatrix, m = 1)
Simulates `m` independent and identically distributed realisations from
a mean-zero multivariate Gaussian random vector with associated lower Cholesky 
factor `L`. 

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are
returned as an ``n``x`m` matrix.

# Examples
```julia
using NeuralEstimators, Distances, LinearAlgebra

n = 500
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
ρ = 0.6
Σ = Symmetric(exp.(-D / ρ))
L = cholesky(Σ).L
simulategaussian(L)
```
"""
function simulategaussian(obj::M, m::Integer) where {M <: AbstractMatrix{T}} where {T <: Number}
    y = [simulategaussian(obj) for _ ∈ 1:m]
    y = stack(y)
    return y
end

function simulategaussian(L::M) where {M <: AbstractMatrix{T}} where {T <: Number}
    L * randn(T, size(L, 1))
end

"""
	simulateschlather(L::Matrix, m = 1; C = 3.5, Gumbel::Bool = false)
Simulates `m` independent and identically distributed realisations from
[Schlather's (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model given the lower Cholesky factor `L` of the covariance matrix of the underlying Gaussian process. 

The function uses the algorithm for approximate simulation given
by [Schlather (2002)](https://link.springer.com/article/10.1023/A:1020977924878).

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are 
returned as an ``n``x`m` matrix.

# Keyword arguments
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorithm. Small `C` favours computational efficiency, while large `C` favours accuracy. 
- `Gumbel = true`: flag indicating whether the data should be log-transformed from the unit Fréchet scale to the Gumbel scale.

# Examples
```julia
using NeuralEstimators, Distances, LinearAlgebra

n = 500
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
ρ = 0.6
Σ = Symmetric(exp.(-D / ρ))
L = cholesky(Σ).L
simulateschlather(L)
```
"""
function simulateschlather(obj::M, m::Integer; kwargs...) where {M <: AbstractMatrix{T}} where {T <: Number}
    y = [simulateschlather(obj; kwargs...) for _ ∈ 1:m]
    y = stack(y)
    return y
end

function simulateschlather(obj::M; C = 3.5, Gumbel::Bool = false) where {M <: AbstractMatrix{T}} where {T <: Number}
    n = size(obj, 1)  # number of observations
    Z = fill(zero(T), n)
    ζ⁻¹ = randexp(T)
    ζ = 1 / ζ⁻¹

    # We must enforce E(max{0, Yᵢ}) = 1. It can
    # be shown that this condition is satisfied if the marginal variance of Y(⋅)
    # is equal to 2π. Now, our simulation design embeds a marginal variance of 1
    # into fields generated from the cholesky factors, and hence
    # simulategaussian(L) returns simulations from a Gaussian
    # process with marginal variance 1. To scale the marginal variance to
    # 2π, we therefore need to multiply the field by √(2π).

    # Note that, compared with Algorithm 1.2.2 of Dey DK, Yan J (2016),
    # some simplifications have been made to the code below. This is because
    # max{Z(s), ζW(s)} ≡ max{Z(s), max{0, ζY(s)}} = max{Z(s), ζY(s)}, since
    # Z(s) is initialised to 0 and increases during simulation.

    while (ζ * C) > minimum(Z)
        Y = simulategaussian(obj)
        Y = √(T(2π)) * Y
        Z = max.(Z, ζ * Y)
        E = randexp(T)
        ζ⁻¹ += E
        ζ = 1 / ζ⁻¹
    end

    # Log transform the data from the unit Fréchet scale to the Gumbel scale,
    # which stabilises the variance and helps to prevent neural-network collapse.
    if Gumbel
        Z = log.(Z)
    end

    return Z
end

# ---- Miscellaneous functions ----

"""
	simulatepotts(grid::Matrix{Int}, β)
	simulatepotts(grid::Matrix{Union{Int, Nothing}}, β)
	simulatepotts(nrows::Int, ncols::Int, num_states::Int, β)
Chequerboard Gibbs sampling from a spatial Potts model with parameter `β`>0 (see, e.g., [Sainsbury-Dale et al., 2025, Sec. 3.3](https://arxiv.org/abs/2501.04330), and the references therein).

Approximately independent simulations can be obtained by setting 
`nsims` > 1 or `num_iterations > burn`. The degree to which the 
resulting simulations can be considered independent depends on the 
thinning factor (`thin`) and the burn-in (`burn`).

# Keyword arguments
- `nsims = 1`: number of approximately independent replicates. 
- `num_iterations = 2000`: number of MCMC iterations.
- `burn = num_iterations`: burn-in.
- `thin = 10`: thinning factor.

# Examples
```julia
using NeuralEstimators 

## Marginal simulation 
β = 0.8
simulatepotts(10, 10, 3, β)

## Marginal simulation: approximately independent samples 
simulatepotts(10, 10, 3, β; nsims = 100, thin = 10)

## Conditional simulation 
β = 0.8
complete_grid   = simulatepotts(100, 100, 3, β)      # simulate marginally 
incomplete_grid = removedata(complete_grid, 0.1)     # randomly remove 10% of the pixels 
imputed_grid    = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels

## Multiple conditional simulations 
imputed_grids   = simulatepotts(incomplete_grid, β; num_iterations = 2000, burn = 1000, thin = 10)

## Recreate Fig. 8.8 of Marin & Robert (2007) “Bayesian Core”
using Plots 
grids = [simulatepotts(100, 100, 2, β) for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio=1)
Plots.plot(heatmaps...)
```
"""
function simulatepotts(grid::AbstractMatrix{I}, β; nsims::Integer = 1, num_iterations::Integer = 2000, burn::Integer = num_iterations, thin::Integer = 10, mask = nothing) where {I <: Integer}
    β = β[1]  # unwrap if β was passed as a container

    @assert burn <= num_iterations
    if burn < num_iterations || nsims > 1
        Z₀ = simulatepotts(grid, β; num_iterations = burn, mask = mask)
        nsims == 1 && (nsims = (num_iterations - burn) ÷ thin)
        Z_chain = Vector{typeof(grid)}(undef, nsims)
        Z_chain[1] = Z₀

        @inbounds for i = 2:nsims
            Z_chain[i] = simulatepotts(copy(Z_chain[i - 1]), β; num_iterations = thin, mask = mask)
        end
        return Z_chain
    end

    nrows, ncols = size(grid)
    states = unique(skipmissing(grid))
    num_states = length(states)
    state_to_idx = Dict(s => i for (i, s) in enumerate(states))

    # Precompute chequerboard patterns
    chequerboard1 = [(i+j) % 2 == 0 for i = 1:nrows, j = 1:ncols]
    chequerboard2 = .!chequerboard1

    if !isnothing(mask)
        @assert size(grid) == size(mask)
        chequerboard1 = chequerboard1 .&& mask
        chequerboard2 = chequerboard2 .&& mask
    end

    chequerboards = if sum(chequerboard1) == 0
        (chequerboard2,)
    elseif sum(chequerboard2) == 0
        (chequerboard1,)
    else
        (chequerboard1, chequerboard2)
    end

    # Precompute neighbor offsets as CartesianIndex
    neighbor_offsets = CartesianIndex.([(0, 1), (1, 0), (0, -1), (-1, 0)])
    n = zeros(Int, num_states)
    probs = zeros(Float64, num_states)
    cum_probs = zeros(Float64, num_states)

    @inbounds for _ = 1:num_iterations
        for chequerboard in chequerboards
            for ci in findall(chequerboard)

                # Reset and count neighbors
                fill!(n, 0)
                for offset in neighbor_offsets
                    ni = ci + offset
                    checkbounds(Bool, grid, ni) || continue
                    state = grid[ni]
                    n[state_to_idx[state]] += 1
                end

                # Calculate probabilities
                max_n = maximum(n)
                if max_n == 0  # All neighbors same or no neighbors
                    grid[ci] = rand(states)
                    continue
                end

                # Compute unnormalized probabilities with log-sum-exp trick
                log_probs = β .* n
                max_log_prob = maximum(log_probs)
                probs = exp.(log_probs .- max_log_prob)
                sum_probs = sum(probs)
                probs ./= sum_probs

                # Sample new state
                u = rand()
                cumsum!(cum_probs, probs)
                new_state_idx = searchsortedfirst(cum_probs, u)
                grid[ci] = states[new_state_idx]
            end
        end
    end

    return grid
end

function simulatepotts(nrows::Integer, ncols::Integer, num_states::Integer, β; kwargs...)
    @assert length(β) == 1
    β = β[1]
    grid = initializepotts(nrows, ncols, num_states, β)
    simulatepotts(grid, β; kwargs...)
end

function simulatepotts(grid::AbstractMatrix{Union{Missing, I}}, β; kwargs...) where {I <: Integer}
    @assert length(β) == 1
    β = β[1]
    mask = ismissing.(grid)
    grid = initializepotts(grid, mask, β)
    simulatepotts(grid, β; kwargs..., mask = mask)
end

function simulatepotts(Z::A, β; kwargs...) where {A <: AbstractArray{T, N}} where {T, N}
    @assert all(size(Z)[3:end] .== 1) "simulatepotts() does not allow for independent replicates"

    # Save the original dimensions
    dims = size(Z)

    # Convert to matrix and pass to the matrix method
    Z = simulatepotts(Z[:, :], β; kwargs...)

    # Convert Z to the correct dimensions
    Z = reshape(Z, dims[1:(end - 1)]..., :)
end

function initializepotts(nrows::Integer, ncols::Integer, num_states::Integer, β; β_crit = log(1 + sqrt(num_states)))
    if β < β_crit
        # Random initialization for high temperature
        grid = rand(1:num_states, nrows, ncols)
    else
        # Clustered initialization for low temperature
        cluster_size = max(1, min(nrows, ncols) ÷ 4)
        clustered_rows = ceil(Int, nrows / cluster_size)
        clustered_cols = ceil(Int, ncols / cluster_size)
        base_grid = rand(1:num_states, clustered_rows, clustered_cols)
        grid = repeat(base_grid, inner = (cluster_size, cluster_size))
        grid = grid[1:nrows, 1:ncols] # trim to exact dimensions

        # Add small random perturbations
        for i = 1:nrows, j = 1:ncols
            if rand() < 0.05
                grid[i, j] = rand(1:num_states)
            end
        end
    end

    return grid
end

function initializepotts(grid::AbstractMatrix{Union{Missing, I}}, mask, β) where {I <: Integer}

    # Avoid mutating input
    grid = copy(grid)
    mask = copy(mask)

    # Early return if no missing values
    sum_mask = sum(mask)
    sum_mask == 0 && return convert(Matrix{I}, grid)

    # Find the number of states and compute the critical inverse temperature
    states = unique(skipmissing(grid))
    num_states = length(states)
    β_crit = log(1 + sqrt(num_states))

    if β < β_crit
        # High temperature: random fill for missing sites
        @inbounds grid[mask] .= rand(1:num_states, sum_mask)
    else
        # Low temperature: iterative neighbor-based fill

        # Pre-allocate neighbor arrays and offsets
        neigh_offsets = CartesianIndex.([(-1, 0), (1, 0), (0, -1), (0, 1)])
        neigh_buf = Vector{Int}(undef, 4)  # max 4 neighbors
        counts = zeros(Int, num_states)

        changed = true
        iterations = 0
        max_iterations = sum_mask * 2  # safety net

        while changed && any(mask) && iterations < max_iterations
            changed = false
            iterations += 1

            # Process all missing cells in each iteration
            for idx in findall(mask)
                count = 0
                # Check neighbors
                for offset in neigh_offsets
                    ni = idx + offset
                    checkbounds(Bool, grid, ni) || continue
                    @inbounds !mask[ni] || continue
                    @inbounds neigh_buf[count += 1] = grid[ni]
                end

                if count > 0
                    # Count neighbor states
                    fill!(counts, 0)
                    @inbounds for i = 1:count
                        s = neigh_buf[i]
                        counts[s] += 1
                    end

                    # Find modes
                    maxcount = maximum(@view counts[1:num_states])
                    n_modes = 0
                    @inbounds for s = 1:num_states
                        if counts[s] == maxcount
                            neigh_buf[n_modes += 1] = s
                        end
                    end

                    # Random tie-break
                    @inbounds grid[idx] = neigh_buf[rand(1:n_modes)]
                    mask[idx] = false
                    changed = true
                end
            end
        end

        # Fill any remaining missing values randomly
        remaining = sum(mask)
        if remaining > 0
            @inbounds grid[mask] .= rand(1:num_states, remaining)
        end
    end

    return convert(Matrix{I}, grid)
end

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
function gaussiandensity(y::V, L::LT; logdensity::Bool = true) where {V <: AbstractVector, LT <: LowerTriangular}
    n = length(y)
    x = L \ y # solution to Lx = y. If we need non-zero μ in the future, use x = L \ (y - μ)
    l = -0.5n*log(2π) - logdet(L) - 0.5dot(x, x)
    return logdensity ? l : exp(l)
end

function gaussiandensity(y::A, L::LT; logdensity::Bool = true) where {A <: AbstractArray{T, N}, LT <: LowerTriangular} where {T, N}
    l = mapslices(y -> gaussiandensity(vec(y), L; logdensity = logdensity), y, dims = 1:(N - 1))
    return logdensity ? sum(l) : prod(l)
end

function gaussiandensity(y::A, Σ::M; args...) where {A <: AbstractArray, M <: AbstractMatrix}
    L = cholesky(Symmetric(Σ)).L
    gaussiandensity(y, L; args...)
end

"""
	schlatherbivariatedensity(z₁, z₂, ψ₁₂; logdensity = true)
The bivariate density function (see, e.g., [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/suppl/10.1080/00031305.2023.2249522?scroll=top), Sec. S6.2) for [Schlather's (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model, where `ψ₁₂` denotes the spatial correlation function evaluated at the locations of observations `z₁` and `z₂`.
"""
schlatherbivariatedensity(z₁, z₂, ψ; logdensity::Bool = true) = logdensity ? logG₁₂(z₁, z₂, ψ) : G₁₂(z₁, z₂, ψ)
_schlatherbivariatecdf(z₁, z₂, ψ) = G(z₁, z₂, ψ)
G(z₁, z₂, ψ) = exp(-V(z₁, z₂, ψ))
G₁₂(z₁, z₂, ψ) = (V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) * exp(-V(z₁, z₂, ψ))
logG₁₂(z₁, z₂, ψ) = log(V₁(z₁, z₂, ψ) * V₂(z₁, z₂, ψ) - V₁₂(z₁, z₂, ψ)) - V(z₁, z₂, ψ)
f(z₁, z₂, ψ) = z₁^2 - 2*z₁*z₂*ψ + z₂^2
V(z₁, z₂, ψ) = (1/z₁ + 1/z₂) * (1 - 0.5(1 - (z₁+z₂)^-1 * f(z₁, z₂, ψ)^0.5))
V₁(z₁, z₂, ψ) = -0.5 * z₁^-2 + 0.5(ψ / z₁ - z₂/(z₁^2)) * f(z₁, z₂, ψ)^-0.5
V₂(z₁, z₂, ψ) = V₁(z₂, z₁, ψ)
V₁₂(z₁, z₂, ψ) = -0.5(1 - ψ^2) * f(z₁, z₂, ψ)^-1.5
