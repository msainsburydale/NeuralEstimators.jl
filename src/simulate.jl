"""
	simulategaussian(L::AbstractMatrix, m = 1)
Simulates `m` independent and identically distributed realisations from
a mean-zero multivariate Gaussian random vector with associated lower Cholesky 
factor `L`. 

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are
returned as an ``n``x`m` matrix.

# Examples
```
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulategaussian(L)
```
"""
function simulategaussian(obj::M, m::Integer) where {M <: AbstractMatrix{T}} where {T <: Number}
    y = [simulategaussian(obj) for _ ∈ 1:m]
    y = stackarrays(y, merge = false)
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
```
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulateschlather(L)
```
"""
function simulateschlather(obj::M, m::Integer; kwargs...) where {M <: AbstractMatrix{T}} where {T <: Number}
    y = [simulateschlather(obj; kwargs...) for _ ∈ 1:m]
    y = stackarrays(y, merge = false)
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

#NB Currently, second order optimisation methods cannot be used
# straightforwardly because besselk() is not differentiable. In the future, we
# can add an argument to matern() and maternchols(), besselfn = besselk, which
# allows the user to change the bessel function to use adbesselk(), which
# allows automatic differentiation: see https://github.com/cgeoga/BesselK.jl.
@doc raw"""
    matern(h, ρ, ν, σ² = 1)
Given distance ``\|\boldsymbol{h}\|`` (`h`), computes the Matérn covariance function

```math
C(\|\boldsymbol{h}\|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\boldsymbol{h}\|}{\rho}\right)^\nu K_\nu \left(\frac{\|\boldsymbol{h}\|}{\rho}\right),
```

where `ρ` is a range parameter, `ν` is a smoothness parameter, `σ²` is the marginal variance, 
``\Gamma(\cdot)`` is the gamma function, and ``K_\nu(\cdot)`` is the modified Bessel
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

@doc raw"""
    paciorek(s, r, ω₁, ω₂, ρ, β)
Given spatial locations `s` and `r`, computes the nonstationary covariance function 
```math
C(\boldsymbol{s}, \boldsymbol{r}) = 
|\boldsymbol{\Sigma}(\boldsymbol{s})|^{1/4}
|\boldsymbol{\Sigma}(\boldsymbol{r})|^{1/4}
\left|\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right|^{-1/2}
C^0\big(\sqrt{Q(\boldsymbol{s}, \boldsymbol{r})}\big), 
```
where $C^0(h) = \exp\{-(h/\rho)^{3/2}\}$ for range parameter $\rho > 0$, 
the matrix $\boldsymbol{\Sigma}(\boldsymbol{s}) = \exp(\beta\|\boldsymbol{s} - \boldsymbol{\omega}\|)\boldsymbol{I}$ 
is a kernel matrix ([Paciorek and Schervish, 2006](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.785)) 
with scale parameter $\beta > 0$ and reference point $\boldsymbol{\omega} \equiv (\omega_1, \omega_2)' \in \mathbb{R}^2$,
and 
```math
Q(\boldsymbol{s}, \boldsymbol{r}) = 
(\boldsymbol{s} - \boldsymbol{r})'
\left(\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right)^{-1}
(\boldsymbol{s} - \boldsymbol{r})
``` 
is the squared Mahalanobis distance between $\boldsymbol{s}$ and $\boldsymbol{r}$. 

Note that, in practical applications, the reference point $\boldsymbol{\omega}$ is often taken to be an estimable parameter rather than fixed and known. 
"""
function paciorek(s, r, ω₁, ω₂, ρ, β)

    # Displacement vector
    h = s - r

    # Distance from each point to ω ≡ (ω₁, ω₂)'
    dₛ = sqrt((s[1] - ω₁)^2 + (s[2] - ω₂)^2)
    dᵣ = sqrt((r[1] - ω₁)^2 + (r[2] - ω₂)^2)

    # Scaling factors of kernel matrices, such that Σ(s) = a(s)I
    aₛ = exp(β * dₛ)
    aᵣ = exp(β * dᵣ)

    # Several computational efficiencies afforded by use of a diagonal kernel matrix:
    # - the inverse of a diagonal matrix is given by replacing the diagonal elements with their reciprocals
    # - the determinant of a diagonal matrix is equal to the product of its diagonal elements

    # Mahalanobis distance
    Q = 2 * h'h / (aₛ + aᵣ)

    # Explicit version of code 
    # Σₛ_det = aₛ^2   
    # Σᵣ_det = aᵣ^2   
    # C⁰ = exp(-sqrt(Q/ρ)^1.5)
    # logC = 1/4*log(Σₛ_det) + 1/4*log(Σᵣ_det) - log((aₛ + aᵣ)/2) + log(C⁰)

    # Numerically stable version of code
    logC = β*dₛ/2 + β*dᵣ/2 - log((aₛ + aᵣ)/2) - (sqrt(Q)/ρ)^1.5

    exp(logC)
end

"""
    maternchols(D, ρ, ν, σ² = 1; stack = true)
Given a matrix `D` of distances, constructs the Cholesky factor of the covariance matrix
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
        @assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)]) "`ρ`, `ν`, and `σ²` should be the same length"
        ρ = _coercetoKvector(ρ, K)
        ν = _coercetoKvector(ν, K)
        σ² = _coercetoKvector(σ², K)
    end

    # compute Cholesky factorization (exploit symmetry of D to minimise computations)
    # NB surprisingly, found that the parallel Folds.map() is slower than map(). Could try FLoops or other parallelisation packages.
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
        @assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)]) "`ρ`, `ν`, and `σ²` should be the same length"
        ρ = _coercetoKvector(ρ, K)
        ν = _coercetoKvector(ν, K)
        σ² = _coercetoKvector(σ², K)
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

# Coerces a single-number or length-one-vector x into a K vector
function _coercetoKvector(x, K)
    @assert length(x) ∈ (1, K)
    if !isa(x, Vector)
        x = [x]
    end
    if length(x) == 1
        x = repeat(x, K)
    end
    return x
end

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
```
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
