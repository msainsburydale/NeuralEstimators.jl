@doc raw"""
    EM(simulateconditional::Function, MAP::Union{Function, NeuralEstimator}, θ₀ = nothing)

Implements a (Bayesian) Monte Carlo expectation-maximization (EM) algorithm for 
parameter estimation with missing data. The algorithm iteratively simulates missing 
data conditional on current parameter estimates, then updates parameters using a 
(neural) maximum a posteriori (MAP) estimator.

The ``l``th iteration is given by:

```math
\boldsymbol{\theta}^{(l)} =
\underset{\boldsymbol{\theta}}{\mathrm{arg\,max}}
\sum_{h = 1}^H \ell(\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2^{(l, h)}) + H\log \pi(\boldsymbol{\theta})
```

where ``\ell((\boldsymbol{\theta};  \boldsymbol{Z}_1,  \boldsymbol{Z}_2)`` denotes the complete-data log-likelihood function, 
``\boldsymbol{Z} \equiv (\boldsymbol{Z}_1', \boldsymbol{Z}_2')'`` denotes the complete 
data with ``\boldsymbol{Z}_1`` and ``\boldsymbol{Z}_2`` the observed and missing 
components respectively, ``\boldsymbol{Z}_2^{(l, h)}``, ``h = 1, \dots, H``, are simulated 
from the distribution of ``\boldsymbol{Z}_2 \mid \boldsymbol{Z}_1, \boldsymbol{\theta}^{(l-1)}``, and 
``\pi(\boldsymbol{\theta})`` is the prior density (which can be viewed as a penalty function).

The algorithm monitors convergence by computing:

```math
\max_i \left| \frac{\bar{\theta}_i^{(l+1)} - \bar{\theta}_i^{(l)}}{|\bar{\theta}_i^{(l)}| + \epsilon} \right|
```

where ``\bar{\theta}^{(l)}`` is the average of parameter estimates from iteration 
`burnin+1` to iteration ``l``, and ``\epsilon`` is machine precision. Convergence is 
declared when this quantity is less than `tol` for `nconsecutive` consecutive iterations (see keyword arguments below).

# Fields

- `simulateconditional::Function`: Function for simulating missing data conditional on 
  observed data and current parameter estimates. Must have signature:
  ```julia
  simulateconditional(Z::AbstractArray{Union{Missing, T}}, θ; nsims = 1, kwargs...)
  ```
  Returns completed data in the format expected by `MAP` (e.g., 4D array for CNNs).

- `MAP::NeuralEstimator`: MAP estimator applied to completed data. 

- `θ₀`: Optional initial parameter values (vector). Can also be provided when calling 
  the `EM` object.

# Methods

Once constructed, objects of type `EM` can be applied to data via the following methods (corresponding to single or multiple data sets, respectively):

	(em::EM)(Z::A, θ₀::Union{Nothing, Vector} = nothing; ...) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}
	(em::EM)(Z::V, θ₀::Union{Nothing, Vector, Matrix} = nothing; ...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

where `Z` is the complete data containing the observed data and `Missing` values.

For multiple datasets, `θ₀` can be a vector (same initial values for all) or a matrix 
with one column per dataset.

# Keyword Arguments

- `niterations::Integer = 50`: Maximum number of EM iterations.
- `nsims::Union{Integer, Vector{<:Integer}} = 1`: Number of conditional simulations per 
  iteration. Can be fixed (scalar) or varying (vector of length `niterations`).
- `burnin::Integer = 1`: Number of initial iterations to discard before averaging 
  parameter estimates for convergence assessment.
- `nconsecutive::Integer = 3`: Number of consecutive iterations meeting the convergence 
  criterion required to halt.
- `tol = 0.01`: Convergence tolerance. Algorithm stops if the relative change in 
  post-burnin averaged parameters is less than `tol` for `nconsecutive` iterations.
- `use_gpu::Bool = true`: Whether to use a GPU (if available) for MAP estimation.
- `verbose::Bool = false`: Whether to print iteration details.
- `kwargs...`: Additional arguments passed to `simulateconditional`.

# Returns

For a single data set, returns a named tuple containing:
- `estimate`: Final parameter estimate (post-burnin average).
- `iterates`: Matrix of all parameter estimates across iterations (each column is one iteration).
- `burnin`: The burnin value used.

For multiple data set, returns a matrix with one column per dataset.

# Notes

- If `Z` contains no missing values, the MAP estimator is applied directly (after 
  passing through `simulateconditional` to ensure correct format).
- When using a GPU, data are moved to the GPU for MAP estimation and back to the CPU 
  for conditional simulation.

# Examples

```julia
# Below we give a pseudocode example; see the "Missing data" section in "Advanced usage" for a concrete example.

# Define conditional simulation function
function sim_conditional(Z, θ; nsims = 1)
    # Your implementation here
    # Returns completed data in format suitable for MAP estimator
end

# Define or load MAP estimator
MAP_estimator = ... # Neural MAP estimator

# Create EM object
em = EM(sim_conditional, MAP_estimator, θ₀ = [1.0, 2.0])

# Apply to data with missing values
Z = ... # Array with Missing entries
result = em(Z, niterations = 100, nsims = 5, tol = 0.001, verbose = true)

# Access results
θ_final = result.estimate
θ_sequence = result.iterates

# Multiple datasets
Z_list = [Z1, Z2, Z3]
estimates = em(Z_list, θ₀ = [1.0, 2.0])  # Matrix with 3 columns
```
"""
struct EM{F, T, S}
    simulateconditional::F
    MAP::T
    θ₀::S
end
EM(simulateconditional, MAP) = EM(simulateconditional, MAP, nothing)
EM(em::EM, θ₀) = EM(em.simulateconditional, em.MAP, θ₀)
EM(simulateconditional, MAP, θ₀::Number) = EM(simulateconditional, MAP, [θ₀])

function (em::EM)(
    Z::A, θ₀ = nothing;
    niterations::Integer = 50,
    nsims::Union{Integer, AbstractVector{<:Integer}} = 1,
    burnin::Integer = 1,              
    nconsecutive::Integer = 3,
    tol = 0.01,
    use_gpu::Bool = true,
    verbose::Bool = false,
    kwargs...
) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

    @assert burnin < niterations
    @assert burnin >= 0
    
    # Validate nsims
    if isa(nsims, AbstractVector)
        @assert length(nsims) == niterations "When nsims is a vector, its length must equal niterations ($(niterations))"
        @assert all(nsims .> 0) "All elements of nsims must be positive"
    else
        @assert nsims > 0 "nsims must be positive"
    end

    if isnothing(θ₀)
        @assert !isnothing(em.θ₀) "Initial estimates θ₀ must be provided either in the `EM` object or in the function call when applying the `EM` object"
        θ₀ = em.θ₀
    end
    @assert !all(ismissing.(Z)) "The data consist of missing elements only: we cannot make inference without any information"
    
    if isa(θ₀, Number)
        θ₀ = [θ₀]
    end

    θ₀ = cpu(θ₀)

    device = _checkgpu(use_gpu, verbose = verbose)
    MAP = em.MAP |> device

    if !any(ismissing.(Z))
        verbose && @warn """
        Data passed to the EM algorithm contains no missing values.
        The MAP estimator will be applied directly to the data, but the data
        will first be passed through the conditional simulation function to
        ensure correct dimensions.

        (Tip: your conditional simulation function should handle complete-data cases explicitly.)
        """
        # Use first element of nsims vector if provided, otherwise use scalar
        nsims_current = isa(nsims, AbstractVector) ? nsims[1] : nsims
        Z = em.simulateconditional(Z, θ₀, nsims = nsims_current, kwargs...)
        Z = convert(Array{nonmissingtype(eltype(Z))}, Z)
        Z = Z |> device
        return (estimate = MAP(Z), ) #NB return as named tuple for type stability
    end

    verbose && @show θ₀
    θₗ = θ₀
    θ_all = reshape(θ₀, :, 1) 
    convergence_counter = 0
    barθₗ = nothing
    for l ∈ 1:niterations

        # Get current nsims value (either from vector or use scalar)
        nsims_current = isa(nsims, AbstractVector) ? nsims[l] : nsims
        
        # Complete the data by conditional simulation
        Z̃ = em.simulateconditional(Z, θₗ; nsims = nsims_current, kwargs...)
        Z̃ = Z̃ |> device

        # Apply the MAP estimator to the complete data
        θₗ₊₁ = MAP(Z̃)

        # Move back to the cpu for conditional simulation in the next iteration
        θₗ₊₁ = cpu(θₗ₊₁)
        θ_all = hcat(θ_all, θₗ₊₁)

        # Compute post burn-in mean and check convergence
        if l > burnin
            barθₗ₊₁ = mean(θ_all[:, burnin+1:end]; dims=2)

            if l > (burnin + 1) && maximum(abs.(barθₗ₊₁-barθₗ) ./ (abs.(barθₗ) .+ eps())) < tol
                convergence_counter += 1
                if convergence_counter == nconsecutive
                    verbose && @info "The EM algorithm has converged"
                    break
                end
            else
                convergence_counter = 0
            end
            barθₗ = barθₗ₊₁
        end

        if l == niterations 
            verbose && @warn "The EM algorithm has failed to converge"
        end

        θₗ = θₗ₊₁

        if verbose
            @show l
            @show nsims_current
            @show θₗ
            if l > burnin
                @show barθₗ
            end
        end
    end

    return (estimate = barθₗ, burnin = burnin, iterates = θ_all)
end

# Helper function for applying the EM algorithm to many data sets at once
function (em::EM)(Z::V, θ₀::Union{Number, Vector, Matrix, Nothing} = nothing; kwargs...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}
    if isnothing(θ₀)
        @assert !isnothing(em.θ₀) "Please provide initial estimates `θ₀` in the function call or in the `EM` object."
        θ₀ = em.θ₀
    end

    if isa(θ₀, Number)
        θ₀ = [θ₀]
    end

    if isa(θ₀, Vector)
        θ₀ = repeat(θ₀, 1, length(Z))
    end

    θ₀ = cpu(θ₀)

    estimates = Folds.map(eachindex(Z)) do i
        em(Z[i], θ₀[:, i]; kwargs...).estimate
    end
    estimates = reduce(hcat, estimates)

    return estimates
end

"""
	removedata(Z::Array, Iᵤ::Vector{T}) where T <: Union{Integer, CartesianIndex}
	removedata(Z::Array, p::Union{Float, Vector{Float}}; prevent_complete_missing = true)
	removedata(Z::Array, n::Integer; fixed_pattern = false, contiguous_pattern = false, variable_proportion = false)
Replaces elements of `Z` with `missing`.

The simplest method accepts a vector `Iᵤ` that specifes the indices
of the data to be removed.

Alternatively, there are two methods available to randomly generate missing data.

First, a vector `p` may be given that specifies the proportion of missingness
for each element in the response vector. Hence, `p` should have length equal to
the dimension of the response vector. If a single proportion is given, it
will be replicated accordingly. If `prevent_complete_missing = true`, no
replicates will contain 100% missingness (note that this can slightly alter the
effective values of `p`).

Second, if an integer `n` is provided, all replicates will contain
`n` observations after the data are removed. If `fixed_pattern = true`, the
missingness pattern is fixed for all replicates. If `contiguous_pattern = true`,
the data will be removed in a contiguous block based on a randomly selected starting index. If `variable_proportion = true`,
the proportion of missingness will vary across replicates, with each replicate containing
between 1 and `n` observations after data removal, sampled uniformly (note that
`variable_proportion` overrides `fixed_pattern`).

The return type is `Array{Union{T, Missing}}`.

# Examples
```
d = 5           # dimension of each replicate
m = 2000        # number of replicates
Z = rand(d, m)  # simulated data

# Passing a desired proportion of missingness
p = rand(d)
removedata(Z, p)

# Passing a desired final sample size
n = 3  # number of observed elements of each replicate: must have n <= d
removedata(Z, n)
```
"""
function removedata(Z::A, n::Integer;
    fixed_pattern::Bool = false,
    contiguous_pattern::Bool = false,
    variable_proportion::Bool = false
) where {A <: AbstractArray{T, N}} where {T, N}
    if isa(Z, Vector)
        Z = reshape(Z, :, 1)
    end
    m = size(Z)[end]           # number of replicates
    d = prod(size(Z)[1:(end - 1)]) # dimension of each replicate  NB assumes a singleton channel dimension

    if n == d
        # If the user requests fully observed data, we still convert Z to
        # an array with an eltype that allows missing data for type stability
        Iᵤ = Int64[]

    elseif variable_proportion
        Zstar = map(eachslice(Z; dims = N)) do z
            # Pass number of observations between 1:n into removedata()
            removedata(
                reshape(z, size(z)..., 1),
                StatsBase.sample(1:n, 1)[1],
                fixed_pattern = fixed_pattern,
                contiguous_pattern = contiguous_pattern,
                variable_proportion = false
            )
        end

        return stackarrays(Zstar)

    else

        # Generate the missing elements
        if fixed_pattern
            if contiguous_pattern
                start = StatsBase.sample(1:(n + 1), 1)[1]
                Iᵤ = start:(start + (d - n) - 1)
            else
                Iᵤ = StatsBase.sample(1:d, d-n, replace = false)
            end
            Iᵤ = [Iᵤ .+ (i-1) * d for i ∈ 1:m]
        else
            if contiguous_pattern
                Iᵤ = map(1:m) do i
                    start = (StatsBase.sample(1:(n + 1), 1) .+ (i - 1) * d)[1]
                    start:(start + (d - n) - 1)
                end
            else
                Iᵤ = [StatsBase.sample((1:d) .+ (i-1) * d, d - n, replace = false) for i ∈ 1:m]
            end
        end
        Iᵤ = vcat(Iᵤ...)
    end

    return removedata(Z, Iᵤ)
end
function removedata(Z::V, n::Integer; args...) where {V <: AbstractVector{T}} where {T}
    removedata(reshape(Z, :, 1), n)[:]
end

function removedata(Z::A, p::F; args...) where {A <: AbstractArray{T, N}} where {T, N, F <: AbstractFloat}
    if isa(Z, Vector)
        Z = reshape(Z, :, 1)
    end
    d = prod(size(Z)[1:(end - 1)]) # dimension of each replicate  NB assumes singleton channel dimension
    p = repeat([p], d)
    return removedata(Z, p; args...)
end
function removedata(Z::V, p::F; args...) where {V <: AbstractVector{T}} where {T, F <: AbstractFloat}
    removedata(reshape(Z, :, 1), p)[:]
end

function removedata(Z::A, p::Vector{F}; prevent_complete_missing::Bool = true) where {A <: AbstractArray{T, N}} where {T, N, F <: AbstractFloat}
    if isa(Z, Vector)
        Z = reshape(Z, :, 1)
    end
    m = size(Z)[end]           # number of replicates
    d = prod(size(Z)[1:(end - 1)]) # dimension of each replicate  NB assumes singleton channel dimension
    @assert length(p) == d "The length of `p` should equal the dimenison d of each replicate"

    if all(p .== 1)
        prevent_complete_missing = false
    end

    if prevent_complete_missing
        Iᵤ = map(1:m) do _
            complete_missing = true
            while complete_missing
                Iᵤ = collect(rand(length(p)) .< p) # sample from multivariate bernoulli
                complete_missing = !(0 ∈ Iᵤ)
            end
            Iᵤ
        end
    else
        Iᵤ = [collect(rand(length(p)) .< p) for _ ∈ 1:m]
    end

    Iᵤ = stackarrays(Iᵤ)
    Iᵤ = findall(Iᵤ)

    return removedata(Z, Iᵤ)
end
function removedata(Z::V, p::Vector{F}; args...) where {V <: AbstractVector{T}} where {T, F <: AbstractFloat}
    removedata(reshape(Z, :, 1), p)[:]
end

function removedata(Z::A, Iᵤ::V) where {A <: AbstractArray{T, N}, V <: AbstractVector{I}} where {T, N, I <: Union{Integer, CartesianIndex}}

    # Convert the Array to a type that allows missing data
    Z₁ = convert(Array{Union{T, Missing}}, Z)

    # Remove the data from the missing elements
    Z₁[Iᵤ] .= missing

    return Z₁
end

"""
	encodedata(Z::A; c::T = zero(T)) where {A <: AbstractArray{Union{Missing, T}, N}} where T, N
For data `Z` with missing entries, returns an encoded data set `(U, W)` where 
`U` is the original data `Z` with missing entries replaced by a fixed constant `c`, 
and `W` encodes the missingness pattern as an indicator array 
equal to one if the corresponding element of `Z` is observed and zero otherwise.

The behavior depends on the dimensionality of `Z`. If `Z` has 1 or 2 dimensions, 
the indicator array `W` is concatenated along the first dimension of `Z`. If `Z` has more than 2 
dimensions, `W` is concatenated along the second-to-last dimension of `Z`. 

# Examples
```
using NeuralEstimators

Z = rand(16, 16, 1, 1)
Z = removedata(Z, 0.25)	# remove 25% of the data at random
UW = encodedata(Z)
```
"""
function encodedata(Z::A; c::T = zero(T)) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

    # Store the container type for later use
    ArrayType = containertype(Z)

    # Compute the indicator variable and the encoded data
    W = isnotmissing.(Z)
    U = copy(Z) # copy to avoid mutating the original data
    U[ismissing.(U)] .= c

    # Convert from eltype of U from Union{Missing, T} to T
    U = convert(Array{T, N}, U)

    # Concatenate the data and indicator variable along the appropriate dimension
    if N <= 2
        # Concatenate along the first dimension for 1D or 2D data
        UW = cat(U, W; dims = 1)
    else
        # Concatenate along the penultimate dimension for higher-dimensional data
        UW = cat(U, W; dims = N - 1)
    end

    return UW
end
isnotmissing(x) = !(ismissing(x))
