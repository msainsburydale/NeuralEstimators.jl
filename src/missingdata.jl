"""
    NeuralEM(simulateconditional::Function, neuralMAP::NeuralPointEstimator, θ₀ = nothing)

A type that implements the neural expectation-maximisation (EM) algorithm using
a function for conditional simulation over the missing values
(`simulateconditional`), a neural approximation of the MAP estimator
(`neuralMAP`), and a vector of starting parameter values (`θ₀`).

# Fields of `NeuralEM` objects

The function `simulateconditional` should be of the form,

	simulateconditional(Z::A, θ, ξ = nothing; nsims::Integer = 1) where {A <: AbstractArray{Union{Missing, T}}} where T

The data `Z` should be returned in whatever form is amenable to the architecture
of the neural MAP estimator. For instance, if the data
are gridded and the neural MAP estimator is based on a CNN architecture, then
`Z` should be returned as a four-dimensional array.

The `neuralMAP` estimator should be a neural point estimator trained to approximate
the joint posterior mode.

The starting value `θ₀` should be a vector, and it can be provided either during
construction of the `NeuralEM` object, or when applying the `NeuralEM` object to
data (see below). The starting values given in a function call take precedence
over those stored in the object.

# Methods

Once constructed, obects of type `NeuralEM` can be applied to data via the method,

	(neuralem::NeuralEM)(
		Z::A, θ₀ = nothing;
		niterations::Integer = 50,
		nsims::Integer = 1,
		ξ = nothing,
		ϵ = 0.01,
		return_iterates::Bool = false,
		use_gpu::Bool = true,
		verbose::Bool = false
	)  where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

The key arguments are:

- `Z`: the complete-data vector containing the observed data Z₁ and `Missing` values in the missing component Z₂. The last dimension contains the replicates (if any); the other dimensions store the response variable.
- `θ₀`: starting parameter values.
- `niterations`: the maximum number of iterations to apply the algorithm for.
- `nsims`: the number of conditional replicates used to approximate the conditional expectation (should align with the number of replicates that was used during training of the neural MAP estimator).
- `ξ`: model information needed for conditional simulation (e.g., distance matrices).
- `ϵ`: tolerance used to assess convergence.
- `return_iterates`: if `true`, the estimate at each iteration of the algorithm is returned; otherwise, only the final estimate is returned.

The algorithm is stopped after `niterations` iterations or if the relative
change in parameter values from successive iterations is sufficiently small,
specifically, if

```math
max_k (|θ_k^{(l+1)} - θ_k^{(l)}| / |θ_k^{(l)}|) < ϵ.
```

The following wrapper can be used when one has multiple data sets:

	(neuralem::NeuralEM)(Z::V, θ₀::Union{Vector, Matrix, Nothing} = nothing; args...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

Here, the starting values `θ₀` will be repeated if given as a vector.

# Examples
```
# Please see the example given in the main documentation.
```
"""
struct NeuralEM{F,T,S}
	simulateconditional::F
	neuralMAP::T
	θ₀::S
end
NeuralEM(simulateconditional, neuralMAP) = NeuralEM(simulateconditional, neuralMAP, nothing)
@functor NeuralEM (NeuralMAP,)

function (neuralem::NeuralEM)(Z::A, θ₀ = nothing; args...)  where {A <: AbstractArray{T, N}} where {T, N}
	@warn "Data has been passed to the EM algorithm that contains no missing elements... the neural MAP estimator will be applied directly to the data"
	neuralem.neuralMAP(Z)
end

function (neuralem::NeuralEM)(
	Z::A, θ₀ = nothing;
	niterations::Integer = 50,
	nsims::Integer = 1,
	ϵ = 0.01,
	ξ = nothing,
	use_gpu::Bool = true,
	verbose::Bool = false,
	return_iterates::Bool = false
	)  where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	if isnothing(θ₀)
		@assert !isnothing(neuralem.θ₀) "Please provide initial estimates θ₀ in the function call when applying the `NeuralEM` object, or in the `NeuralEM` object itself."
		θ₀ = neuralem.θ₀
	end

	@assert !all(ismissing.(Z))  "The data `Z` consists of missing elements only"

	device    = _checkgpu(use_gpu, verbose = verbose)
	neuralMAP = neuralem.neuralMAP |> device

	verbose && @show θ₀
    θₗ = θ₀
	θ_all = reshape(θ₀, :, 1)
	for l ∈ 1:niterations

		# "Complete" the data by simulating missing data conditionally on the
		# incomplete observed data and the current parameters
		Z̃ = isnothing(ξ) ? neuralem.simulateconditional(Z, θₗ, nsims = nsims) : neuralem.simulateconditional(Z, θₗ, ξ, nsims = nsims)
		Z̃ = Z̃ |> device

		# Apply the nerual MAP estimator to the complete data
		θₗ₊₁ = neuralMAP(Z̃)

		# Move back to the cpu (need to do this for simulateconditional in the next iteration)
		θₗ₊₁   = cpu(θₗ₊₁)
		θ_all = hcat(θ_all, θₗ₊₁)

		if maximum(abs.(θₗ₊₁-θₗ)./abs.(θₗ)) < ϵ
			verbose && @info "The EM algorithm has converged"
			θₗ = θₗ₊₁
			break
		end

		l == niterations && verbose && @warn "The EM algorithm has failed to converge"

		θₗ = θₗ₊₁

		verbose && @show θₗ
	end

    return_iterates ? (θₗ, θ_all) : θₗ
end

function (neuralem::NeuralEM)(Z::V, θ₀::Union{Vector, Matrix, Nothing} = nothing; args...) where {V <: AbstractVector{A}} where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	if isnothing(θ₀)
		@assert !isnothing(neuralem.θ₀) "Please provide initial estimates θ₀ either in the function call or in the NeuralEM object."
		θ₀ = neuralem.θ₀
	end

	if isa(θ₀, Vector)
		θ₀ = repeat(θ₀, 1, length(Z))
	end

	estimates = map(eachindex(Z)) do i # NB should we use Folds.map() or another parallel function?
		neuralem(Z[i], θ₀[:, i]; args...)
	end
	estimates = hcat(estimates...)

	return estimates
end


"""
	removedata(Z::Array, Iᵤ::Vector{Integer})
	removedata(Z::Array, p::Union{Float, Vector{Float}}; prevent_complete_missing = true)
	removedata(Z::Array, n::Integer; fixed_pattern = false, contiguous_pattern = false, variable_proportion = false)

Replaces elements of `Z` with `missing`.

The simplest method accepts are of integers `Iᵤ` that give the specific indices
of the data to be removed.

Alterntivaly, there are two methods available to generate data that are
missing completely at random (MCAR).

First, a vector `p` may be given that specifies the proportion of missingness
for each element in the response vector. Hence, `p` should have length equal to
the dimension of the response vector. If a single proportion is given, it
will be replicated accordingly. If `prevent_complete_missing = true`, no
replicates will contain 100% missingness (note that this can slightly alter the
effective values of `p`).

Second, if an integer `n` is provided, all replicates will contain
`n` observations after the data are removed. If `fixed_pattern = true`, the
missingness pattern is fixed for all replicates. If `contiguous_pattern = true`,
the data will be removed in a contiguous block. If `variable_proportion = true`,
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
	if isa(Z, Vector) Z = reshape(Z, :, 1) end
	m = size(Z)[end]           # number of replicates
	d = prod(size(Z)[1:end-1]) # dimension of each replicate  NB assumes a singleton channel dimension

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
				start = StatsBase.sample(1:n+1, 1)[1]
				Iᵤ = start:(start+(d-n)-1)
				Iᵤ = [Iᵤ .+ (i-1) * d for i ∈ 1:m]
			else
				Iᵤ = StatsBase.sample(1:d, d-n, replace = false)
				Iᵤ = [Iᵤ .+ (i-1) * d for i ∈ 1:m]
			end
		else
			if contiguous_pattern
				Iᵤ = map(1:m) do i
					start = (StatsBase.sample(1:n+1, 1) .+ (i-1) * d)[1]
					start:(start+(d-n)-1)
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
	if isa(Z, Vector) Z = reshape(Z, :, 1) end
	d = prod(size(Z)[1:end-1]) # dimension of each replicate  NB assumes singleton channel dimension
	p = repeat([p], d)
	return removedata(Z, p; args...)
end
function removedata(Z::V, p::F; args...) where {V <: AbstractVector{T}} where {T, F <: AbstractFloat}
	removedata(reshape(Z, :, 1), p)[:]
end


function removedata(Z::A, p::Vector{F}; prevent_complete_missing::Bool = true) where {A <: AbstractArray{T, N}} where {T, N, F <: AbstractFloat}
	if isa(Z, Vector) Z = reshape(Z, :, 1) end
	m = size(Z)[end]           # number of replicates
	d = prod(size(Z)[1:end-1]) # dimension of each replicate  NB assumes singleton channel dimension
	@assert length(p) == d "The length of `p` should equal the dimenison d of each replicate"
	multivariatebernoulli = Product([Bernoulli(p[i]) for i ∈ eachindex(p)])

	if all(p .== 1) prevent_complete_missing = false end

	if prevent_complete_missing
		Iᵤ = map(1:m) do _
			complete_missing = true
			while complete_missing
				Iᵤ = rand(multivariatebernoulli)
				complete_missing = !(0 ∈ Iᵤ)
			end
			Iᵤ
		end
	else
		Iᵤ = [rand(multivariatebernoulli) for _ ∈ 1:m]
	end

	Iᵤ = stackarrays(Iᵤ)
	Iᵤ = findall(Iᵤ)

	return removedata(Z, Iᵤ)
end
function removedata(Z::V, p::Vector{F}; args...) where {V <: AbstractVector{T}} where {T, F <: AbstractFloat}
	removedata(reshape(Z, :, 1), p)[:]
end


function removedata(Z::A, Iᵤ::V) where {A <: AbstractArray{T, N}, V <: AbstractVector{I}} where {T, N, I <: Integer}

	# Convert the Array to a type that allows missing data
	Z₁ = convert(Array{Union{T, Missing}}, Z)

	# Remove the data from the missing elements
	Z₁[Iᵤ] .= missing

	return Z₁
end



"""
	encodedata(Z::A; fixed_constant::T = zero(T)) where {A <: AbstractArray{Union{Missing, T}, N}} where T, N
For data `Z` with missing entries, returns an augmented data set (U, W) where
W encodes the missingness pattern as an indicator vector and U is the original data Z
with missing entries replaced by a `fixed_constant`.

The indicator vector W is stored in the second-to-last dimension of `Z`, which
should be a singleton. If the second-to-last dimension is not singleton, then
two singleton dimensions will be added to the array, and W will be stored in
the new second-to-last dimension.

# Examples
```
using NeuralEstimators

# Generate some missing data
Z = rand(16, 16, 1, 1)
Z = removedata(Z, 0.25)	 # remove 25% of the data

# Encode the data
UW = encodedata(Z)
```
"""
function encodedata(Z::A; fixed_constant::T = zero(T)) where {A <: AbstractArray{Union{Missing, T}, N}} where {T, N}

	# Store the container type for later use
	ArrayType = containertype(Z)

	# Make some space for the indicator variable
	if N == 1 || size(Z, N-1) != 1
		Z = reshape(Z, (size(Z)..., 1, 1))
		Ñ = N + 2
	else
		Ñ = N
	end

	# Compute the indicator variable and the augmented data
	W = isnotmissing.(Z)
	U = copy(Z) # copy to avoid mutating the original data
	U[ismissing.(U)] .= fixed_constant

	# Convert from eltype of U from Union{Missing, T} to T
	# U = convert(Array{T, N}, U) # NB this doesn't work if Z was modified in the if statement
	U = convert(ArrayType{T, Ñ}, U)

	# Combine the augmented data and the indicator variable
	UW = cat(U, W; dims = Ñ - 1)

	return UW
end
isnotmissing(x) = !(ismissing(x))
