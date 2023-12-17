"""
	interval(θ̃::Matrix; probs = [0.05, 0.95], parameter_names = nothing)
	interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu = true)

Compute a confidence interval based on a p × B matrix of bootstrap estimates, `θ̃`,
where p is the number of parameters in the model, or from an `IntervalEstimator`
and data `Z`.

The bootstrap-based interval is constructed by taking the quantiles of `θ̃`,
where the quantile levels are controlled by the keyword argument `probs`.

The return type is a p × 2 matrix, whose first and second columns respectively
contain the lower and upper bounds of the interval. The rows of this matrix can
be named by passing a vector of strings to the keyword argument `parameter_names`.

# Examples
```
using NeuralEstimators
p = 3
B = 50
θ̃ = rand(p, B)
θ̂ = rand(p)
interval(θ̃)
```
"""
function interval(θ̃; probs = [0.05, 0.95], parameter_names = ["θ$i" for i ∈ 1:size(θ̃, 1)])

	p, B = size(θ̃)

	# Compute the quantiles
	ci = mapslices(x -> quantile(x, probs), θ̃, dims = 2)

	# Add labels to the confidence intervals
	l = ci[:, 1]
	u = ci[:, 2]
	labelinterval(l, u, parameter_names)
end


function interval(estimator::Union{IntervalEstimator, IntervalEstimatorCompactPrior, PointIntervalEstimator}, Z; parameter_names = nothing, use_gpu::Bool = true)

	ci = estimateinbatches(estimator, Z, use_gpu = use_gpu)
	ci = cpu(ci)

	if typeof(estimator) <: IntervalEstimator || typeof(estimator) <: IntervalEstimatorCompactPrior
		@assert size(ci, 1) % 2 == 0
		p = size(ci, 1) ÷ 2
	elseif typeof(estimator) <: PointIntervalEstimator
		@assert size(ci, 1) % 3 == 0
		p = size(ci, 1) ÷ 3
		ci = ci[p+1:end, :]
	end

	if isnothing(parameter_names)
		parameter_names = ["θ$i" for i ∈ 1:p]
	else
		@assert length(parameter_names) == p
	end

	labelinterval(ci, parameter_names)
end


function labelinterval(l::V, u::V, parameter_names = ["θ$i" for i ∈ length(l)]) where V <: AbstractVector
	@assert length(l) == length(u)
	NamedArray(hcat(l, u), (parameter_names, ["lower", "upper"]))
end

function labelinterval(ci::V, parameter_names = ["θ$i" for i ∈ (length(ci) ÷ 2)]) where V <: AbstractVector

	@assert length(ci) % 2 == 0
	p = length(ci) ÷ 2
	l = ci[1:p]
	u = ci[(p+1):end]
	labelinterval(l, u, parameter_names)
end

function labelinterval(ci::M, parameter_names = ["θ$i" for i ∈ (size(ci, 1) ÷ 2)]) where M <: AbstractMatrix

	@assert size(ci, 1) % 2 == 0
	p = size(ci, 1) ÷ 2
	K = size(ci, 2)

	[labelinterval(ci[:, k], parameter_names) for k ∈ 1:K]
end

# ---- Parameteric bootstrap ----

"""
	bootstrap(θ̂, parameters::P, Z) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, parameters::P, simulator, m::Integer; B = 400) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, Z; B = 400, blocks = nothing)

Generates `B` bootstrap estimates from an estimator `θ̂`.

Parametric bootstrapping is facilitated by passing a single parameter
configuration, `parameters`, and corresponding simulated data, `Z`, whose length
implicitly defines `B`. Alternatively, one may provide a `simulator` and the
desired sample size, in which case the data will be simulated using
`simulator(parameters, m)`.

Non-parametric bootstrapping is facilitated by passing a single data set, `Z`.
The argument `blocks` caters for block bootstrapping, and it should be a vector
of integers specifying the block for each replicate. For example, with 5 replicates,
the first two corresponding to block 1 and the remaining three corresponding to
block 2, `blocks` should be `[1, 1, 2, 2, 2]`. The resampling algorithm aims to
produce resampled data sets that are of a similar size to `Z`, but this can only
be achieved exactly if all blocks are equal in length.

The keyword argument `use_gpu` is a flag determining whether to use the GPU,
if it is available (default `true`).

The return type is a p × `B` matrix, where p is the number of parameters in the model.
"""
function bootstrap(θ̂, parameters::P, simulator, m::Integer; B::Integer = 400, use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"

	# simulate the data
	v = [simulator(parameters, m) for i ∈ 1:B]
	if typeof(v[1]) <: Tuple
		z = vcat([v[i][1] for i ∈ eachindex(v)]...)
		x = vcat([v[i][2] for i ∈ eachindex(v)]...)
		v = (z, x)
	else
		v = vcat(v...)
	end

	θ̃ = estimateinbatches(θ̂, v, use_gpu = use_gpu)
	return θ̃
end

function bootstrap(θ̂, parameters::P, Z̃; use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	θ̃ = estimateinbatches(θ̂, Z̃, use_gpu = use_gpu)
	return θ̃
end


# ---- Non-parametric bootstrapping ----

function bootstrap(θ̂, Z; B::Integer = 400, use_gpu::Bool = true, blocks = nothing)

	# Generate B bootstrap samples of Z
	if !isnothing(blocks)
		Z̃ = _blockresample(Z, B, blocks)
	else
		m = numberreplicates(Z)
		Z̃ = [subsetdata(Z, rand(1:m, m)) for _ in 1:B]
	end
	# Estimate the parameters for each bootstrap sample
	θ̃ = estimateinbatches(θ̂, Z̃, use_gpu = use_gpu)

	return θ̃
end

# simple wrapper to handle the common case that the user forgot to extract the
# array from the single-element vector returned by a simulator
function bootstrap(θ̂, Z::V; args...) where {V <: AbstractVector{A}} where A

	@assert length(Z) == 1
	Z = Z[1]
	return bootstrap(θ̂, Z; args...)
end

"""
Generates `B` bootstrap samples by sampling `Z` with replacement, with the
replicates grouped together in `blocks`, integer vector specifying the block for
each replicate.

For example, with 5 replicates, the first two corresponding to block 1 and the
remaining three corresponding to block 2, `blocks` should be `[1, 1, 2, 2, 2]`.

The resampling algorithm aims to produce data sets that are of a similar size to
`Z`, but this can only be achieved exactly if the blocks are of equal size.
"""
function _blockresample(Z, B::Integer, blocks)

	@assert length(blocks) == numberreplicates(Z) "The number of replicates and the length of `blocks` must match: we recieved `numberreplicates(Z) = $(numberreplicates(Z))` and `length(blocks) = $(length(blocks))`"

	m = length(blocks)
	unique_blocks = unique(blocks)
	num_blocks    = length(unique_blocks)

	# Define c ≡ median(block_counts)/2 and d ≡ maximum(block_counts).
	# The following method ensures that m̃ ∈ [m - c, m - c + d), where
	# m is the sample size  (with respect to the number of independent replicates)
	# and m̃ is the sample size of the resampled data set.

	block_counts = [count(x -> x == i, blocks) for i ∈ unique_blocks]
	c = median(block_counts) / 2
	Z̃ = map(1:B) do _
		sampled_blocks = Int[]
		m̃ = 0
		while m̃ < m - c
			push!(sampled_blocks, rand(unique_blocks))
			m̃ += block_counts[sampled_blocks[end]]
		end
		idx = vcat([findall(x -> x == i, blocks) for i ∈ sampled_blocks]...)
		subsetdata(Z, idx)
	end

	return Z̃
end
