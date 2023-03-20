using StatsBase: ecdf
"""
	confidenceinterval(θ̃, θ̂ = nothing; type, probs = [0.05, 0.95], parameter_names)

Compute a confidence interval using the quantiles of the p × B matrix of
bootstrap samples, `θ̃`, where p is the number of parameters in the model.

The quantile levels are controlled with the argument `probs`. The rows can be
named with a vector of strings `parameter_names` (sensible defaults provided).

The return type is a p × 2 matrix, whose first and second columns respectively
contain the lower and upper bounds of the confidence interval.

# Examples
```
using NeuralEstimators
p = 3
B = 50
θ̃ = rand(p, B)
θ̂ = rand(p)
confidenceinterval(θ̃)
confidenceinterval(θ̃, θ̂, type = "basic")
```
"""
function confidenceinterval(θ̃, θ̂ = nothing; type::String = "percentile", probs = [0.05, 0.95], parameter_names = ["θ$i" for i ∈ 1:size(θ̃, 1)])

	#TODO add assertions and add type on θ̃
	p, B = size(θ̃)
	type = lowercase(type)

	if type ∈ ["percentile", "quantile"]
		ci = mapslices(x -> quantile(x, probs), θ̃, dims = 2)
	elseif type ∈ ["basic", "reverse-percentile", "reverse-quantile"]
		q = mapslices(x -> quantile(x, probs), θ̃, dims = 2)
		ci = [[2θ̂[i] - q[i, 2], 2θ̂[i] - q[i, 1]] for i ∈ 1:p]
		ci = hcat(ci...)'
	elseif type ∈ ["studentised", "studentized"]
		#TODO
	elseif type == "bca"
		#TODO
		# z₀ # bias-correction
		# a  # acceleration
		# G = mapslices(ecdf, θ̃, dims = 2)
	else
		error("argument `type` not matched: it should be one of 'percentile', 'basic', 'studentised', or 'bca'.")
	end

	# Add labels to the confidence intervals
	l = ci[:, 1]
	u = ci[:, 2]
	labelconfidenceinterval(l, u, parameter_names)
end

function labelconfidenceinterval(l::V, u::V, parameter_names = ["θ$i" for i ∈ length(l)]) where V <: AbstractVector
	@assert length(l) == length(u)
	NamedArray(hcat(l, u), (parameter_names, ["lower", "upper"]))
end

function labelconfidenceinterval(ci::V, parameter_names = ["θ$i" for i ∈ (length(ci) ÷ 2)]) where V <: AbstractVector

	@assert length(ci) % 2 == 0
	p = length(ci) ÷ 2
	l = ci[1:p]
	u = ci[(p+1):end]
	labelconfidenceinterval(l, u, parameter_names)
end

function labelconfidenceinterval(ci::M, parameter_names = ["θ$i" for i ∈ (size(ci, 1) ÷ 2)]) where M <: AbstractMatrix

	@assert size(ci, 1) % 2 == 0
	p = size(ci, 1) ÷ 2
	K = size(ci, 2)

	map(1:K) do k
		lₖ = ci[1:p, k]
		uₖ = ci[(p+1):end, k]
		labelconfidenceinterval(lₖ, uₖ, parameter_names)
	end
end

# ---- Parameteric bootstrap ----

"""
	bootstrap(θ̂, parameters::P, Z̃) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, parameters::P, m::Integer; B = 400) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(θ̂, Z; B = 400)
	bootstrap(θ̂, Z, blocks::Vector{Integer}; B = 400)

Generates `B` bootstrap estimates from an estimator `θ̂`.

Parametric bootstrapping is facilitated by passing a single parameter
configuration, `parameters`, and corresponding simulated data, `Z̃`, whose length
implicitly defines `B`. Alternatively, if the user has defined a method
`simulate(parameters, m)`, one may simply pass the desired sample size `m` for
the simulated data sets.

Non-parametric bootstrapping is facilitated by passing a single data set, `Z`.
The argument `blocks` caters for block bootstrapping, and it should be an integer
vector specifying the block for each replicate. For example, with 5 replicates,
the first two corresponding to block 1 and the remaining three corresponding to
block 2, `blocks` should be `[1, 1, 2, 2, 2]`. The resampling algorithm aims to
produce resampled data sets that are of a similar size to `Z`, but this can only
be achieved exactly if all blocks are equal in length.

The keyword argument `use_gpu` is a flag determining whether to use the GPU, if it is available (default `true`).

The return type is a p × `B` matrix, where p is the number of parameters in the model.
"""
function bootstrap(θ̂, parameters::P, m::Integer; B::Integer = 400, use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	Z̃ = simulate(parameters, m, B)
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃)
	return θ̃
end

function bootstrap(θ̂, parameters::P, Z; use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃)
	return θ̃
end


# ---- Non-parametric bootstrapping ----

function bootstrap(θ̂, Z; B::Integer = 400, use_gpu::Bool = true)

	# Generate B bootstrap samples of Z
	m = numberreplicates(Z)
	Z̃ = [subsetdata(Z, rand(1:m, m)) for _ in 1:B]

	# Estimate the parameters for each bootstrap sample
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃)

	return θ̃
end

# simple wrapper to handle the common case that the user forgot to extract the
# array from the single-element vector returned by simulate()
function bootstrap(θ̂, Z::V; args...) where {V <: AbstractVector{A}} where A

	@assert length(Z) == 1
	Z = Z[1]
	return bootstrap(θ̂, Z; args...)

end


function bootstrap(θ̂, Z, blocks::V; B::Integer = 400, use_gpu::Bool = true) where {V <: AbstractVector{I}} where {I <: Integer}
	Z̃ = _blockresample(Z, B, blocks)
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃)
	return θ̃
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


"""
	coverage(ci::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

Given a p×K matrix of true parameters `θ`, determine the empirical coverage of
the confidence intervals `ci` (a K-vector of px2 matrices).

The overall empirical coverage is obtained by averaging the resulting 0-1 matrix
over all data sets.
"""
function coverage(ci::V, θ) where  {V <: AbstractArray{M}} where M <: AbstractMatrix

    p, K = size(θ)
	@assert length(ci) == K
	@assert all(size.(ci, 1) .== p)
	@assert all(size.(ci, 2) .== 2)

	# for each confidence interval, determine if the true parameters, θ, are
	# within the interval.
	within = map(eachindex(ci)) do k

		c = ci[k]

		# Determine if the confidence intervals contain the true parameter.
		# The result is an indicator vector specifying which parameters are
		# contained in the interval
		[c[i, 1] < θ[i, k] < c[i, 2] for i ∈ 1:p]
	end

	# combine the counts into a single matrix p x K matrix
	within = hcat(within...)

	# compute the empirical coverage
	cvg = mean(within, dims = 2)

	return cvg
end
