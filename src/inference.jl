#TODO parallel computations in outer broadcasting functions
#TODO when we add them, these methods will be easily extended to NLE and NPE
#     (whatever methods allows a density to be evaluated)

# ---- Posterior sampling ----

#TODO Basic MCMC sampler (will proceed based on using θ₀)
@doc raw"""
	sampleposterior(estimator::RatioEstimator, Z, N::Integer = 1000; θ_grid, prior::Function = θ -> 1f0)
Samples from the approximate posterior distribution
$p(\boldsymbol{\theta} \mid \boldsymbol{Z})$ implied by `estimator`.

The positional argument `N` controls the size of the posterior sample.

The keyword agument `θ_grid` requires a (fine) gridding of the parameter
space, given as a matrix with ``p`` rows, with ``p`` the number of parameters
in the statistical model.

The prior distribution $p(\boldsymbol{\theta})$ is controlled through the keyword
argument `prior` (by default, a uniform prior is used).
"""
function sampleposterior(est::RatioEstimator,
				Z,
				N::Integer = 1000;
			    prior::Function = θ -> 1f0,
				θ_grid = nothing, theta_grid = nothing,
				# θ₀ = nothing, theta0 = nothing,
				kwargs...)

	# Check duplicated arguments that are needed so that the R interface uses ASCII characters only
	@assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
	# @assert isnothing(θ₀) || isnothing(theta0) "Only one of `θ₀` or `theta0` should be given"
	if !isnothing(theta_grid) θ_grid = theta_grid end
	# if !isnothing(theta0) θ₀ = theta0 end

	# # Check that we have either a grid to search over or initial estimates
	# @assert !isnothing(θ_grid) || !isnothing(θ₀) "Either `θ_grid` or `θ₀` should be given"
	# @assert isnothing(θ_grid) || isnothing(θ₀) "Only one of `θ_grid` and `θ₀` should be given"

	if !isnothing(θ_grid)
		θ_grid = Float32.(θ_grid) # convert for efficiency and to avoid warnings
		rZθ = vec(estimateinbatches(est, Z, θ_grid; kwargs...))
		pθ  = prior.(eachcol(θ_grid))
		density = pθ .* rZθ
		θ = StatsBase.wsample(eachcol(θ_grid), density, N; replace = true)
		reduce(hcat, θ)
	end
end
function sampleposterior(est::RatioEstimator, Z::AbstractVector, args...; kwargs...)
	sampleposterior.(Ref(est), Z, args...; kwargs...)
end

# ---- Optimisation-based point estimates ----

#TODO might be better to do this on the log-scale... can do this efficiently
#     through the relation logr(Z,θ) = logit(c(Z,θ)), that is, just apply logit
#     to the deepset object.

@doc raw"""
	mlestimate(estimator::RatioEstimator, Z; θ₀ = nothing, θ_grid = nothing, penalty::Function = θ -> 1, use_gpu = true)
Computes the (approximate) maximum likelihood estimate given data $\boldsymbol{Z}$,
```math
\argmax_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta} ; \boldsymbol{Z})
```
where $\ell(\cdot ; \cdot)$ denotes the approximate log-likelihood function
derived from `estimator`.

If a vector `θ₀` of initial parameter estimates is given, the approximate
likelihood is maximised by gradient descent. Otherwise, if a matrix of parameters
`θ_grid` is given, the approximate likelihood is maximised by grid search.

A maximum penalised likelihood estimate,

```math
\argmax_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta} ; \boldsymbol{Z}) + \log p(\boldsymbol{\theta}),
```

can be obtained by specifying the keyword argument `penalty` that defines the penalty term $p(\boldsymbol{\theta})$.

See also [`mapestimate()`](@ref) for computing (approximate) maximum a posteriori estimates.
"""
mlestimate(est::RatioEstimator, Z; kwargs...) = _maximisedensity(est, Z; kwargs...)
mlestimate(est::RatioEstimator, Z::AbstractVector; kwargs...) = reduce(hcat, mlestimate.(Ref(est), Z; kwargs...))

@doc raw"""
	mapestimate(estimator::RatioEstimator, Z; θ₀ = nothing, θ_grid = nothing, prior::Function = θ -> 1, use_gpu = true)
Computes the (approximate) maximum a posteriori estimate given data $\boldsymbol{Z}$,
```math
\argmax_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta} ; \boldsymbol{Z}) + \log p(\boldsymbol{\theta})
```
where $\ell(\cdot ; \cdot)$ denotes the approximate log-likelihood function
derived from `estimator`, and $p(\boldsymbol{\theta})$ denotes the prior density
function controlled through the keyword argument `prior`
(by default, a uniform prior is used).

If a vector `θ₀` of initial parameter estimates is given, the approximate
posterior density is maximised by gradient descent. Otherwise, if a matrix of parameters
`θ_grid` is given, the approximate posterior density is maximised by grid search.

See also [`mlestimate()`](@ref) for computing (approximate) maximum likelihood estimates.
"""
mapestimate(est::RatioEstimator, Z; kwargs...) = _maximisedensity(est, Z; kwargs...)
mapestimate(est::RatioEstimator, Z::AbstractVector; kwargs...) = reduce(hcat, mlestimate.(Ref(est), Z; kwargs...))

function _maximisedensity(
	est::RatioEstimator, Z;
	prior::Function = θ -> 1f0, penalty::Union{Function, Nothing} = nothing,
	θ_grid = nothing, theta_grid = nothing,
	θ₀ = nothing, theta0 = nothing,
	kwargs...
	)

	# Check duplicated arguments that are needed so that the R interface uses ASCII characters only
	@assert isnothing(θ_grid) || isnothing(theta_grid) "Only one of `θ_grid` or `theta_grid` should be given"
	@assert isnothing(θ₀) || isnothing(theta0) "Only one of `θ₀` or `theta0` should be given"
	if !isnothing(theta_grid) θ_grid = theta_grid end
	if !isnothing(theta0) θ₀ = theta0 end

	# Change "penalty" to "prior"
	if !isnothing(penalty) prior = penalty end

	# Check that we have either a grid to search over or initial estimates
	@assert !isnothing(θ_grid) || !isnothing(θ₀) "Either `θ_grid` or `θ₀` should be given"
	@assert isnothing(θ_grid) || isnothing(θ₀) "Only one of `θ_grid` and `θ₀` should be given"

	if !isnothing(θ_grid)

		θ_grid = Float32.(θ_grid)       # convert for efficiency and to avoid warnings
		rZθ = vec(estimateinbatches(est, Z, θ_grid; kwargs...))
		pθ  = prior.(eachcol(θ_grid))
		density = pθ .* rZθ
		θ̂ = θ_grid[:, argmax(density), :]   # extra colon to preserve matrix output

	elseif !isnothing(θ₀)

		θ₀ = Float32.(θ₀)       # convert for efficiency and to avoid warnings

		objective(θ) = -first(prior(θ) * est(Z, θ)) # closure that will be minimised

		# Gradient using reverse-mode automatic differentiation with Zygote
		# ∇objective(θ) = gradient(θ -> objective(θ), θ)[1]
		# θ̂ = optimize(objective, ∇objective, θ₀, LBFGS(); inplace = false) |> Optim.minimizer

		# Gradient using finite differences
		# θ̂ = optimize(objective, θ₀, LBFGS()) |> Optim.minimizer

		# Gradient-free NelderMead algorithm (find that this is most stable)
		θ̂ = optimize(objective, θ₀, NelderMead()) |> Optim.minimizer
	end

	return θ̂
end
_maximisedensity(est::RatioEstimator, Z::AbstractVector; kwargs...) = reduce(hcat, _maximisedensity.(Ref(est), Z; kwargs...))


# ---- Interval constructions ----

"""
	interval(θ::Matrix; probs = [0.05, 0.95], parameter_names = nothing)
	interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu = true)

Compute a confidence interval based either on a ``p`` × ``B`` matrix `θ` of
parameters (typically containing bootstrap estimates or posterior draws)
with ``p`` the number of parameters in the model, or from an `IntervalEstimator`
and data `Z`.

When given `θ`, the intervals are constructed by compute quantiles with
probability levels controlled by the keyword argument `probs`.

The return type is a ``p`` × 2 matrix, whose first and second columns respectively
contain the lower and upper bounds of the interval. The rows of this matrix can
be named by passing a vector of strings to the keyword argument `parameter_names`.

# Examples
```
using NeuralEstimators
p = 3
B = 50
θ = rand(p, B)
interval(θ)
```
"""
function interval(bs; probs = [0.05, 0.95], parameter_names = ["θ$i" for i ∈ 1:size(bs, 1)])

	p, B = size(bs)

	# Compute the quantiles
	ci = mapslices(x -> quantile(x, probs), bs, dims = 2)

	# Add labels to the confidence intervals
	l = ci[:, 1]
	u = ci[:, 2]
	labelinterval(l, u, parameter_names)
end


function interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu::Bool = true)

	ci = estimateinbatches(estimator, Z, use_gpu = use_gpu)
	ci = cpu(ci)

	if typeof(estimator) <: IntervalEstimator
		@assert size(ci, 1) % 2 == 0
		p = size(ci, 1) ÷ 2
	end

	if isnothing(parameter_names)
		parameter_names = ["θ$i" for i ∈ 1:p]
	else
		@assert length(parameter_names) == p
	end

	intervals = labelinterval(ci, parameter_names)
	if length(intervals) == 1
		intervals = intervals[1]
	end
	return intervals
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

# ---- Parametric bootstrap ----

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

	bs = estimateinbatches(θ̂, v, use_gpu = use_gpu)
	return bs
end

function bootstrap(θ̂, parameters::P, Z̃; use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	bs = estimateinbatches(θ̂, Z̃, use_gpu = use_gpu)
	return bs
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
	bs = estimateinbatches(θ̂, Z̃, use_gpu = use_gpu)

	return bs
end

# simple wrapper to handle the common case that the user forgot to extract the
# array from the single-element vector returned by a simulator
function bootstrap(θ̂, Z::V; args...) where {V <: AbstractVector{A}} where A

	@assert length(Z) == 1 "bootstrap() is designed for a single data set only"
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
