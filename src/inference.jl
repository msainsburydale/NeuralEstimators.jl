"""
	estimate(estimator, Z; batchsize::Integer = 32, use_gpu::Bool = true, kwargs...)
Applies `estimator` to batches of `Z` of size `batchsize`, which can prevent memory issues that can occur with large data sets. 

Batching will only be done if there are multiple data sets in `Z`, which will be inferred by `Z` being a vector, or a tuple whose first element is a vector.
"""
function estimate(estimator, z, θ = nothing; batchsize::Integer = 32, use_gpu::Bool = true, kwargs...)

	# Convert to Float32 for numerical efficiency
	θ = f32(θ)
	z = f32(z)

	# Tupleise if necessary
  	z = isnothing(θ) ? z : (z, θ)

	# Only do batching if we have multiple data sets
	if typeof(z) <: AbstractVector
		minibatching = true
		batchsize = min(length(z), batchsize)
	elseif typeof(z) <: Tuple && typeof(z[1]) <: AbstractVector
		# Can only batch if the number of data sets in z[1] aligns with the number of sets in z[2]:
		K₁ = length(z[1])
		K₂ = typeof(z[2]) <: AbstractVector ? length(z[2]) : size(z[2], 2)
		minibatching = K₁ == K₂
		batchsize = min(K₁, batchsize)
	else # we dont have replicates: just apply the estimator without batching
		minibatching = false
	end

	device  = _checkgpu(use_gpu, verbose = false)
	estimator = estimator |> device

	if !minibatching
		z = z |> device
		ŷ = estimator(z; kwargs...)
		ŷ = ŷ |> cpu
	else
		data_loader = _DataLoader(z, batchsize, shuffle=false, partial=true)
		ŷ = map(data_loader) do zᵢ
			zᵢ = zᵢ |> device
			ŷ = estimator(zᵢ; kwargs...)
			ŷ = ŷ |> cpu
			ŷ
		end
		ŷ = stackarrays(ŷ)
	end

	return ŷ
end

# ---- Point estimation from estimators that allow for posterior sampling ----

_doc_string = """
based either on a ``d`` × ``N`` matrix `θ` of posterior draws, where ``d`` denotes the number of parameters to make inference on, or directly from an estimator that allows for posterior sampling via [`sampleposterior()`](@ref).
"""

"""
	posteriormean(θ::AbstractMatrix)	
	posteriormean(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, N::Integer = 1000; kwargs...)	
Computes the posterior mean $_doc_string

See also [`posteriormedian()`](@ref), [`posteriormode()`](@ref).
"""
posteriormean(θ::AbstractMatrix) = mean(θ; dims = 2)
posteriormean(θ::AbstractVector{<:AbstractMatrix}) = reduce(hcat, posteriormean.(θ))
posteriormean(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, N::Integer = 1000; kwargs...) = posteriormean(sampleposterior(estimator, Z, N; kwargs...))

"""
	posteriormedian(θ::AbstractMatrix)	
	posteriormedian(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, N::Integer = 1000; kwargs...)	
Computes the vector of marginal posterior medians $_doc_string

See also [`posteriormean()`](@ref), [`posteriorquantile()`](@ref).
"""
posteriormedian(θ::AbstractMatrix) = median(θ; dims = 2)
posteriormedian(θ::AbstractVector{<:AbstractMatrix}) = reduce(hcat, posteriormedian.(θ))
posteriormedian(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, N::Integer = 1000; kwargs...) = posteriormedian(sampleposterior(estimator, Z, N; kwargs...))

"""
	posteriorquantile(θ::AbstractMatrix, probs)	
	posteriormedian(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, probs, N::Integer = 1000; kwargs...)	
Computes the vector of marginal posterior quantiles with (a collection of) probability levels `probs`, $_doc_string

The return value is a ``d`` × `length(probs)` matrix. 

See also [`posteriormedian()`](@ref).
"""
function posteriorquantile(θ::AbstractMatrix, probs) 
	mapslices(row -> quantile(row, probs), θ, dims = 2)
end
posteriorquantile(estimator::Union{PosteriorEstimator, RatioEstimator}, Z, probs, N::Integer = 1000; kwargs...) = posteriorquantile(sampleposterior(estimator, Z, N; kwargs...), probs)


# ---- Posterior sampling ----

@doc raw"""
	sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000)
	sampleposterior(estimator::RatioEstimator, Z, N::Integer = 1000; θ_grid, prior::Function = θ -> 1f0)
Samples from the approximate posterior distribution implied by `estimator`.

The positional argument `N` controls the size of the posterior sample.

Returns $d$ × `N` matrix of posterior samples, where $d$ is the dimension of the parameter vector. If `Z` is a vector containing multiple data sets, a vector of matrices will be returned 

When sampling based on a `RatioEstimator`, the sampling algorithm is based on a fine-gridding of the
parameter space, specified through the keyword argument `θ_grid` (or `theta_grid`). 
The approximate posterior density is evaluated over this grid, which is then
used to draw samples. This is effective when making inference with a
small number of parameters. For models with a large number of parameters,
other sampling algorithms may be needed (please contact the package maintainer). 
The prior distribution $p(\boldsymbol{\theta})$ is controlled through the keyword argument `prior` (by default, a uniform prior is used).
"""
function sampleposterior(est::RatioEstimator,
				Z,
				N::Integer = 1000;
			    prior::Function = θ -> 1f0,
				θ_grid = nothing, theta_grid = nothing,
				# θ₀ = nothing, theta0 = nothing, #TODO Basic MCMC sampler (initialised with θ₀)
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
		θ_grid = f32(θ_grid) 
		rZθ = vec(estimate(est, Z, θ_grid; kwargs...))
		pθ  = prior.(eachcol(θ_grid))
		density = pθ .* rZθ
		θ = StatsBase.wsample(eachcol(θ_grid), density, N; replace = true)
		reduce(hcat, θ)
	end
end
function sampleposterior(est::RatioEstimator, Z::AbstractVector, N::Integer = 1000; kwargs...) 

	Z = f32(Z)

	# Simple broadcasting to handle multiple data sets (NB this could be done in parallel)
	if length(Z) == 1 
		sampleposterior(est, Z[1], N; kwargs...)
	else
		sampleposterior.(Ref(est), Z, N; kwargs...)
	end
end

sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000; kwargs...) = sampleposterior(estimator.q, estimator.network(Z), N; kwargs...)
function sampleposterior(est::PosteriorEstimator, Z::AbstractVector, N::Integer = 1000; kwargs...) 

	Z = f32(Z)

	# Simple broadcasting to handle multiple data sets (NB this could be done in parallel)
	if length(Z) == 1 
		sampleposterior(est, Z[1], N; kwargs...)
	else
		sampleposterior.(Ref(est), Z, N; kwargs...)
	end
end



# ---- Optimisation-based point estimates ----

# TODO density evaluation with PosteriorEstimator would allow us to use these grid and gradient-based methods for computing the posterior mode

@doc raw"""
	posteriormode(estimator::RatioEstimator, Z; θ₀ = nothing, θ_grid = nothing, prior::Function = θ -> 1, use_gpu = true)
Computes the (approximate) posterior mode (maximum a posteriori estimate) given data $\boldsymbol{Z}$,
```math
\underset{\boldsymbol{\theta}}{\mathrm{arg\,max\;}} \ell(\boldsymbol{\theta} ; \boldsymbol{Z}) + \log p(\boldsymbol{\theta}),
```
where $\ell(\cdot ; \cdot)$ denotes the approximate log-likelihood function implied by `estimator`, and $p(\boldsymbol{\theta})$ denotes the prior density function controlled through the keyword argument `prior`. Note that this estimate can be viewed as an approximate maximum penalised likelihood estimate, with penalty term $p(\boldsymbol{\theta})$. 

If a vector `θ₀` of initial parameter estimates is given, the approximate
posterior density is maximised by gradient descent (requires `Optim.jl` to be loaded). Otherwise, if a matrix of parameters
`θ_grid` is given, the approximate posterior density is maximised by grid search.

See also [`posteriormedian()`](@ref), [`posteriormean()`](@ref).
"""
function posteriormode(
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
	@assert !isnothing(θ_grid) || !isnothing(θ₀) "One of `θ_grid` or `θ₀` should be given"
	@assert isnothing(θ_grid) || isnothing(θ₀) "Only one of `θ_grid` and `θ₀` should be given"

	if !isnothing(θ_grid)

		θ_grid = f32(θ_grid)      
		rZθ = vec(estimate(est, Z, θ_grid; kwargs...))
		pθ  = prior.(eachcol(θ_grid))
		density = pθ .* rZθ
		θ̂ = θ_grid[:, argmax(density), :]   # extra colon to preserve matrix output

	else
		θ̂ = _optimdensity(θ₀, prior, est)
	end

	return θ̂
end
posteriormode(est::RatioEstimator, Z::AbstractVector; kwargs...) = reduce(hcat, posteriormode.(Ref(est), Z; kwargs...))


# Here, we define _optimdensity() for the case that Optim has not been loaded
# For the case that Optim is loaded, _optimdensity() is overloaded in ext/NeuralEstimatorsOptimExt.jl
# NB Julia complains if we overload functions in package extensions... to get around this, here we
# use a slightly different function signature (omitting ::Function)
function _optimdensity(θ₀, prior, est)
	error("A vector of initial parameter estimates has been provided, indicating that the approximate likelihood or posterior density will be maximised by numerical optimisation; please load the Julia package `Optim` to facilitate this")
end


# ---- Interval constructions ----

"""
	interval(θ::Matrix; probs = [0.05, 0.95], parameter_names = nothing)
	interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu = true)
Computes a confidence/credible interval based either on a ``d`` × ``B`` matrix `θ` of
parameters (typically containing bootstrap estimates or posterior draws),
where ``d`` denotes the number of parameters to make inference on, or from an `IntervalEstimator`
and data `Z`.

When given `θ`, the intervals are constructed by computing quantiles with
probability levels controlled by the keyword argument `probs`.

The return type is a ``d`` × 2 matrix, whose first and second columns respectively
contain the lower and upper bounds of the interval. The rows of this matrix can
be named by passing a vector of strings to the keyword argument `parameter_names`. 
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

	ci = estimate(estimator, Z, use_gpu = use_gpu)
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
	bootstrap(estimator::PointEstimator, parameters::P, Z; use_gpu = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(estimator::PointEstimator, parameters::P, simulator, m::Integer; B = 400, use_gpu = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	bootstrap(estimator::PointEstimator, Z; B = 400, blocks = nothing, trim = true, use_gpu = true)
Generates `B` bootstrap estimates using `estimator`.

Parametric bootstrapping is facilitated by passing a single parameter
configuration, `parameters`, and corresponding simulated data, `Z`, whose length
implicitly defines `B`. Alternatively, one may provide a `simulator` and the
desired sample size, in which case the data will be simulated using
`simulator(parameters, m)`.

Non-parametric bootstrapping is facilitated by passing a single data set, `Z`.
The argument `blocks` caters for block bootstrapping, and it should be a vector
of integers specifying the block for each replicate. For example, with 5 replicates,
the first two corresponding to block 1 and the remaining three corresponding to
block 2, `blocks` should be `[1, 1, 2, 2, 2]`. The resampling algorithm generates resampled data sets by sampling blocks with replacement. If `trim = true`, the final block is trimmed as needed to ensure that the resampled data set matches the original size of `Z`. 

The return type is a ``d`` × `B` matrix, where ``d`` is the dimension of the parameter vector. 
"""
function bootstrap(estimator, parameters::P, simulator, m::Integer; B::Integer = 400, use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
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

	bs = estimate(estimator, v, use_gpu = use_gpu)
	return bs
end

function bootstrap(estimator, parameters::P, Z̃; use_gpu::Bool = true) where P <: Union{AbstractMatrix, ParameterConfigurations}
	K = size(parameters, 2)
	@assert K == 1 "Parametric bootstrapping is designed for a single parameter configuration only: received `size(parameters, 2) = $(size(parameters, 2))` parameter configurations"
	bs = estimate(estimator, Z̃, use_gpu = use_gpu)
	return bs
end

# ---- Non-parametric bootstrapping ----

function bootstrap(estimator, Z; B::Integer = 400, use_gpu::Bool = true, blocks = nothing, trim::Bool = true)

	
	@assert !(typeof(Z) <: Tuple) "bootstrap() is not currently set up for dealing with set-level information; please contact the package maintainer"

	# Generate B bootstrap data sets 
	if !isnothing(blocks)
		@assert length(blocks) == numberreplicates(Z) "The number of replicates and the length of `blocks` must match: recieved `numberreplicates(Z) = $(numberreplicates(Z))` and `length(blocks) = $(length(blocks))`"
		m = length(blocks)
		unique_blocks = unique(blocks)
		block_counts = Dict(b => count(==(b), blocks) for b in unique_blocks)
		Z̃ = map(1:B) do _
			sampled_blocks = Int[]
			m̃ = 0
			while m̃ < m
				b = rand(unique_blocks)
				push!(sampled_blocks, b)
				m̃ += block_counts[b]
			end
			idx = vcat([findall(==(b), blocks) for b in sampled_blocks]...)
			if trim && length(idx) > m
				idx = idx[1:m]  # trim to match original sample size
			end
			subsetdata(Z, idx)
		end	
	else
		m = numberreplicates(Z)
		Z̃ = [subsetdata(Z, rand(1:m, m)) for _ in 1:B]
	end

	# Estimate the parameters for each bootstrap sample
	bs = estimate(estimator, Z̃, use_gpu = use_gpu)

	return bs
end

# simple wrapper to handle the common case that the user forgot to extract the
# array from the single-element vector returned by a simulator
function bootstrap(estimator, Z::V; args...) where {V <: AbstractVector{A}} where A
	@assert length(Z) == 1 "bootstrap() is designed for a single data set only"
	Z = Z[1]
	return bootstrap(estimator, Z; args...)
end