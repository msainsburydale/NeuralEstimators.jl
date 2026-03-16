


## Inference with observed data {#Inference-with-observed-data}

The following functions facilitate the use of a trained neural estimator with observed data. 
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.estimate' href='#NeuralEstimators.estimate'><span class="jlbinding">NeuralEstimators.estimate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
estimate(estimator::BayesEstimator, Z; batchsize::Integer = 32, use_gpu::Bool = true, kwargs...)
```


Applies `estimator` to data `Z` and returns the resulting estimates.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L1-L4" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.bootstrap' href='#NeuralEstimators.bootstrap'><span class="jlbinding">NeuralEstimators.bootstrap</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bootstrap(estimator::PointEstimator, parameters::P, Z; use_gpu = true) where P <: Union{AbstractMatrix, AbstractParameterSet}
bootstrap(estimator::PointEstimator, parameters::P, simulator, m::Integer; B = 400, use_gpu = true) where P <: Union{AbstractMatrix, AbstractParameterSet}
bootstrap(estimator::PointEstimator, Z; B = 400, blocks = nothing, trim = true, use_gpu = true)
```


Generates `B` bootstrap estimates using `estimator`.

Parametric bootstrapping is facilitated by passing a single parameter configuration, `parameters`, and corresponding simulated data, `Z`, whose length implicitly defines `B`. Alternatively, one may provide a `simulator` and the desired sample size, in which case the data will be simulated using `simulator(parameters, m)`.

Non-parametric bootstrapping is facilitated by passing a single data set, `Z`. The argument `blocks` caters for block bootstrapping, and it should be a vector of integers specifying the block for each replicate. For example, with 5 replicates, the first two corresponding to block 1 and the remaining three corresponding to block 2, `blocks` should be `[1, 1, 2, 2, 2]`. The resampling algorithm generates resampled data sets by sampling blocks with replacement. If `trim = true`, the final block is trimmed as needed to ensure that the resampled data set matches the original size of `Z`. 

The return type is a $d$ × `B` matrix, where $d$ is the dimension of the parameter vector. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L174-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.interval' href='#NeuralEstimators.interval'><span class="jlbinding">NeuralEstimators.interval</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
interval(θ::Matrix; probs = [0.05, 0.95], parameter_names = nothing)
interval(estimator::IntervalEstimator, Z; parameter_names = nothing, use_gpu = true)
```


Computes a confidence/credible interval based either on a $d$ × $B$ matrix `θ` of parameters (typically containing bootstrap estimates or posterior draws), where $d$ denotes the number of parameters to make inference on, or from an `IntervalEstimator` and data `Z`.

When given `θ`, the intervals are constructed by computing quantiles with probability levels controlled by the keyword argument `probs`.

The return type is a $d$ × 2 matrix, whose first and second columns respectively contain the lower and upper bounds of the interval. The rows of this matrix can be named by passing a vector of strings to the keyword argument `parameter_names`. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L102-L116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.sampleposterior' href='#NeuralEstimators.sampleposterior'><span class="jlbinding">NeuralEstimators.sampleposterior</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sampleposterior(estimator::PosteriorEstimator, Z, N::Integer = 1000; kwargs...)
sampleposterior(estimator::RatioEstimator, Z, N::Integer = 1000; θ_grid, logprior::Function = θ -> 0f0, kwargs...)
```


Samples from the approximate posterior distribution implied by `estimator`.

The positional argument `N` controls the size of the posterior sample.

If `Z` represents a single data set as determined by `Flux.numobs()`, returns a $d$ × `N` matrix of posterior samples, where $d$ is the dimension of the parameter vector. Otherwise, if `Z` contains multiple data sets, a vector of matrices will be returned. 

When using a `RatioEstimator`, the prior distribution $p(\boldsymbol{\theta})$ is controlled through the keyword argument `logprior` (by default, a uniform prior is used). The sampling algorithm is based on a fine-gridding of the parameter space, specified through the keyword argument `θ_grid`. The approximate posterior density is  evaluated over this grid, which is then used to draw samples. This is effective when making inference with a small number of parameters. For models with a large number of parameters, other sampling algorithms (e.g., MCMC) may be needed (please contact the package maintainer).

Keyword arguments are passed onto [summarystatistics](/API/estimators#NeuralEstimators.summarystatistics).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L53-L69" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.posteriormean' href='#NeuralEstimators.posteriormean'><span class="jlbinding">NeuralEstimators.posteriormean</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
posteriormean(θ::AbstractMatrix)	
posteriormean(estimator, Z, N::Integer = 1000; kwargs...)
```


Computes the posterior mean based either on a $d$ × $N$ matrix `θ` of posterior draws, where $d$ denotes the number of parameters to make inference on, or directly from an estimator that allows for posterior sampling via [`sampleposterior()`](/API/inference#NeuralEstimators.sampleposterior).

See also [`posteriormedian()`](/API/inference#NeuralEstimators.posteriormedian), [`posteriormode()`](/API/inference#NeuralEstimators.posteriormode).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L16-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.posteriormedian' href='#NeuralEstimators.posteriormedian'><span class="jlbinding">NeuralEstimators.posteriormedian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
posteriormedian(θ::AbstractMatrix)	
posteriormedian(estimator, Z, N::Integer = 1000; kwargs...)
```


Computes the vector of marginal posterior medians based either on a $d$ × $N$ matrix `θ` of posterior draws, where $d$ denotes the number of parameters to make inference on, or directly from an estimator that allows for posterior sampling via [`sampleposterior()`](/API/inference#NeuralEstimators.sampleposterior).

See also [`posteriormean()`](/API/inference#NeuralEstimators.posteriormean), [`posteriorquantile()`](/API/inference#NeuralEstimators.posteriorquantile).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L27-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.posteriorquantile' href='#NeuralEstimators.posteriorquantile'><span class="jlbinding">NeuralEstimators.posteriorquantile</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
posteriorquantile(θ::AbstractMatrix, probs)	
posteriorquantile(estimator, Z, probs, N::Integer = 1000; kwargs...)
```


Computes the vector of marginal posterior quantiles with (a collection of) probability levels `probs`, based either on a $d$ × $N$ matrix `θ` of posterior draws, where $d$ denotes the number of parameters to make inference on, or directly from an estimator that allows for posterior sampling via [`sampleposterior()`](/API/inference#NeuralEstimators.sampleposterior).

The return value is a $d$ × `length(probs)` matrix. 

See also [`posteriormedian()`](/API/inference#NeuralEstimators.posteriormedian).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L38-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.posteriormode' href='#NeuralEstimators.posteriormode'><span class="jlbinding">NeuralEstimators.posteriormode</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
posteriormode(estimator::RatioEstimator, Z; θ₀ = nothing, θ_grid = nothing, logprior::Function = θ -> 0f0, use_gpu = true)
```


Computes the (approximate) posterior mode (maximum a posteriori estimate) given data $\boldsymbol{Z}$,

$$\underset{\boldsymbol{\theta}}{\mathrm{arg\,max\;}} \ell(\boldsymbol{\theta} ; \boldsymbol{Z}) + \log p(\boldsymbol{\theta}),$$

where $\ell(\cdot ; \cdot)$ denotes the approximate log-likelihood function implied by `estimator`, and $p(\boldsymbol{\theta})$ denotes the prior density function controlled through the keyword argument `prior`. Note that this estimate can be viewed as an approximate maximum penalised likelihood estimate, with penalty term $p(\boldsymbol{\theta})$. 

If a vector `θ₀` of initial parameter estimates is given, the approximate posterior density is maximised by gradient descent (requires `Optim.jl` to be loaded). Otherwise, if a matrix of parameters `θ_grid` is given, the approximate posterior density is maximised by grid search.

See also [`posteriormedian()`](/API/inference#NeuralEstimators.posteriormedian), [`posteriormean()`](/API/inference#NeuralEstimators.posteriormean).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/inference.jl#L76-L89" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.logratio' href='#NeuralEstimators.logratio'><span class="jlbinding">NeuralEstimators.logratio</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
logratio(estimator::RatioEstimator, Z, θ_grid)
logratio(estimator::RatioEstimator, Z; θ_grid)
```


Compute the log likelihood-to-evidence ratios over a grid of parameter values `θ_grid`  for the data `Z`.

**Arguments**
- `estimator`: a `RatioEstimator`
  
- `Z`: observed data
  
- `θ_grid`: matrix of parameter values, where each column is a parameter configuration
  

**Returns**

A vector of log ratios, one for each column of `θ_grid`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/Estimators/RatioEstimator.jl#L144-L158" target="_blank" rel="noreferrer">source</a></Badge>

</details>

