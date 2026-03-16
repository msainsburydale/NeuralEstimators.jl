


## Post-training assessment {#Post-training-assessment}

The function [`assess`](/API/assessment#NeuralEstimators.assess) can be used to assess a trained estimator. The resulting [`Assessment`](/API/assessment#NeuralEstimators.Assessment) object contains ground-truth parameters, estimates, and other quantities that can be used to compute quantitative and qualitative diagnostics.
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.assess' href='#NeuralEstimators.assess'><span class="jlbinding">NeuralEstimators.assess</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
assess(estimator, θ, Z; ...)
assess(estimators::Vector, θ, Z; ...)
```


Assesses an `estimator` (or a collection of `estimators`) based on true parameters `θ` and corresponding simulated data `Z`.

The parameters `θ` should be given as a $d$ × $K$ matrix, where $d$ is the parameter dimension and $K$ is the number of sampled parameter vectors.

When `Z` contains more simulated data sets than the number $K$ of sampled parameter vectors, `θ` will be recycled via horizontal concatenation: `θ = repeat(θ, outer = (1, J))`, where `J = numobs(Z) ÷ K` is the number of simulated data sets for each parameter vector. This allows assessment of the estimator&#39;s sampling distribution under fixed parameters.

The return value is of type [`Assessment`](/API/assessment#NeuralEstimators.Assessment).

**Keyword arguments**
- `estimator_name::String` (or `estimator_names::Vector{String}` for multiple estimators): name(s) of the estimator(s) (sensible defaults provided).
  
- `parameter_names::Vector{String}`: names of the parameters (sensible default provided).
  
- `use_gpu = true`: `Bool` or `Vector{Bool}` with length equal to the number of estimators.
  
- `probs = nothing` (applicable only to [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator)): probability levels taking values between 0 and 1. By default, no bootstrap uncertainty quantification is done; if `probs` is provided, it must be a two-element vector specifying the lower and upper probability levels for non-parametric bootstrap intervals (note that parametric bootstrap is not currently supported with `assess()`).
  
- `B::Integer = 400` (applicable only to [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator)): number of bootstrap samples.
  
- `pointsummary::Function = mean` (applicable only to estimators that yield posterior samples): a function that summarises a vector of posterior samples into a single point estimate for each marginal; any function mapping a vector to a scalar is valid (e.g., `median` for the posterior median).
  
- `N::Integer = 1000` (applicable only to estimators that yield posterior samples): number of posterior samples drawn for each data set.
  
- `kwargs...` (applicable only to estimators that yield posterior samples): additional keyword arguments passed to [`sampleposterior`](/API/inference#NeuralEstimators.sampleposterior).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L102-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.Assessment' href='#NeuralEstimators.Assessment'><span class="jlbinding">NeuralEstimators.Assessment</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Assessment
```


A type for storing the output of `assess()`. The field `runtime` contains the total time taken for each estimator. The field `estimates` is a long-form `DataFrame` with columns:
- `parameter`: the name of the parameter
  
- `truth`:     the true value of the parameter
  
- `estimate`:  the estimated value of the parameter
  
- `k`:         the index of the parameter vector
  
- `j`:         the index of the data set (only relevant in the case that multiple data sets are associated with each parameter vector)
  

If the estimator is a [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator) or a [`RatioEstimator`](/API/estimators#NeuralEstimators.RatioEstimator), in addition to the fields listed above, the field `samples` stores the posterior samples as a long-form `DataFrame` with the columns `parameter`, `truth`, `k`, `j` (as given above), as well as:
- `draw`: the index of the draw within the posterior samples
  
- `value`: the value of the posterior sample for a given parameter and draw.
  

If the estimator is an [`IntervalEstimator`](/API/estimators#NeuralEstimators.IntervalEstimator), the column `estimate` will be replaced by the columns `lower` and `upper`, containing the lower and upper bounds of the interval, respectively.

If the estimator is a [`QuantileEstimator`](/API/estimators#NeuralEstimators.QuantileEstimator), there will also be a column `prob` indicating the probability level of the corresponding quantile estimate.

Use `merge()` to combine assessments from multiple estimators of the same type or `join()` to combine assessments from a [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator) and an [`IntervalEstimator`](/API/estimators#NeuralEstimators.IntervalEstimator).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L1-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='Makie.plot-Tuple{Assessment}' href='#Makie.plot-Tuple{Assessment}'><span class="jlbinding">Makie.plot</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
plot(assessment::Assessment; prob = 0.99)
```


Visualise the performance of a neural estimator. Accepts the `Assessment` object returned by [`assess`](/API/assessment#NeuralEstimators.assess).

::: tip Extension

This function is defined in the `NeuralEstimatorsPlottingMakieExt` extension and requires `CairoMakie` (or another Makie backend) to be loaded.

:::

The plot produced depends on the type of estimator being assessed:

[`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator): produces a scatter plot of estimates vs. true values, faceted by parameter.

[`IntervalEstimator`](/API/estimators#NeuralEstimators.IntervalEstimator): produces a plot of estimated credible intervals vs. true values, faceted by parameter. Each interval is drawn as a vertical line segment from lower to upper bound, with tick marks at the endpoints.

[`QuantileEstimator`](/API/estimators#NeuralEstimators.QuantileEstimator): produces a calibration plot of the empirical coverage probability vs. the nominal probability level τ, faceted by parameter. A well-calibrated estimator will follow the red diagonal line. Specifically, the diagnostic is constructed as follows:
1. For k = 1,…,K, sample pairs (θᵏ, Zᵏ) with θᵏ ∼ p(θ), Zᵏ ∼ p(Z ∣ θᵏ). This gives K &quot;posterior draws&quot;, θᵏ ∼ p(θ ∣ Zᵏ).
  
2. For each k and each τ ∈ {τⱼ : j = 1,…,J}, estimate the posterior quantile Q(Zᵏ, τ).
  
3. For each τ, compute the proportion of quantiles Q(Zᵏ, τ) exceeding the corresponding θᵏ, and plot this proportion against τ.
  

[`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator): produces a three-row figure:
1. **Recovery plot**: posterior mean vs. true value (scatter), with vertical line segments showing the 95% posterior credible interval, faceted by parameter.
  
2. **ECDF plot**: for each parameter, the empirical CDF of the fractional rank of the true value within the posterior samples, together with a simultaneous `prob`-level confidence band. A well-calibrated posterior yields an ECDF that stays within the band.
  
3. **Z-score / contraction plot**: posterior z-score (posterior mean − truth) / posterior SD vs. posterior contraction 1 − Var(posterior) / Var(prior), faceted by parameter. Ideally z-scores are centred near zero and contractions are near one.
  

**Keyword arguments**
- `prob = 0.99`: nominal simultaneous coverage level for the SBC confidence band. Only used when `assessment` contains posterior samples.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/ext/NeuralEstimatorsPlottingMakieExt.jl#L50-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.risk' href='#NeuralEstimators.risk'><span class="jlbinding">NeuralEstimators.risk</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
risk(assessment::Assessment; ...)
```


Computes a Monte Carlo approximation of an estimator&#39;s Bayes risk,

$$r(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K L(\boldsymbol{\theta}^{(k)}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)})),$$

where $\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}$ denotes a set of $K$ parameter vectors sampled from the prior and, for each $k$, data $\boldsymbol{Z}^{(k)}$ are simulated from the statistical model conditional on $\boldsymbol{\theta}^{(k)}$.

If the `Assessment` object corresponds to an estimator with a self-defined loss (e.g., [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator)), the precomputed risk is returned directly. Otherwise, the risk is computed from the estimates and true parameters using the provided `loss` function.

**Keyword arguments**
- `loss = (x, y) -> abs(x - y)`: a binary operator defining the loss function (default: absolute-error loss)
  
- `average_over_parameters::Bool = false`: if `true`, the loss is averaged over all parameters; otherwise (default), it is computed separately for each parameter.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L428-L447" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.bias' href='#NeuralEstimators.bias'><span class="jlbinding">NeuralEstimators.bias</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bias(assessment::Assessment; average_over_parameters = false)
```


Computes a Monte Carlo approximation of an estimator&#39;s bias,

$${\textrm{bias}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\},$$

where $\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}$ denotes a set of $K$ parameter vectors sampled from the prior and, for each $k$, data $\boldsymbol{Z}^{(k)}$ are simulated from the statistical model conditional on $\boldsymbol{\theta}^{(k)}$.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L462-L475" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.rmse' href='#NeuralEstimators.rmse'><span class="jlbinding">NeuralEstimators.rmse</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
rmse(assessment::Assessment; average_over_parameters = false)
```


Computes a Monte Carlo approximation of an estimator&#39;s root-mean-squared error,

$${\textrm{rmse}}(\hat{\boldsymbol{\theta}}(\cdot))
\approx
\sqrt{\frac{1}{K} \sum_{k=1}^K \{\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(k)}) - \boldsymbol{\theta}^{(k)}\}^2},$$

where $\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}$ denotes a set of $K$ parameter vectors sampled from the prior and, for each $k$, data $\boldsymbol{Z}^{(k)}$ are simulated from the statistical model conditional on $\boldsymbol{\theta}^{(k)}$.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L487-L500" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.coverage' href='#NeuralEstimators.coverage'><span class="jlbinding">NeuralEstimators.coverage</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
coverage(assessment::Assessment; ...)
```


Computes a Monte Carlo approximation of an interval estimator&#39;s expected coverage, as defined in [Hermans et al. (2022, Definition 2.1)](https://arxiv.org/abs/2110.06581), and the proportion of parameters below and above the lower and upper bounds, respectively.

**Keyword arguments**
- `average_over_parameters::Bool = false`: if true, the coverage is averaged over all parameters; otherwise (default), it is computed over each parameter separately.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/assess.jl#L512-L521" target="_blank" rel="noreferrer">source</a></Badge>

</details>

