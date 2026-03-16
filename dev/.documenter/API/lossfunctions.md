


# Loss functions {#Loss-functions}

When training an estimator of type [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator), a loss function must be specified that determines the Bayes estimator that will be approximated. In addition to the standard loss functions provided by `Flux` (e.g., `mae`, `mse`, which allow for the approximation of posterior medians and means, respectively), the following loss functions are provided with the package. 
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.tanhloss' href='#NeuralEstimators.tanhloss'><span class="jlbinding">NeuralEstimators.tanhloss</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
tanhloss(θ̂, θ, κ; joint::Bool = true, scale_by_parameter_dim::Bool = true)
```


For `κ` &gt; 0, computes the loss function given in [Sainsbury-Dale et al. (2025; Eqn. 14)](https://arxiv.org/abs/2501.04330), namely,

$$L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \tanh\big(\big\|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1/\kappa\big),$$

which yields the 0-1 loss function in the limit `κ` → 0.

If `joint = true` (default), the L₁ norm is computed over each parameter vector, so that with `κ` close to zero, the resulting Bayes estimator approximates the mode of the joint posterior distribution. Otherwise, if `joint = false`, the loss function is computed as 

$$L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \sum_{i=1}^d \tanh\big(|\hat{\theta}_i - \theta_i|/\kappa\big),$$

where $d$ denotes the dimension of the parameter vector $\boldsymbol{\theta}$. In this case, with `κ` close to zero, the resulting Bayes estimator approximates the vector containing the modes of the marginal posterior distributions.

Compared with the [`kpowerloss()`](/API/lossfunctions#NeuralEstimators.kpowerloss), which may also be used as a continuous approximation of the 0–1 loss function, the gradient of this loss is bounded as $\|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1 \to 0$, which can improve numerical stability during training. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/losses.jl#L15-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.kpowerloss' href='#NeuralEstimators.kpowerloss'><span class="jlbinding">NeuralEstimators.kpowerloss</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
kpowerloss(θ̂, θ, κ; agg = mean, safeorigin = true, ϵ = 0.1)
```


For `κ` &gt; 0, the `κ`-th power absolute-distance loss function,

$$L(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1^\kappa,$$

contains the squared-error (`κ` = 2), absolute-error (`κ` = 2), and 0–1 (`κ` → 0) loss functions as special cases. It is Lipschitz continuous if `κ` = 1, convex if `κ` ≥ 1, and strictly convex if `κ` &gt; 1. It is quasiconvex for all `κ` &gt; 0.

If `safeorigin = true`, the loss function is modified to be piecewise, continuous, and linear in the `ϵ`-interval surrounding the origin, to avoid pathologies around the origin. 

See also [`tanhloss()`](/API/lossfunctions#NeuralEstimators.tanhloss).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/losses.jl#L50-L63" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.quantileloss' href='#NeuralEstimators.quantileloss'><span class="jlbinding">NeuralEstimators.quantileloss</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
quantileloss(θ̂, θ, τ; agg = mean)
quantileloss(θ̂, θ, τ::Vector; agg = mean)
```


The asymmetric quantile loss function,

$$  L(θ̂, θ; τ) = (θ̂ - θ)(𝕀(θ̂ - θ > 0) - τ),$$

where `τ` ∈ (0, 1) is a probability level and 𝕀(⋅) is the indicator function.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/losses.jl#L95-L104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.intervalscore' href='#NeuralEstimators.intervalscore'><span class="jlbinding">NeuralEstimators.intervalscore</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
intervalscore(l, u, θ, α; agg = mean)
intervalscore(θ̂, θ, α; agg = mean)
intervalscore(assessment::Assessment; average_over_parameters::Bool = false, average_over_sample_sizes::Bool = true)
```


Given an interval [`l`, `u`] with nominal coverage 100×(1-`α`)%  and true value `θ`, the interval score ([Gneiting and Raftery, 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)) is defined as

$$S(l, u, θ; α) = (u - l) + 2α⁻¹(l - θ)𝕀(θ < l) + 2α⁻¹(θ - u)𝕀(θ > u),$$

where `α` ∈ (0, 1) and 𝕀(⋅) is the indicator function.

The method that takes a single value `θ̂` assumes that `θ̂` is a matrix with $2d$ rows, where $d$ is the dimension of the parameter vector to make inference on. The first and second sets of $d$ rows will be used as `l` and `u`, respectively.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/losses.jl#L158-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

