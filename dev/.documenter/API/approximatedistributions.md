


# Approximate distributions {#Approximate-distributions}

When constructing a [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator), one must choose a family of distributions $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$, parameterized by $\boldsymbol{\kappa} \in \mathcal{K}$, used to approximate the posterior distribution. These families of distributions are implemented as subtypes of the abstract supertype [ApproximateDistribution](/API/approximatedistributions#NeuralEstimators.ApproximateDistribution).

## Distributions {#Distributions}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.ApproximateDistribution' href='#NeuralEstimators.ApproximateDistribution'><span class="jlbinding">NeuralEstimators.ApproximateDistribution</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ApproximateDistribution
```


An abstract supertype for approximate posterior distributions used in conjunction with a [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator). 

Subtypes `A <: ApproximateDistribution` must implement the following methods: 
- `logdensity(q::A, θ::AbstractMatrix, t::AbstractMatrix)` 
  - Used during training and therefore must support automatic differentiation.
    
  - `θ` is a `d × K` matrix of parameter vectors.
    
  - `t` is a `dstar × K` matrix of learned summary statistics obtained by applying the neural network in the `PosteriorEstimator` to a collection of `K` data sets. 
    
  - Should return a `1 × K` matrix, where each entry is the log density `log q(θₖ | tₖ)` for the `k`-th data set evaluated at the `k`-th parameter vector `θ[:, k]`.
    
  
- `sampleposterior(q::A, t::AbstractMatrix, N::Integer)`
  - Used during inference and therefore does not need to be differentiable.
    
  - Should return a `Vector` of length `K`, where each element is a `d × N` matrix containing `N` samples from the approximate posterior `q(θ | tₖ)` for the `k`-th data set.
    
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/ApproximateDistributions/ApproximateDistributions.jl#L1-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.GaussianMixture' href='#NeuralEstimators.GaussianMixture'><span class="jlbinding">NeuralEstimators.GaussianMixture</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GaussianMixture <: ApproximateDistribution
GaussianMixture(d::Integer, dstar::Integer; num_components::Integer = 10, kwargs...)
```


A mixture of Gaussian distributions for amortised posterior inference, where `d` is the dimension of the parameter vector. 

The density of the distribution is: 

$$q(\boldsymbol{\theta}; \boldsymbol{\kappa}) = \sum_{j=1}^{J} \pi_j \cdot \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j), $$

where the parameters $\boldsymbol{\kappa}$ comprise the mixture weights $\pi_j \in [0, 1]$ subject to $\sum_{j=1}^{J} \pi_j = 1$, the mean vector $\boldsymbol{\mu}_j$ of each component, and the variance parameters of the diagonal covariance matrix $\boldsymbol{\Sigma}_j$.

When using a `GaussianMixture` as the approximate distribution of a [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator),  the neural network should be a mapping from the sample space to $\mathbb{R}^{d^*}$,  where $d^*$ is an appropriate number of summary statistics for the parameter vector $\boldsymbol{\theta}$. The summary statistics are then mapped to the mixture parameters using a conventional multilayer perceptron ([MLP](/API/architectures#NeuralEstimators.MLP)) with approporiately chosen output activation functions (e.g., [softmax](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#NNlib.softmax) for the mixture weights, [softplus](https://fluxml.ai/Flux.jl/stable/reference/models/activation/#NNlib.softplus) for the variance parameters).

**Keyword arguments**
- `num_components::Integer = 10`: number of components in the mixture. 
  
- `kwargs`: additional keyword arguments passed to [`MLP`](/API/architectures#NeuralEstimators.MLP). 
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/ApproximateDistributions/GaussianMixture.jl#L1-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.NormalisingFlow' href='#NeuralEstimators.NormalisingFlow'><span class="jlbinding">NeuralEstimators.NormalisingFlow</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NormalisingFlow <: ApproximateDistribution
NormalisingFlow(d::Integer, dstar::Integer; num_coupling_layers::Integer = 6, kwargs...)
```


A normalising flow for amortised posterior inference (e.g., [Ardizzone et al., 2019](https://openreview.net/forum?id=rJed6j0cKX); [Radev et al., 2022](https://ieeexplore.ieee.org/document/9298920)), where `d` is the dimension of  the parameter vector and `dstar` is the dimension of the summary statistics for the data. 

Normalising flows are diffeomorphisms (i.e., invertible, differentiable transformations with differentiable inverses) that map a simple base distribution (e.g., standard Gaussian) to a more complex target distribution (e.g., the posterior). They achieve this by applying a sequence of learned transformations, the forms of which are chosen to be invertible and allow for tractable density computation via the change of variables formula. This allows for efficient density evaluation during the training stage, and efficient sampling during the inference stage. For further details, see the reviews by [Kobyzev et al. (2020)](https://ieeexplore.ieee.org/document/9089305) and [Papamakarios (2021)](https://dl.acm.org/doi/abs/10.5555/3546258.3546315).

`NormalisingFlow` uses affine coupling blocks (see [`AffineCouplingBlock`](/API/approximatedistributions#NeuralEstimators.AffineCouplingBlock)), with activation normalisation ([Kingma and Dhariwal, 2018](https://dl.acm.org/doi/10.5555/3327546.3327685)) and permutations used between each block. The base distribution is taken to be a standard multivariate Gaussian distribution. 

When using a `NormalisingFlow` as the approximate distribution of a [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator),  the neural network should be a mapping from the sample space to $\mathbb{R}^{d^*}$,  where $d^*$ is an appropriate number of summary statistics for the given parameter vector (e.g., $d^* = d$). The summary statistics are then mapped to the parameters of the affine coupling blocks using conventional multilayer perceptrons (see [`AffineCouplingBlock`](/API/approximatedistributions#NeuralEstimators.AffineCouplingBlock)).

**Keyword arguments**
- `num_coupling_layers::Integer = 6`: number of coupling layers. 
  
- `kwargs`: additional keyword arguments passed to [`AffineCouplingBlock`](/API/approximatedistributions#NeuralEstimators.AffineCouplingBlock). 
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/ApproximateDistributions/NormalisingFlow.jl#L175-L192" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Methods {#Methods}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.numdistributionalparams' href='#NeuralEstimators.numdistributionalparams'><span class="jlbinding">NeuralEstimators.numdistributionalparams</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
numdistributionalparams(q::ApproximateDistribution)
numdistributionalparams(estimator::PosteriorEstimator)
```


The number of distributional parameters (i.e., the dimension of the space $\mathcal{K}$ of approximate-distribution parameters $\boldsymbol{\kappa}$). 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/ApproximateDistributions/ApproximateDistributions.jl#L17-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Building blocks {#Building-blocks}
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.AffineCouplingBlock' href='#NeuralEstimators.AffineCouplingBlock'><span class="jlbinding">NeuralEstimators.AffineCouplingBlock</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AffineCouplingBlock(κ₁::MLP, κ₂::MLP)
AffineCouplingBlock(d₁::Integer, dstar::Integer, d₂; kwargs...)
```


An affine coupling block used in a [`NormalisingFlow`](/API/approximatedistributions#NeuralEstimators.NormalisingFlow). 

An affine coupling block splits its input $\boldsymbol{\theta}$ into two disjoint components, $\boldsymbol{\theta}_1$ and $\boldsymbol{\theta}_2$, with dimensions $d_1$ and $d_2$, respectively. The block then applies the following transformation: 

$$\begin{aligned}
    \tilde{\boldsymbol{\theta}}_1 &= \boldsymbol{\theta}_1,\\
    \tilde{\boldsymbol{\theta}}_2 &= \boldsymbol{\theta}_2 \odot \exp\{\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z}))\} + \boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\tilde{\boldsymbol{\theta}}_1, \boldsymbol{T}(\boldsymbol{Z})),
\end{aligned}$$

where $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ and $\boldsymbol{\kappa}_{\boldsymbol{\gamma},2}(\cdot)$ are generic, non-invertible multilayer perceptrons (MLPs) that are functions of both the (transformed) first input component $\tilde{\boldsymbol{\theta}}_1$ and the learned $d^*$-dimensional summary statistics $\boldsymbol{T}(\boldsymbol{Z})$ (see [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator)). 

To prevent numerical overflows and stabilise the training of the model, the scaling factors $\boldsymbol{\kappa}_{\boldsymbol{\gamma},1}(\cdot)$ are clamped using the function 

$$f(\boldsymbol{s}) = \frac{2c}{\pi}\tan^{-1}(\frac{\boldsymbol{s}}{c}),$$

where $c = 1.9$ is a fixed clamping threshold. This transformation ensures that the scaling factors do not grow excessively large.

Additional keyword arguments `kwargs` are passed to the [`MLP`](/API/architectures#NeuralEstimators.MLP) constructor when creating `κ₁` and `κ₂`. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/8897312e13e5ff49dc41be0d960c7da307b8575f/src/ApproximateDistributions/NormalisingFlow.jl#L45-L66" target="_blank" rel="noreferrer">source</a></Badge>

</details>

