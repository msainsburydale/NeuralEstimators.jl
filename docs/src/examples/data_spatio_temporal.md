# Spatio-Temporal Data

Here, we develop a neural estimator to infer 
$\boldsymbol{\theta} \equiv (\theta_1, \theta_2)'$ from data 
modelled as a **Spatio-Temporal Gaussian Random Field (GRF)**. 
This model is appropriate for environmental phenomena such as 
Sea Surface Temperature (SST) that exhibit correlation across 
both space and time.

The two parameters are:

- $\theta_1$ **(Spatial Range):** controls spatial correlation 
  via the Matérn covariance function ($\nu = 1.5$).
- $\theta_2$ **(Temporal Range):** controls temporal persistence 
  via a first-order autoregressive [AR(1)] structure.

We adopt the priors $\theta_1 \sim \text{Uniform}(0.05, 0.5)$ 
and $\theta_2 \sim \text{Uniform}(0.1, 0.9)$.

## Package dependencies
```julia
using NeuralEstimators
using Flux
using Distances
using LinearAlgebra
using Statistics
using CairoMakie
```

To improve computational efficiency, various GPU backends are supported. Once the relevant package is loaded and a compatible GPU is available, it will be used automatically:

::: code-group
```julia [NVIDIA GPUs]
using CUDA
```
```julia [AMD ROCm GPUs]
using AMDGPU
```
```julia [Metal M-Series GPUs]
using Metal
```
```julia [Intel GPUs]
using oneAPI
```

:::

## Sampling parameters

We first define a function to sample parameters from the prior distribution. Here, we store the parameters as a `NamedMatrix` so that parameter estimates are automatically labelled, though this is not required:
```julia
function sampler(K)
    NamedMatrix(
        θ₁ = rand(K) .* (0.5 - 0.05) .+ 0.05,
        θ₂ = rand(K) .* (0.9 - 0.1)  .+ 0.1
    )
end
```

## Simulating data

Next, we define the statistical model implicitly through simulation. The data are observed on a regular spatial grid over $T$ time steps. The spatio-temporal covariance is modelled as the product of a **Matérn spatial covariance** and an **AR(1) temporal covariance**:

$$\text{Cov}(Z(\mathbf{s}, t),\, Z(\mathbf{s}', t')) = C_{\text{Matérn}}(\|\mathbf{s} - \mathbf{s}'\|;\, \theta_1) \cdot \theta_2^{|t - t'|}$$
```julia
# Regular spatial grid
grid_size = 10
locs = [(i/grid_size, j/grid_size) for i in 1:grid_size for j in 1:grid_size]
n_spatial = length(locs)

# Matérn covariance (ν = 1.5)
function matern15(d, ρ)
    d == 0.0 && return 1.0
    r = sqrt(3) * d / ρ
    return (1 + r) * exp(-r)
end

# Simulate a single spatio-temporal field
function simulator(θ::AbstractVector, T::Integer)
    ρ = θ["θ₁"]
    φ = θ["θ₂"]
    Σ = [matern15(norm(collect(locs[i]) .- collect(locs[j])), ρ)
         for i in 1:n_spatial, j in 1:n_spatial]
    L = cholesky(Σ + 1e-6 * I).L
    Z = zeros(n_spatial, T)
    Z[:, 1] = L * randn(n_spatial)
    for t in 2:T
        Z[:, t] = φ .* Z[:, t-1] .+ sqrt(1 - φ^2) .* (L * randn(n_spatial))
    end
    return reshape(Z, grid_size, grid_size, T)
end

simulator(θ::AbstractVector, T)         = simulator(θ, rand(T))
simulator(θ::AbstractMatrix, T = 10:30) = [simulator(ϑ, T) for ϑ in eachcol(θ)]
```

## Constructing the neural network

The data are observed on a **regular spatial grid** over $T$ time steps, yielding arrays of shape $(n_x, n_y, T)$. We therefore adopt a **Convolutional Neural Network (CNN)** as the inner network of a `DeepSet`, which is well-suited to extracting spatial features from gridded data.

The CNN processes each spatial slice independently via shared weights, and the resulting feature vectors are aggregated across time using a mean-pooling operation before being passed to a fully connected outer network.
```julia
d             = 2       # number of parameters
num_summaries = 3d      # number of summary statistics
w             = 64      # hidden layer width

ψ = Chain(
    x -> reshape(x, grid_size, grid_size, 1, :),
    Conv((3, 3), 1  => 16, relu; pad = 1),
    Conv((3, 3), 16 => 32, relu; pad = 1),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(32 * 5 * 5, w, relu)
)

ϕ = Chain(Dense(w, w, relu), Dense(w, num_summaries))

network = DeepSet(ψ, ϕ)
```

## Constructing the neural estimator

We now construct a `NeuralEstimator` by wrapping the neural network in the subtype corresponding to the intended inferential method:

::: code-group
```julia [Point estimator]
estimator = PointEstimator(network, d; num_summaries = num_summaries)
```
```julia [Posterior estimator]
estimator = PosteriorEstimator(network, d; num_summaries = num_summaries)
```
```julia [Ratio estimator]
estimator = RatioEstimator(network, d; num_summaries = num_summaries)
```

:::

## Training the estimator

Next, we train the estimator using `train`. The training loop automatically simulates spatio-temporal fields on-the-fly, enabling **simulation-based inference (SBI)** without the need to evaluate the likelihood function:
```julia
estimator = train(estimator, sampler, simulator)
```

The empirical risk (average loss) over the training and validation sets can be plotted using .`plotrisk`

One may wish to save a trained estimator and load it in a later session: see [Saving and loading neural estimators](@ref) for details on how this can be done.


## Assessing the estimator

The function `assess` can be used to assess the trained estimator:
```julia
θ_test = sampler(1000)
Z_test = simulator(θ_test, 20)
assessment = assess(estimator, θ_test, Z_test)
```

The resulting `Assessment` object contains ground-truth parameters, estimates, and other quantities that can be used to compute quantitative and qualitative diagnostics:
```julia
bias(assessment)    # θ₁ = ..., θ₂ = ...
rmse(assessment)    # θ₁ = ..., θ₂ = ...
risk(assessment)    # overall risk
plot(assessment)
```

## Applying the estimator to observed data

Once an estimator is deemed to be well calibrated, it may be applied to observed data (below, we use simulated data as a stand-in for observed data):
```julia
θ = sampler(1)
Z = simulator(θ, 20)
```

::: code-group
```julia [Point estimator]
estimate(estimator, Z)
interval(bootstrap(estimator, Z))
```
```julia [Posterior estimator]
sampleposterior(estimator, Z)
```
```julia [Ratio estimator]
sampleposterior(estimator, Z)
```

:::