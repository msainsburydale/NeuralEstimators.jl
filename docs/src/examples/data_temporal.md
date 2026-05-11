# Time-series data

Here, we develop a neural estimator to infer parameters from a $p$-dimensional VAR(1) process,

$$\boldsymbol{Z}_t = \boldsymbol{A}\boldsymbol{Z}_{t-1} + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \overset{\mathrm{iid}}{\sim} N(\boldsymbol{0}, \boldsymbol{\Sigma}),$$

observed over $T$ time steps. For simplicity we take $p = 2$ and target the transition matrix entries and innovation standard deviations $\boldsymbol{\theta} \equiv (A_{11}, A_{12}, A_{21}, A_{22}, \sigma_1, \sigma_2)'$, with $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2, \sigma_2^2)$. We adopt independent $N(0, 0.5^2)$ priors on each entry of $\boldsymbol{A}$, truncated to the stationary region $\rho(\boldsymbol{A}) < 1$, and independent $\mathrm{Half\text{-}Normal}(0, 0.5^2)$ priors on $\sigma_1$ and $\sigma_2$.

## Package dependencies

```julia
using NeuralEstimators
using CairoMakie
using LinearAlgebra
using MatrixEquations: lyapd
using MLUtils: flatten
```

To improve computational efficiency, various GPU backends are supported. Once the relevant package is loaded and a compatible GPU is available, it will be used automatically:

::: code-group

```julia [NVIDIA GPUs]
using CUDA, cuDNN
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

Select a deep-learning backend:

::: code-group

```julia [Flux]
using Flux
```

```julia [Lux]
using Lux

# NB: for the most computationally efficient setup, pass device = reactant_device() to train()
using Reactant
Reactant.set_default_backend("gpu")
```

:::

## Sampling parameters

We first define a function to sample parameters from the prior distribution. Each column of the returned `NamedMatrix` holds a single draw: the four entries of $\boldsymbol{A}$ and the two innovation standard deviations stacked as $(A_{11}, A_{12}, A_{21}, A_{22}, \sigma_1, \sigma_2)'$. We reject draws whose spectral radius $\rho(\boldsymbol{A}) \geq 1$ to ensure stationarity:

```julia
function sampler(K)
    p = 2   # dimension of the VAR process
    A = Matrix{Float64}(undef, p^2, 0)
    while size(A, 2) < K
        candidates = 0.5 .* randn(p^2, 2K)   # oversample then filter
        valid = [ρ(reshape(candidates[:, k], p, p)) < 1 for k in 1:2K]
        A = hcat(A, candidates[:, valid])
    end
    NamedMatrix(
        A₁₁ = A[1, 1:K],
        A₁₂ = A[2, 1:K],
        A₂₁ = A[3, 1:K],
        A₂₂ = A[4, 1:K],
        σ₁  = abs.(0.5 .* randn(K)),
        σ₂  = abs.(0.5 .* randn(K)),
    )
end

ρ(A) = maximum(abs.(eigvals(A)))   # spectral radius
```

## Simulating data

Next, we define the statistical model through simulation. Each data set is a single multivariate time series of length $T$, stored as a $p \times T$ matrix, while a batch of $B$ data sets is stored as a $p \times T \times B$ array.

```julia
function simulator(θ::AbstractVector, T::Integer; stationary_init::Bool = true)
    p = 2   # dimension of the VAR process
    A = reshape([θ["A₁₁"], θ["A₁₂"], θ["A₂₁"], θ["A₂₂"]], p, p)
    σ = [θ["σ₁"], θ["σ₂"]]
    Z = Matrix{Float64}(undef, p, T)
    Z[:, 1] = if stationary_init
        Σ = lyapd(A, Diagonal(σ.^2)) # solve Σ = AΣAᵀ + diag(σ²)
        L = cholesky(Σ).L
        L * randn(p)
    else
        σ .* randn(p)
    end
    for t in 2:T
        Z[:, t] = A * Z[:, t-1] + σ .* randn(p)
    end
    return Z
end
simulator(θ::AbstractMatrix, T::Integer) = stack([simulator(ϑ, T) for ϑ in eachcol(θ)])
```

## Constructing the neural network

Because the data are a time series, the neural network should capture temporal dependencies. Two natural choices are a **1D CNN** (which extracts local patterns via convolution over time) and an **LSTM** (which maintains a hidden state across the full sequence).

A common heuristic is to set the number of summary statistics $d^*$ to a multiple of $d$, the number of unknown parameters.

```julia
p = 2    # dimension of each observation
d = 6    # number of parameters (entries of A and innovation standard deviations)
num_summaries = 3d
```

::: code-group

```julia [CNN]
network = Chain(
    # 1D CNN: convolve over the time dimension
    x -> permutedims(x, (2, 1, 3)),    # CNN expects time-step along first dimension
    Conv((3,), p => 32, gelu; pad = 1),
    Conv((3,), 32 => 64, gelu; pad = 1),
    GlobalMeanPool(),
    flatten,
    Dense(64, 64, gelu),
    Dense(64, num_summaries, gelu)
)
```

```julia [LSTM]
network = Chain(
    # LSTM: maps observation at each step to a hidden state
    LSTM(p => 64),
    x -> x[:, end, :],                # take the final hidden state
    Dense(64, 64, gelu),
    Dense(64, num_summaries, gelu)
)
```

:::

## Constructing the neural estimator

We now construct a [`NeuralEstimator`](@ref "Estimators") by wrapping the neural network in the subtype corresponding to the intended inferential method:

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

Next, we train the estimator using [`train`](@ref):

```julia
K = 10000 # number of parameter vectors in the training/validation sets
T = 100   # number of time steps in each data set
```

::: code-group

```julia [Simulation on-the-fly]
estimator = train(estimator, sampler, simulator; K = 10000, simulator_args = T)
```

```julia [Fixed parameters]
θ_train   = sampler(K)
θ_val     = sampler(K)
estimator = train(estimator, θ_train, θ_val, simulator; simulator_args = T)
```

```julia [Fixed parameters and data]
θ_train   = sampler(K)
θ_val     = sampler(K)
Z_train   = simulator(θ_train, T)
Z_val     = simulator(θ_val, T)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val)
```

:::

## Assessing the estimator

The function [`assess`](@ref) can then be used to assess the trained estimator based on unseen test data simulated from the statistical model:

```julia
θ_test = sampler(1000)
Z_test = simulator(θ_test, T)
assessment = assess(estimator, θ_test, Z_test)
```

The resulting [`Assessment`](@ref) object contains ground-truth parameters, estimates, and other quantities that can be used to compute quantitative and qualitative diagnostics:

```julia
bias(assessment)
rmse(assessment)
plot(assessment)
```

## Applying the estimator to observed data

Once an estimator is deemed to be well calibrated, it may be applied to observed data (below, we use simulated data as a stand-in for observed data):

```julia
θ = sampler(1)                      # ground truth (not known in practice)
Z = simulator(θ, 100)               # stand-in for a real VAR(1) time series
```

::: code-group

```julia [Point estimator]
estimate(estimator, Z)             # point estimate of (A₁₁, A₁₂, A₂₁, A₂₂, σ₁, σ₂)
```

```julia [Posterior estimator]
sampleposterior(estimator, Z)      # posterior sample
```

```julia [Ratio estimator]
sampleposterior(estimator, Z)      # posterior sample
```

:::
