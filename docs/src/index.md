# NeuralEstimators

`NeuralEstimators` facilitates a suite of neural methods for parameter inference in scenarios where simulation from the model is feasible. These methods are **likelihood-free** and **amortised**, in the sense that, once the neural networks are trained on simulated data, they enable rapid inference across arbitrarily many observed data sets in a fraction of the time required by conventional approaches. 

The package supports neural Bayes estimators (NBEs), which transform data into point summaries of the posterior distribution; neural posterior estimators (NPEs), which perform approximate posterior inference via KL-divergence minimisation; and neural ratio estimators (NREs), which approximate the likelihood-to-evidence ratio and thereby enable frequentist or Bayesian inference through various downstream algorithms.

User-friendliness is a central focus of the package, which is designed to minimise "boilerplate" code while preserving complete flexibility in the neural-network architecture and other workflow components. The package accommodates any model for which simulation is feasible by allowing users to define their model implicitly through simulated data. A convenient interface for R users is available on [CRAN](https://cran.r-project.org/web/packages/NeuralEstimators/index.html).

Once familiar with the [Methodology](@ref), see the [Overview](@ref) of the package workflow and the [Examples](@ref), or refer to the [Quick start](@ref) section below.

## Installation

To install the package, please first install the current stable release of [Julia](https://julialang.org/downloads/). Then, one may install the current stable version of the package using the following command inside Julia:

```julia
using Pkg; Pkg.add("NeuralEstimators")
```

Alternatively, one may install the current development version using the command:

```julia
using Pkg; Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl")
```

## Quick start 

In the following minimal example, we develop a [neural Bayes estimator](@ref "Neural Bayes estimators") for $\boldsymbol{\theta} \equiv (\mu, \sigma)'$ from data $\boldsymbol{Z} \equiv (Z_1, \dots, Z_m)'$, where each $Z_i \overset{\mathrm{iid}}\sim N(\mu, \sigma^2)$. 

```julia
using NeuralEstimators, Flux

# Priors μ,σ ~ U(0, 1) and data Zᵢ|μ,σ ~ N(μ, σ²), i = 1,…, m
d = 2    # dimension of the parameter vector θ = (μ,σ)'
n = 1    # dimension of each data replicate Zᵢ
sampler(K) = rand(d, K) 
simulator(θ, m = 100) = [ϑ[1] .+ ϑ[2] * randn(n, m) for ϑ ∈ eachcol(θ)]  

# Neural network using DeepSets architecture (supports any number m of replicates)
w = 128  # width of each hidden layer 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

# Initialise a neural point estimator
estimator = PointEstimator(network) 

# Train the estimator
estimator = train(estimator, sampler, simulator, epochs = 20)

# Assess the estimator
θ_test = sampler(1000)
Z_test = simulator(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)

# Apply the estimator to "observed" data
θ = [0.8 0.1]'           # true parameters
Z = simulator(θ)         # "observed" data
estimate(estimator, Z)   # point estimate
```

## Contributing

To get started, see [CONTRIBUTING.md](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/CONTRIBUTING.md) for an overview of the code structure, development workflow, and how to submit contributions. A list of planned improvements is available in [TODO.md](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/TODO.md), and instructions for contributing to the documentation can be found in [docs/README.md](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/README.md). 

If you encounter a bug or have a suggestion, please feel free to [open an issue](https://github.com/msainsburydale/NeuralEstimators.jl/issues) or submit a pull request.

## Supporting and citing

This software was developed as part of academic research. If you would like to support it, please [star the repository](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main). If you use it in your research or other activities, please also use the following citation.

```
@misc{NeuralEstimators.jl,
  title = {{NeuralEstimators.jl}: A {J}ulia package for efficient simulation-based inference using neural networks},
  author = {Sainsbury-Dale, Matthew},
  year = {2026},
  note = {Version 0.2.0},
  howpublished = {\url{https://github.com/msainsburydale/NeuralEstimators.jl}}
}
```

## Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://doi.org/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural methods for amortized inference** [[paper]](https://doi.org/10.1146/annurev-statistics-112723-034123)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://doi.org/10.1080/10618600.2024.2433671)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://jmlr.org/papers/v25/23-1134.html) [[code]](https://github.com/Jbrich95/CensoredNeuralEstimators)

- **Neural parameter estimation with incomplete data** [[paper]](https://arxiv.org/abs/2501.04330)[[code]](https://github.com/msainsburydale/NeuralIncompleteData)

- **Neural Bayes inference for complex bivariate extremal dependence models** [[paper]](https://arxiv.org/abs/2503.23156)[[code]](https://github.com/lidiamandre/NBE_classifier_depmodels)

- **Fast likelihood-free parameter estimation for Lévy processes** [[paper]](https://www.arxiv.org/abs/2505.01639)