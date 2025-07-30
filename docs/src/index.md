# NeuralEstimators

`NeuralEstimators` facilitates a suite of neural methods for parameter inference in scenarios where simulation from the model is feasible. These methods are **likelihood-free** and **amortised**, in the sense that, once the neural networks are trained on simulated data, they enable rapid inference across arbitrarily many observed data sets in a fraction of the time required by conventional approaches. 

The package supports neural Bayes estimators, which transform data into point summaries of the posterior distribution; neural posterior estimators, which perform approximate posterior inference via KL-divergence minimisation; and neural ratio estimators, which approximate the likelihood-to-evidence ratio and thereby enable frequentist or Bayesian inference through various downstream algorithms, such as MCMC sampling. 

User-friendliness is a central focus of the package, which is designed to minimise "boilerplate" code while preserving complete flexibility in the neural-network architecture and other workflow components. The package accommodates any model for which simulation is feasible by allowing users to define their model implicitly through simulated data. A convenient interface for R users is available on [CRAN](https://cran.r-project.org/web/packages/NeuralEstimators/index.html).

Once familiar with the [Methodology](@ref), see the [Overview](@ref) of the package workflow and the [Examples](@ref), or refer to the [Quick start](@ref) section below.

## Installation

To install the package, please first install the current stable release of [`Julia`](https://julialang.org/downloads/). Then, one may install the current stable version of the package using the following command inside `Julia`:

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
d = 2    # dimension of the parameter vector θ
n = 1    # dimension of each data replicate Zᵢ
sample(K) = rand(d, K) 
simulate(θ, m = 100) = [ϑ[1] .+ ϑ[2] * randn(n, m) for ϑ ∈ eachcol(θ)]  

# Neural network, based on the DeepSets architecture
w = 128  # width of each hidden layer 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

# Initialise a neural point estimator
estimator = PointEstimator(network) 

# Train the estimator
estimator = train(estimator, sample, simulate, epochs = 20)

# Assess the estimator
θ_test = sample(1000)
Z_test = simulate(θ_test)
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)   # μ = 0.001, σ = 0.001
rmse(assessment)   # μ = 0.05,  σ = 0.04

# Apply the estimator to observed data
θ = [0.8 0.1]'          # true parameters
Z = simulate(θ)         # "observed" data
estimate(estimator, Z)  # point estimate: μ̂ = 0.797, σ̂ = 0.087
```

## Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the [repository](https://github.com/msainsburydale/NeuralEstimators.jl). If you use it in your research or other activities, please also use the following citations.

```
@Manual{,
    title = {{NeuralEstimators}: Likelihood-Free Parameter Estimation
      using Neural Networks},
    author = {Matthew Sainsbury-Dale},
    year = {2024},
    note = {R package version 0.1-2},
    url = {https://CRAN.R-project.org/package=NeuralEstimators},
    doi = {10.32614/CRAN.package.NeuralEstimators},
  }

@Article{,
    title = {Likelihood-Free Parameter Estimation with Neural {B}ayes
      Estimators},
    author = {Matthew Sainsbury-Dale and Andrew Zammit-Mangion and
      Raphael Huser},
    journal = {The American Statistician},
    year = {2024},
    volume = {78},
    pages = {1--14},
    doi = {10.1080/00031305.2023.2249522},
  }
```

## Contributing

If you encounter a bug or have a suggestion, please consider [opening an issue](https://github.com/msainsburydale/NeuralEstimators.jl/issues) or submitting a pull request. Instructions for contributing to the documentation can be found in [docs/README.md](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/README.md). When adding functionality to the package, you may wish to add unit tests to the file [test/runtests.jl](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/test/runtests.jl). You can then run these tests locally by executing the following command from the root folder:
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://doi.org/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural methods for amortized inference** [[paper]](https://doi.org/10.1146/annurev-statistics-112723-034123)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://doi.org/10.1080/10618600.2024.2433671)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://jmlr.org/papers/v25/23-1134.html) [[code]](https://github.com/Jbrich95/CensoredNeuralEstimators)

- **Neural parameter estimation with incomplete data** [[paper]](https://arxiv.org/abs/2501.04330)[[code]](https://github.com/msainsburydale/NeuralIncompleteData)

- **Neural Bayes inference for complex bivariate extremal dependence models** [[paper]](https://arxiv.org/abs/2503.23156)[[code]]( https://github.com/lidiamandre/NBE_classifier_depmodels)

- **Fast likelihood-free parameter estimation for Lévy processes** [[paper]](https://www.arxiv.org/abs/2505.01639)