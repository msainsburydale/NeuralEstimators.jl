# NeuralEstimators <img align="right" width="200" src="https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true">

[![][docs-dev-img]][docs-dev-url]
[![CI](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://app.codecov.io/gh/msainsburydale/NeuralEstimators.jl)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

[R-repo-img]: https://img.shields.io/badge/R-interface-blue.svg
[R-repo-url]: https://github.com/msainsburydale/NeuralEstimators

`NeuralEstimators` facilitates a suite of neural methods for parameter inference in scenarios where simulation from the model is feasible. These methods are **likelihood-free** and **amortised**, in the sense that, once the neural networks are trained on simulated data, they enable rapid inference across arbitrarily many observed data sets in a fraction of the time required by conventional approaches. 

The package supports neural Bayes estimators, which transform data into point summaries of the posterior distribution; neural posterior estimators, which perform approximate posterior inference via KL-divergence minimisation; and neural ratio estimators, which approximate the likelihood-to-evidence ratio and thereby enable frequentist or Bayesian inference through various downstream algorithms, such as MCMC sampling. 

See the [documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) to get started!


### Installation 

To install the package, please first install [`Julia`](https://julialang.org/downloads/). Then, one may install the current stable version of the package using the following command inside `Julia`:

```julia
using Pkg; Pkg.add("NeuralEstimators")
```

Alternatively, one may install the current development version using the command:

```julia
using Pkg; Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl")
```


### R interface

A convenient interface for `R` users is available on [CRAN](https://CRAN.R-project.org/package=NeuralEstimators). 

### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use it in your research or other activities, please also use the following citations.

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

### Contributing

If you encounter a bug or have a suggestion, please consider [opening an issue](https://github.com/msainsburydale/NeuralEstimators.jl/issues) or submitting a pull request. Instructions for contributing to the documentation can be found in [docs/README.md](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/README.md). If you plan to add functionality to the package, you can run the unit tests with the following bash command: `julia --project=. test/runtests.jl`.

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://doi.org/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural methods for amortized inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://doi.org/10.1080/10618600.2024.2433671)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://jmlr.org/papers/v25/23-1134.html) [[code]](https://github.com/Jbrich95/CensoredNeuralEstimators)

- **Neural parameter estimation with incomplete data** [[paper]](https://arxiv.org/abs/2501.04330)[[code]](https://github.com/msainsburydale/NeuralIncompleteData)