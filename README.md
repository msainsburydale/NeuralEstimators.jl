# NeuralEstimators <img align="right" width="200" src="https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true">

<!-- ![NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true) -->

[![][docs-dev-img]][docs-dev-url]
[![CI](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://app.codecov.io/gh/msainsburydale/NeuralEstimators.jl)
<!-- [![][R-repo-img]][R-repo-url] -->

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

[R-repo-img]: https://img.shields.io/badge/R-interface-blue.svg
[R-repo-url]: https://github.com/msainsburydale/NeuralEstimators

`NeuralEstimators` facilitates the user-friendly development of neural point estimators, which are neural networks that map data to a point summary of the posterior distribution. These estimators are likelihood-free and amortised, in the sense that, after an initial setup cost, inference from observed data can be made in a fraction of the time required by conventional approaches. It also facilitates the construction of neural networks that approximate the likelihood-to-evidence ratio in an amortised fashion, which allows for making inference based on the likelihood function or the entire posterior distribution. The package caters for any model for which simulation is feasible by allowing the user to implicitly define their model via simulated data. See the [documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) to get started!


### Installation 

To install the package, please:

1. Install [`Julia`](https://julialang.org/downloads/).
1. Install `NeuralEstimators.jl`: 
	- To install the current stable version of the package, run the command `using Pkg; Pkg.add("NeuralEstimators")` inside `Julia`. 
	- Alternatively, one may install the development version with the command `using Pkg; Pkg.add(url="https://github.com/msainsburydale/NeuralEstimators.jl")`.


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

If you find a bug or have a suggestion, please [open an issue](https://github.com/msainsburydale/NeuralEstimators.jl/issues).

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://doi.org/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://doi.org/10.1080/10618600.2024.2433671)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Neural Methods for Amortized Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)

- **Neural parameter estimation with incomplete data** [[paper]](https://arxiv.org/abs/2501.04330)[[code]](https://github.com/msainsburydale/NeuralEM)