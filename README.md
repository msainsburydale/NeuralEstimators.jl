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

1. Install `Julia` (see [here](https://julialang.org/downloads/)).
1. Install `NeuralEstimators.jl`: 
	- To install the current stable version of the package, run the command `using Pkg; Pkg.add("NeuralEstimators")` inside `Julia`. 
	- Alternatively, one may install the development version with the command `using Pkg; Pkg.add(url="https://github.com/msainsburydale/NeuralEstimators.jl")`.


### R interface

A convenient interface for `R` users is available [here](https://github.com/msainsburydale/NeuralEstimators) and on [CRAN](https://cran.r-project.org/web/packages/NeuralEstimators/index.html).

### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use it in your research or other activities, please also use the following citation.

```
@article{,
		author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
		title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
		journal = {The American Statistician},
		year = {2024},
		volume = {78},
		pages = {1--14},
		doi = {10.1080/00031305.2023.2249522},
		url = {https://doi.org/10.1080/00031305.2023.2249522}
}
```

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Modern extreme value statistics for Utopian extremes** [[paper]](https://arxiv.org/abs/2311.11054)

- **Neural Methods for Amortised Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)


### Related packages

Several other software packages have been developed to facilitate neural likelihood-free inference. These include:

- [BayesFlow](https://github.com/stefanradev93/BayesFlow) (TensorFlow)
- [LAMPE](https://github.com/probabilists/lampe) (PyTorch)
- [sbi](https://github.com/sbi-dev/sbi) (PyTorch)
- [swyft](https://github.com/undark-lab/swyft) (PyTorch)


A summary of the functionality in these packages is given in [Zammit-Mangion et al. (2024, Section 6.1)](https://arxiv.org/abs/2404.12484). Note that this list of related packages was created in July 2024; if you have software to add to this list, please contact the package maintainer.
