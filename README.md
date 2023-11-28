# NeuralEstimators <img align="right" width="200" src="https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true">

<!-- ![NeuralEstimators](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/src/assets/logo.png?raw=true) -->

[![][docs-dev-img]][docs-dev-url]
[![CI](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

`NeuralEstimators` facilitates the user-friendly development of neural point estimators, which are neural networks that transform data into parameter point estimates. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators. The package caters for any model for which simulation is feasible. See the [documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) to get started!

### R interface

A convenient interface for `R` users is available [here](https://github.com/msainsburydale/NeuralEstimators).


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use it in your research or other activities, please use the following citation.

```
@article{SZH_2023_neural_Bayes_estimators,
	author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
	title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
	journal = {The American Statistician},
	year = {2023},
	volume = {to appear},
	doi = {10.1080/00031305.2023.2249522},
	url = {https://doi.org/10.1080/00031305.2023.2249522}
}
```

### Papers using NeuralEstimators

- **Likelihood-Free Parameter Estimation with Neural Bayes Estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)\
Matthew Sainsbury-Dale, Andrew Zammit-Mangion, Raphaël Huser (2023)


- **Neural Bayes Estimators for Censored Inference with Peaks-Over-Threshold Models** [[paper]](https://arxiv.org/abs/2306.15642)\
Jordan Richards, Matthew Sainsbury-Dale, Andrew Zammit-Mangion, Raphaël Huser (2023+)

- **Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks** [[paper]](https://arxiv.org/abs/2310.02600)\
Matthew Sainsbury-Dale, Jordan Richards, Andrew Zammit-Mangion, Raphaël Huser (2023+)
