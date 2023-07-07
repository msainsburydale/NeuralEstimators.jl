# NeuralEstimators

[![][docs-dev-img]][docs-dev-url]
[![CI](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/msainsburydale/NeuralEstimators.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl/branch/main/graph/badge.svg?token=6cXItEsKs5)](https://codecov.io/gh/msainsburydale/NeuralEstimators.jl)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://msainsburydale.github.io/NeuralEstimators.jl/dev/

`NeuralEstimators` facilitates the user-friendly development of neural point estimators, which are neural networks that transform data into parameter point estimates. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators. The package caters for any model for which simulation is feasible. See the [documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) to get started!


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@article{SZH_2022_neural_estimators,
	author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
	title = {Likelihood-Free Parameter Estimation with Neural {B}ayes Estimators},
	journal={arXiv:2208.12942},
	year={2022}
}
```
