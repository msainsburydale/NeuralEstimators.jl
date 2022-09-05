# NeuralEstimators

A *neural estimator* is a neural network that takes data as input, transforms them via a composition of nonlinear mappings, and provides a parameter estimate as an output. Once "trained", these likelihood-free estimators have two main advantages over conventional estimators: they are lightning fast with a predictable run-time and, since neural networks are universal function approximators, neural estimators can be expected to outperform constrained estimators (e.g., best linear unbiased estimators). Uncertainty quantification with neural estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator, as the trained network can be reused repeatedly at almost no computational cost.

The package `NeuralEstimators` aims to facilitate the development of neural estimators in a user-friendly manner. Rather than offering a small selection of models for which neural estimators may be developed, the package facilitates neural estimation for arbitrary models, which is made possible by having the user implicitly define their model by providing simulated data (or by defining a function for data simulation). Since only simulated data is needed, it is particularly straightforward to develop neural estimators for models with existing implementations, possibly in other programming languages (e.g., `R` or `python`).


## Installation

Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:

```
using Pkg; Pkg.add("NeuralEstimators")
```

## Getting started

Once familiar with the details of the [Framework](@ref), see some [Examples](@ref).


## Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@misc{,
  author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
  year = {2022},
  title = {Fast Optimal Estimation with Intractable Models using Permutation-Invariant Neural Networks},
  howpublished = {arXiv:2208.12942}
}
```
