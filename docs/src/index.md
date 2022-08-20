# NeuralEstimators

A *neural estimator* is a neural network that takes data as input, transforms them via a composition of nonlinear mappings, and provides a parameter estimate as an output. Neural estimators, have recently emerged as a promising alternative to traditional likelihood-free approaches, such approximate Bayesian computation (ABC). Neural estimators, once "trained", have two main advantages over conventional estimators:  First, neural estimators are fast with a predictable run-time and, . Second, neural estimators are universal function approximators, and they can therefore be expected to outperform constrained estimators such as best linear unbiased estimators. Uncertainty quantification with neural estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator, as the trained network can be reused repeatedly at almost no computational cost.

The package `NeuralEstimators` aims to facilitate the development of neural estimators in a user-friendly manner. The package is able to cater for arbitrary statistical models by relying on the user to define the statistical model implicitly, either by providing data simulated from the model or by defining a function for data simulation.


## Installation

Download [Julia](https://julialang.org/), then install `NeuralEstimators` from Julia's package manager using the following commands inside the Julia REPL:

```
using Pkg
Pkg.add("NeuralEstimators")
```

## Getting started

See our [Workflow overview](@ref).


## Supporting and citing

This software was developed as part of academic research. If you would like to help support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@article{
  <bibtex citation>
}
```
