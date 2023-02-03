# NeuralEstimators

Neural estimators are neural networks that transform data into parameter estimates, and they are a promising recent approach to inference. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators.  Uncertainty quantification with neural estimators is also straightforward through the bootstrap distribution, which is usually computationally intensive to sample from but which is essentially available "for free" with a neural estimator.

The package `NeuralEstimators` facilitates the development of neural estimators in a user-friendly manner. The package facilitates neural estimation for arbitrary statistical models, which is made possible by having the user implicitly define their model by providing simulated data (or by defining a function for data simulation). Since only simulated data is needed, it is particularly straightforward to develop neural estimators for models with existing implementations, possibly in other programming languages (e.g., `R` or `python`).


### Getting started
Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:

```
using Pkg; Pkg.add("NeuralEstimators")
```

Once familiar with the details of the [Theoretical framework](@ref), see the [Examples](@ref).


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@misc{,
  author = {Sainsbury-Dale, Matthew and Zammit-Mangion, Andrew and Huser, Raphaël},
  year = {2022},
  title = {Fast Optimal Estimation with Intractable Models using Permutation-Invariant Neural Networks},
  howpublished = {arXiv:2208.12942}
}
```
