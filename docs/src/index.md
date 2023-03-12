*Note to reviewers.* 

This is an anonymous version of the online documentation. Since it is being shared locally, links in the left sidebar do not work exactly as intended: navigating with this sidebar will display the local folder structure for the corresponding page, which requires one to click `index.html` to proceed to the documentation.


# NeuralEstimators

Neural estimators are neural networks that transform data into parameter point estimates, and they are a promising recent approach to inference. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators.  Uncertainty quantification with neural estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator.

The package `NeuralEstimators` facilitates the development of neural estimators in a user-friendly manner. It caters for arbitrary models by having the user implicitly define their model via simulated data. This makes the development of neural estimators particularly straightforward for models with existing implementations (possibly in other programming languages, e.g., `R` or `python`). A convenient interface for `R` users is available *link to be inserted here*.


### Getting started
Install `NeuralEstimators` from [Julia](https://julialang.org/)'s package manager using the following command inside Julia:

```
using Pkg; Pkg.add("NeuralEstimators")
```

Once familiar with the details of the [Theoretical framework](@ref), see the [Examples](@ref).


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
omitted for anonymity
```
