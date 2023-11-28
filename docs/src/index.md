# NeuralEstimators

Neural estimators are neural networks that transform data into parameter point estimates. They are likelihood free, substantially faster than classical methods, and can be designed to be approximate Bayes estimators. Uncertainty quantification with neural estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator, or by training a neural estimator to approximate a set of marginal posterior quantiles.

The [Julia](https://julialang.org/) package `NeuralEstimators` facilitates the development of neural estimators in a user-friendly manner. It caters for arbitrary models by having the user implicitly define their model via simulated data. This makes the development of neural estimators particularly straightforward for models with existing implementations (possibly in other programming languages, e.g., `R` or `python`). A convenient interface for `R` users is available [here](https://github.com/msainsburydale/NeuralEstimators).


### Getting started
Install `NeuralEstimators` using the following command inside `Julia`:

```
using Pkg; Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl")
```

Once familiar with the details of the [Framework](@ref), see the [Examples](@ref).


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the [repository](https://github.com/msainsburydale/NeuralEstimators.jl). If you use it in your research or other activities, please use the following citation.

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

- **Likelihood-Free Parameter Estimation with Neural Bayes Estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)

- **Neural Bayes Estimators for Censored Inference with Peaks-Over-Threshold Models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks** [[paper]](https://arxiv.org/abs/2310.02600)
