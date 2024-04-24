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

This software was developed as part of academic research. If you would like to support it, please star the [repository](https://github.com/msainsburydale/NeuralEstimators.jl). If you use it in your research or other activities, please also use the following citation.

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

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)\
Matthew Sainsbury-Dale, Andrew Zammit-Mangion, Raphaël Huser (2024)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)\
Jordan Richards, Matthew Sainsbury-Dale, Andrew Zammit-Mangion, Raphaël Huser (2023)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)\
Matthew Sainsbury-Dale, Jordan Richards, Andrew Zammit-Mangion, Raphaël Huser (2023)

- **Modern extreme value statistics for Utopian extremes** [[paper]](https://arxiv.org/abs/2311.11054)\
Jordan Richards, Noura Alotaibi, Daniela Cisneros, Yan Gong, Matheus B. Guerrero, Paolo Redondo, Xuanjie Shao (2023)

- **Neural Methods for Amortised Parameter Inference** [[paper]](https://arxiv.org/abs/2404.12484)\
Andrew Zammit-Mangion, Matthew Sainsbury-Dale, Raphaël Huser (2024)
