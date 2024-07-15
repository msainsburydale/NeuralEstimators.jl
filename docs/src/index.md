# NeuralEstimators

Neural Bayes estimators are neural networks that transform data into point summaries of the posterior distribution. They are likelihood free and, once constructed, substantially faster than classical methods. Uncertainty quantification with neural Bayes estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator, or by training a neural Bayes estimator to approximate a set of marginal posterior quantiles. A related class of methods uses neural networks to approximate the likelihood function, the likelihood-to-evidence ratio, and the full posterior distribution. 

The package `NeuralEstimators` facilitates the development of neural Bayes estimators and related neural inferential methods in a user-friendly manner. It caters for arbitrary models by having the user implicitly define their model via simulated data. This makes development particularly straightforward for models with existing implementations (possibly in other programming languages, e.g., `R` or `python`). A convenient interface for `R` users is available [here](https://github.com/msainsburydale/NeuralEstimators).


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

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Modern extreme value statistics for Utopian extremes** [[paper]](https://arxiv.org/abs/2311.11054)

- **Neural Methods for Amortised Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)


### Related packages 

Several other software packages have been developed to facilitate neural likelihood-free inference. These include:

1. [BayesFlow](https://github.com/stefanradev93/BayesFlow) (TensorFlow)
1. [LAMPE](https://github.com/probabilists/lampe) (PyTorch)
1. [sbi](https://github.com/sbi-dev/sbi) (PyTorch)
1. [swyft](https://github.com/undark-lab/swyft) (PyTorch)


A summary of the functionality in these packages is given in [Zammit-Mangion et al. (2024, Section 6.1)](https://arxiv.org/abs/2404.12484). Note that this list of related packages was created in July 2024; if you have software to add to this list, please contact the package maintainer. 