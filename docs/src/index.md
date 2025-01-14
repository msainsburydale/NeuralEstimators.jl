# NeuralEstimators

Neural Bayes estimators are neural networks that transform data into point summaries of the posterior distribution. They are likelihood free and, once constructed, substantially faster than classical methods. Uncertainty quantification with neural Bayes estimators is also straightforward through the bootstrap distribution, which is essentially available "for free" with a neural estimator, or by training a neural Bayes estimator to approximate a set of marginal posterior quantiles. A related class of methods uses neural networks to approximate the likelihood function, the likelihood-to-evidence ratio, and the full posterior distribution. 

The package `NeuralEstimators` facilitates the development of neural Bayes estimators and related neural inferential methods in a user-friendly manner. It caters for arbitrary models by having the user implicitly define their model via simulated data. This makes development particularly straightforward for models with existing implementations (possibly in other programming languages, e.g., `R` or `python`). A convenient interface for `R` users is available [here](https://github.com/msainsburydale/NeuralEstimators) and on [CRAN](https://cran.r-project.org/web/packages/NeuralEstimators/index.html).


### Getting started

One may install the current stable version of `NeuralEstimators` using the following command inside `Julia`:

```
using Pkg; Pkg.add("NeuralEstimators")
```

Alternatively, one may install the current development version using the command:

```
using Pkg; Pkg.add(url = "https://github.com/msainsburydale/NeuralEstimators.jl")
```

Once familiar with the details of the [Framework](@ref), see the [Examples](@ref).


### Supporting and citing

This software was developed as part of academic research. If you would like to support it, please star the [repository](https://github.com/msainsburydale/NeuralEstimators.jl). If you use it in your research or other activities, please also use the following citations.

```
@Manual{,
    title = {{NeuralEstimators}: Likelihood-Free Parameter Estimation
      using Neural Networks},
    author = {Matthew Sainsbury-Dale},
    year = {2024},
    note = {R package version 0.1-2},
    url = {https://CRAN.R-project.org/package=NeuralEstimators},
    doi = {10.32614/CRAN.package.NeuralEstimators},
  }

@Article{,
    title = {Likelihood-Free Parameter Estimation with Neural {B}ayes
      Estimators},
    author = {Matthew Sainsbury-Dale and Andrew Zammit-Mangion and
      Raphael Huser},
    journal = {The American Statistician},
    year = {2024},
    volume = {78},
    pages = {1--14},
    doi = {10.1080/00031305.2023.2249522},
  }
```

### Papers using NeuralEstimators

- **Likelihood-free parameter estimation with neural Bayes estimators** [[paper]](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522) [[code]](https://github.com/msainsburydale/NeuralBayesEstimators)

- **Neural Bayes estimators for censored inference with peaks-over-threshold models** [[paper]](https://arxiv.org/abs/2306.15642)

- **Neural Bayes estimators for irregular spatial data using graph neural networks** [[paper]](https://arxiv.org/abs/2310.02600)[[code]](https://github.com/msainsburydale/NeuralEstimatorsGNN)

- **Neural Methods for Amortized Inference** [[paper]](https://arxiv.org/abs/2404.12484)[[code]](https://github.com/andrewzm/Amortised_Neural_Inference_Review)

- **Neural parameter estimation with incomplete data** [[paper]](https://arxiv.org/abs/2501.04330)[[code]](https://github.com/msainsburydale/NeuralEM)

