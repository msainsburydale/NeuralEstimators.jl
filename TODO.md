# TODO List

A checklist of planned tasks, improvements, and ideas for the package.

Feel free to update this file as tasks are completed, added, or changed.

---

### Features
- [ ] Explicit learning of summary statistics (see [Zammit-Mangion et al., 2025, Sec. 4](https://arxiv.org/pdf/2404.12484))
- [ ] Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality 
- [ ] During training, add an ption to check the validation risk (and save the estimator) more frequently than the end of each epoch. This could avoid wasted computation when we have very large training sets.
- [ ] Improve assessment stage with `PosteriorEstimator` and `RatioEstimator`: add methods that assess the full posterior rather than point estimates, and add diagnostics (e.g., CRPS and interval score)
- [ ] Ensemble methods with `PosteriorEstimator` and `RatioEstimator`
- [ ] Support for [Enzyme](https://fluxml.ai/Flux.jl/dev/reference/training/enzyme/). Currently, DeepSet does not work with `Enzyme.Duplicated` due to an error about using it with nested networks
- [ ] Model selection/comparison: see the [BayesFlow example](https://bayesflow.org/main/_examples/One_Sample_TTest.html) and [this paper](https://arxiv.org/pdf/2503.23156)
- [ ] [Telescopic ratio estimation](https://arxiv.org/pdf/2006.12204)
- [ ] Gaussian mixture [approximate distribution](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/), as per, for example, [Papamakarios and Murray (2016)](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html). In this case, the density of the approximate posterior distribution is: $q(\theta \mid Z) = \sum_{j=1}^{J} \pi_j(Z) \cdot \mathcal{N}(\theta \mid \mu_j(Z), \Sigma_j(Z))$, where the neural network outputs the mixture weights $\pi_j(Z) \in [0, 1]$ with $\sum_{j=1}^{J} \pi_j(Z) = 1$, the mean $\mu_j(Z) \in \mathbb{R}^d$ of each component, and the covariance matrix $\Sigma_j(Z) \in \mathbb{R}^{d \times d}$ of each component. Use a [softmax activation](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/#NNlib.softmax) for the network outputs that correspond to the mixture weights. Use diagonal covariance matrices by default; optionally use full covariance matrices, implemented with [CovarianceMatrix](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.CovarianceMatrix).
- [ ] Parameter bounds when doing posterior inference (see [#38](https://github.com/msainsburydale/NeuralEstimators.jl/issues/38)).

### Documentation
- [ ] Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs). See [Flux's in-built support for recurrence](https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/). 
- [ ] Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)
- [ ] Example: Spatio-temporal data

### Testing
- [ ] Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to avoid documentation examples from becoming outdated
- [ ] Clean `test/runtest.jl` (including the TODO comments), and make the unit tests more systematic

### Performance 
- [ ] Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/). In particular, some of the custom structs in this package could be made more efficient by simply adding type parameters, as discusses in the [Flux's custom model example](https://fluxml.ai/Flux.jl/stable/tutorials/custom_layers/#Custom-Model-Example). 

### Refactoring
- [ ] Move DeepSet to Flux.jl
- [ ] For long term stability, it might be better to use Plots.jl, rather than AlgebraOfGraphics.jl and CairoMakie.jl

---


