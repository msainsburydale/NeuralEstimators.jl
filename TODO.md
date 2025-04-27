# TODO List

A checklist of planned tasks, improvements, and ideas for the package.

Feel free to update this file as tasks are completed, added, or changed.

---

### Features
- [ ] Explicit learning of summary statistics (see [Zammit-Mangion et al. Sec 4](https://arxiv.org/pdf/2404.12484))
- [ ] Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality 
- [ ] Training, option to check validation risk (and save the optimal estimator) more frequently than the end of each epoch. This could avoid wasted computation when we have very large training sets. 
- [ ] Improve assessment stage with `PosteriorEstimator` and `RatioEstimator` (i.e., improve `assess(estimator::PosteriorEstimator)` and `assess(estimator::RatioEstimator)` and add corresponding diagnostics, e.g., CRPS and interval score)
- [ ] Ensemble methods with `PosteriorEstimator` and `RatioEstimator`
- [ ] Support for [Enzyme](https://fluxml.ai/Flux.jl/dev/reference/training/enzyme/). Currently, DeepSet does not work with `Enzyme.Duplicated` due to an error about using it with nested networks
- [ ] Support for model selection/comparison: see the [BayesFlow example](https://bayesflow.org/main/_examples/One_Sample_TTest.html) and [this paper](https://arxiv.org/pdf/2503.23156)

### Documentation
- [ ] Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs). See [Flux's in-built support for recurrence](https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/). 
- [ ] Example: Bivariate data in a "multivariate" section
- [ ] Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)

### Testing
- [ ] Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to avoid documentation examples from becoming outdated

### Performance 
- [ ] Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/). In particular, some of the custom structs in this package could be made more efficient by simply adding type parameters, as discusses in the [Flux's custom model example](https://fluxml.ai/Flux.jl/stable/tutorials/custom_layers/#Custom-Model-Example). 

### Refactoring
- [ ] Move DeepSet to Flux.jl
- [ ] For long term stability, it might be better to use Plots.jl, rather than AlgebraOfGraphics.jl and CairoMakie.jl

---


