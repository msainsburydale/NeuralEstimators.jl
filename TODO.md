# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed.

---

### Features
- [ ] Support for [Lux.jl](https://lux.csail.mit.edu/stable/).
- [ ] Support for [Enzyme](https://fluxml.ai/Flux.jl/dev/reference/training/enzyme/). Currently, DeepSet does not work with `Enzyme.Duplicated` due to an error about using it with nested networks.
- [ ] Sequential training methods.
- [ ] Better functionality to resume training.
- [ ] Better functionality to visualize training/validation risk functions.
- [ ] Option to check the validation risk (and save the estimator) more frequently than the end of each epoch, to avoid wasted computation with very large training sets.
- [ ] Improve assessment stage with `RatioEstimator`.
- [ ] Ensemble methods with `PosteriorEstimator` and `RatioEstimator`.
- [ ] Explicit learning of summary statistics (see [Zammit-Mangion et al., 2025, Sec. 4](https://arxiv.org/pdf/2404.12484))
- [ ] Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality.
- [ ] Model misspecification detection: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.diagnostics.summary_space_comparison.html) and the references therein. 
- [ ] Model selection/comparison: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- [ ] Methods for high-dimensional parameter vectors (e.g., [telescopic ratio estimation](https://arxiv.org/pdf/2006.12204)).
- [ ] Parameter bounds when doing posterior inference (see [#38](https://github.com/msainsburydale/NeuralEstimators.jl/issues/38)).
- [ ] Learned embedding of the sample size $m$ in [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet). 

### Documentation
- [ ] Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs). See [Flux's in-built support for recurrence](https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/). 
- [ ] Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)
- [ ] Example: Spatio-temporal data.
- [ ] Document the plot() extension in the API section. 

### Testing
- [ ] Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to prevent examples becoming outdated.
- [ ] Clean `test/runtest.jl` and make the unit tests more systematic.
- [ ] Mirror the `src/` structure where possible (e.g., tests for `Architectures.jl` in `test/test_architectures.jl`).

### Performance 
- [ ] Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/). In particular, some of the custom structs in this package could be made more efficient by simply adding type parameters, as discussed in the [Flux's custom model example](https://fluxml.ai/Flux.jl/stable/tutorials/custom_layers/#Custom-Model-Example). 

### Refactoring
- [ ] Clarify the sections on "Adding a New Estimator" and "Adding a New Approximate Distribution" in [CONTRIBUTING.md](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/CONTRIBUTING.md) (also clean the latter approach if possible).
- [ ] Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl.
- [ ] For long term stability, it might be better to use Plots.jl, rather than AlgebraOfGraphics.jl and CairoMakie.jl. Or, try to only use CairoMakie.jl (i.e., remove AlgebraOfGraphics.jl).

---


