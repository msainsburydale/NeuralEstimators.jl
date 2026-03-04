# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed.

---

### Features
- [ ] Support for [Lux.jl](https://lux.csail.mit.edu/stable/).
- [ ] Support for [Enzyme](https://fluxml.ai/Flux.jl/dev/reference/training/enzyme/). Currently, [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) does not work with `Enzyme.Duplicated` due to an error about using it with nested networks.
- [ ] Sequential training methods.
- [ ] Summary-statistic-based model-misspecification detection: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.diagnostics.summary_space_comparison.html) and the references therein. 
- [ ] Model selection/comparison: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- [ ] Option to check the validation risk (and save the estimator) more frequently than the end of each epoch, to avoid wasted computation with very large training sets.
- [ ] Support for reading data from disk during training, to handle data sets that are too large to fit in memory.
- [ ] Improve assessment stage with `RatioEstimator`.
- [ ] Ensemble methods with `PosteriorEstimator` and `RatioEstimator`.
- [ ] Explicit learning of summary statistics (see [Zammit-Mangion et al., 2025, Sec. 4](https://arxiv.org/pdf/2404.12484)).
- [ ] Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality.
- [ ] Hierarchical models: see [this paper](https://arxiv.org/abs/2408.13230) and [this paper](https://arxiv.org/abs/2505.14429).
- [ ] Methods for high-dimensional parameter vectors (e.g., [telescoping ratio estimation](https://arxiv.org/abs/2510.04042)).
- [ ] Parameter bounds when doing posterior inference (see [#38](https://github.com/msainsburydale/NeuralEstimators.jl/issues/38)).
- [ ] Additional [approximate distributions](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/) for full posterior inference (see Tables 1 and 3 of [BayesFlow 2.0](https://arxiv.org/abs/2602.07098)).

### Documentation
- [ ] Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs). See [Flux's in-built support for recurrence](https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/). 
- [ ] Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)
- [ ] Example: Spatio-temporal data.

### Testing
- [] Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to prevent examples becoming outdated.
- [ ] Clean `test/runtest.jl`: make the tests more systematic, and mirror the `src/` structure where possible (e.g., possibly split the tests based on `src/`; tests for `Architectures.jl` in `test/test_architectures.jl`, etc.).

### Performance 
- [ ] Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/). In particular, some of the custom structs in this package could be made more efficient by simply adding type parameters, as discussed in the [Flux's custom model example](https://fluxml.ai/Flux.jl/stable/tutorials/custom_layers/#Custom-Model-Example). [Lux.jl](https://lux.csail.mit.edu/stable/) might also be faster.

### Refactoring
- [ ] Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl.
- [ ] Clean and improve the plotting code/logic.

---

### Breaking changes to decide upon before proceeding to version 1.0

* It might be best to have a `summary_network` and `inference_network` decomposition like `BayesFlow`. This would (i) facilitate summary-statistic-based model-misspecification detection; (ii) improve user-friendliness since, given the summary network, the inference network can be constructed automatically for most estimator types (irrespective of the data format, the inference network is always an MLP), and switching between estimator types would be easier; and (iii) facilitate transfer learning with pretrained summary networks. 
    * This could also be done in a looser, non-breaking way: just add constructors that allow the user to provide a summary network and the inference network (or the number of parameters in the model), and then define the network as a `Chain(summary_network = ..., inference_network = ...)`. 
    
    * Might also be able to do this without introducing the explicit notion of a summary network, by applying every component of `network` to the data and defining the "summary statistics" as the layer output with the smallest dimension. However, this would be difficult to make general/robust, particularly for networks that are not simple `Chain` objects that can be easily indexed (e.g., `DeepSet`).

* Might be helpful to store the loss function in `PointEstimator` objects. Mainly useful for knowing post-training (e.g., in a different session, later in time) how the estimator was trained, and for computing the risk at the assessment stage (however, this is a minor convenience, we could also just add a `loss` argument to `assess()`).

* Might be helpful to store number `d` of parameters in the estimator object (this also makes sense if we decide to make the first breaking change above, where we introduce `summary_network` and `inference_network` fields, since the default constructors would then `d` as input anyway). This would be useful for basic checks which would lead to better/more intuitive error messages.

* Rename the main plotting assessment function as `plotdiagnostics()` rather than `plot()`? Then, NeuralEstimators would own the function name and there wouldn't be any need to do `using CairoMakie: plot` in the tutorial.