# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed. Tasks marked 🔴 or 🟡 are high or medium priority; unmarked tasks are lower priority.

---

### Features

**Backend**
- 🟡 Support for [Lux.jl](https://lux.csail.mit.edu/stable/).

**Training**
- 🟡 Sequential training methods.
- Support for reading data from disk during training, to handle data sets that are too large to fit in memory.
- 🟡 Post-training calibration for better inference (especially PointEstimator and RatioEstimator, but can easily do this for general estimators).

**Estimator types & methods**
- Hierarchical models: see [this paper](https://arxiv.org/abs/2408.13230) and [this paper](https://arxiv.org/abs/2505.14429).
- Model selection/comparison: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- 🟡 Methods for high-dimensional parameter vectors (e.g., [telescoping ratio estimation](https://arxiv.org/abs/2510.04042)).
- Additional [approximate distributions](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/) for full posterior inference (see Tables 1 and 3 of [BayesFlow 2.0](https://arxiv.org/abs/2602.07098)).
- Explicit learning of summary statistics (see [Zammit-Mangion et al., 2025, Sec. 4](https://arxiv.org/pdf/2404.12484)).
- Ensemble methods with general estimator types (e.g., PosteriorEstimator, RatioEstimator).
- Better inference methods with RatioEstimator.

**Inference & diagnostics**
- 🟡 Summary-statistic-based model-misspecification detection: see the [BayesFlow documentation](https://bayesflow.org/main/api/bayesflow.diagnostics.summary_space_comparison.html) and the references therein.
- Parameter bounds when doing posterior inference (see [#38](https://github.com/msainsburydale/NeuralEstimators.jl/issues/38)).
- Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality.
- assess.jl/inference.jl for more general parameter shapes (currently assumes the parameters are stored as a matrix).

**Summary network architecture**
- Functions for automated neural architecture search (see, e.g., [this paper](https://www.jmlr.org/papers/volume20/18-598/18-598.pdf)) using for example, [evolutionary algorithms](https://en.wikipedia.org/wiki/Neural_architecture_search#Evolution) or [Bayesian optimization](https://en.wikipedia.org/wiki/Neural_architecture_search#Bayesian_optimization).
- Automatically and reliably infer the number of summaries from an arbitrary `summary_network`, so that the user need not specify it when constructing an estimator.
- Add several ready-to-go summary networks (e.g., for gridded data, time-series, etc.).


### Documentation
- 🟡 Update the examples to reflect the new API (in particular, the explicit summary-inference-network decomposition).
- Use [DocumenterVitepress.jl](https://luxdl.github.io/DocumenterVitepress.jl/dev/) as the backend for Documenter.jl (more modern and polished docs; see, e.g., [Lux.jl](https://lux.csail.mit.edu/stable/)).
- Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs). See [Flux's in-built support for recurrence](https://fluxml.ai/Flux.jl/stable/guide/models/recurrence/). 
- Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)
- Example: Spatio-temporal data.
- Add a gif to the README (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif)).

### Performance 
- Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/). Using [Lux.jl](https://lux.csail.mit.edu/stable/) might also be faster?
- 🟡 Some operations should always be done on the CPU, specifically those involving only matrices and MLPs (e.g., inference network transformations on (learned) summary statistics, mappings performed in `GaussianMixture` and `NormalisingFlow`).

### Refactoring/API improvements
- Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl.
- Clean and improve the plotting code/logic.
- Update all `QuantileEstimator` types to employ `summary_network`s.
- Improve console output during training (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif), which uses [this](https://github.com/CarloLucibello/Tsunami.jl/blob/main/src/ProgressMeter/ProgressMeter.jl) code based on [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl/issues)).
- 🟡 rename `subsetdata` to `subsetreplicates`.

### Testing
- Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to prevent examples becoming outdated.
- Clean `test/runtest.jl`: make the tests more systematic, and mirror the `src/` structure where possible (e.g., possibly split the tests based on `src/`; tests for `Architectures.jl` in `test/test_architectures.jl`, etc.).
- Improve code coverage (e.g., plotting extensions).

---

### Breaking changes to decide upon before version 1.0

- Add a `base_distribution` field in `NormalisingFlow` (default standard Normal).

- Might be helpful to store the loss function in `PointEstimator` objects. 
   * Mainly useful for knowing post-training how the estimator was trained, and for computing the risk at the assessment stage (however, this is a minor convenience, we could also just add a `loss` argument to `assess`).
   * Otherwise, could just save the loss function as metadata during training (`assess` could then load it from `tmp` or `savepath`).

- Might be helpful to store the number of parameters `num_parameters` in the estimator object. 
   * This would be useful for basic checks which would lead to better/more intuitive error messages. This also makes sense with the new summary network decomposition, since these constructors already require `num_parameters`.
   * Main drawback is that it bloats the struct slightly; and estimators don't necessarily need to be initialised with `num_parameters` explicitly given, in which case this field would then be empty.


