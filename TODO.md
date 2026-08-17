# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed. Tasks marked 🔴 or 🟡 are high or medium priority; unmarked tasks are lower priority.

---

### Functionality

**Estimator types & methods**
- Model selection/comparison: see [here](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- Hierarchical models: see [this paper](https://arxiv.org/abs/2408.13230) and [this paper](https://arxiv.org/abs/2505.14429).
- Additional [approximate distributions](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/) for full posterior inference.
- Ensemble methods with general estimator types (e.g., PosteriorEstimator, RatioEstimator).

**Summary network architecture**
- 🟡 By default, [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) should condition on the (log) sample size (it's easy to forget). This can be done via a convenience constructor; given keyword argument `latent_dim`, calls `MLP` to construct the outer network and automatically conditions on (a learned embedding of) the sample size.

**Training**
- Support for reading data from disk during training, to handle data sets that are too large to fit in memory.

**Inference & diagnostics**
- Straightforward way to incorporate box parameter constraints and ensure that posterior samples are in the prior support.
- assess.jl/inference.jl for more general parameter shapes (currently assumes the parameters are stored as a matrix).
- Unidimensional coverage checks with TREs.

### Documentation
- 🟡 Add illustrative data figures, terminal training output, and diagnostic plots in all examples.
- In the Examples tab, index "Global parameters" and "Spatially indexed parameters" so it is clear that these are subsections, and put a hyperlink on "Gridded spatial data" with the two subsections in it (or at least with links to them).
- Add code groups for Lux/Flux (containing `using Lux`/`using Flux`) in the examples.
- Example: In the time-series example, also illustrate partially-exchangeable networks using DeepSet.
- Example: Illustrate Lévy Processes (a time-series model) using DeepSet (see [here](https://arxiv.org/abs/2505.01639)).
- Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)).
- Add [`::: tabs`](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/markdown-examples#Tabs) in the assessment stage of the examples to show the various diagnostic plots (recovery plots for point estimates; SBC and posterior contraction for posterior samples).
- Document the internal functions and add them to `API/Internal` or `API/Developer docs`. This will help with maintenance/contributions, and allow us to reference the internals when documenting public functions (e.g., "`kwargs...` are passed onto `_internal_function`").
- Add a gif to the README (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif)).
- Improve the [landing page](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) (see, e.g., [here](https://beautiful.makie.org/dev/) for inspiration).

### Performance
- 🟡 TRE: allow the summary statistics to stay on GPU during inference.
- Precompilation to reduce time-to-first-X (see, e.g., [here](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/precompilation.jl)).
- Reactant.jl in the inference stage.
- Find and remove type instabilities (test using [JET.jl](https://github.com/aviatesk/JET.jl)).
- For some operations involving only matrices and MLPs (e.g., inference-network transformations of summary statistics), it might be faster to always use the CPU (at least for certain batchsize ranges).
- SimpleChains.jl: are the user-friendly constructors for each estimator type correctly converted to `SimpleChainsLayers`?
- Lux.jl: Initial risks are much larger than Flux.jl when training NPEs.
- Add a check for NaNs in the inputs/outputs. Also, if the training risk or validation risk becomes NaN, immediately halt training.

### Backend
- 🟡 Lux support for [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures#Modules).
- 🟡 Lux support for [SpatialGraphConv](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures#Layers).
- The initial risks seem to be quite large when using Lux; use the same weight initialisation used by Flux.
- Lux support for [CovarianceMatrix/CorrelationMatrix](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures#Output-layers).
- Reactant support for [Gaussian](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions#Distributions) (issue is likely the triangular solve when computing the density).
- SimpleChains.jl: enforce `CPUDevice`/`AutoZygote` during training and `CPUDevice` during inference (dispatching on `SimpleChainsLayer` within `_resolvedevice` and `_resolve_adtype`).
- EnzymeRuntimeActivityError when using [NormalisingFlow](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions#Distributions) with Lux + Enzyme + CPU.

### Refactoring/API improvements
- Improve console output during training (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif), which uses [this](https://github.com/CarloLucibello/Tsunami.jl/blob/main/src/ProgressMeter/ProgressMeter.jl) code based on [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl/issues)).
- Clean and improve the plotting code/logic.
- Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl/Lux.jl.
- Automatically and reliably infer the number of summaries from an arbitrary `summary_network`, so that the user need not specify it when constructing an estimator.
   * This can be easily done for the common cases (Chain, DeepSet), with an `@info` given to tell the user what we inferred. For other cases, just error and tell the user to specify the number of summaries explicitly. Can also make the function used to compute the number of summaries public (and overloadable for custom structs). 

### Testing
- Automatic type-stability testing using [JET.jl](https://github.com/aviatesk/JET.jl).
- Automatic quality testing with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl).
- Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to prevent examples becoming outdated.