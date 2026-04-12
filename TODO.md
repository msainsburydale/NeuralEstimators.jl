# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed. Tasks marked 🔴 or 🟡 are high or medium priority; unmarked tasks are lower priority.

---

### Features

**Backend**
- 🟡 Support for [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures#Modules) with Lux.
- 🟡 Support for [NormalisingFlow](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions#Distributions) with Lux.
- SimpleChains.jl: for user-friendliness, enforce `CPUDevice`/`AutoZygote` during training and `CPUDevice` during inference (dispatching on `SimpleChainsLayer` within `_resolvedevice` and `_resolve_adtype`). NB: `SimpleChainsLayer` is not in LuxCore, so this dispatch may need to live in the extension.

**Training**
- Support for reading data from disk during training, to handle data sets that are too large to fit in memory.

**Estimator types & methods**
- Model selection/comparison: see [here](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- Hierarchical models: see [this paper](https://arxiv.org/abs/2408.13230) and [this paper](https://arxiv.org/abs/2505.14429).
- Additional [approximate distributions](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/) for full posterior inference.
- Ensemble methods with general estimator types (e.g., PosteriorEstimator, RatioEstimator).

**Inference & diagnostics**
- assess.jl/inference.jl for more general parameter shapes (currently assumes the parameters are stored as a matrix).

**Summary network architecture**
- By default, [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) should condition on the (log) sample size (it's messy for the user to include manually and easy to forget). This can be done via a convenience constructor; given keyword argument `latent_dim`, calls `mlp` to construct the outer network, and we automatically condition on the sample size (and we can add a learned embedding of the sample size).

### Documentation
- 🟡 Update/improve the logo.
- 🟡 Diagram illustrating the general workflow.
- Improve the [landing page](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) (see, e.g., [here](https://beautiful.makie.org/dev/), [here](https://lux.csail.mit.edu/stable/), and [here](https://timweiland.github.io/GaussianMarkovRandomFields.jl/stable/) for inspiration). For example, we could add some landing-page boxes (Box 1: NPEs, NREs, NBEs. Box 2: Multibackend: Flux.jl, Lux.jl, SimpleChains.jl)
- Once [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures#Modules) is supported with Lux, add code groups for Flux/Lux (containing `using Flux`/`using Lux`) in the examples.
- Example: Sequence (e.g., time-series) data, either using recurrent neural networks (RNNs) or partially-exchangeable networks based on DeepSet.
- Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)).
- Example: Lévy Processes using DeepSet (see [here](https://arxiv.org/abs/2505.01639)).
- Add [`::: tabs`](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/markdown-examples#Tabs) in the assessment stage of the examples to show the various diagnostic plots (recovery plots for point estimates; SBC and posterior contraction for posterior samples).
- Clean Advanced usage; move "expert summary statistics", "censored data", and possibly "missing data" to the examples section (each with their own example page), and merge "Variable sample sizes" into the section on replicated data (possibly as a "Bonus" subsection at the end).
- Citations: Use proper citation manager (see [here](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/citations)).
- Document the internal functions and add them to `API/Internal` or `API/Developer docs`. This will help with maintenance/contributions, and allow us to reference the internals when documenting public functions (e.g., "`kwargs...` are passed onto `_internal_function`").
- Add a gif to the README (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif)).

### Performance
- 🟡 Precompilation to reduce time-to-first-X (see, e.g., [here](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/precompilation.jl)).
- 🟡 Lux + Reactant currently has extra overhead during training: see the TODO in the Reactant extension.
- 🟡 Find and remove type instabilities (test using [JET.jl](https://github.com/aviatesk/JET.jl)).
- For some operations involving only matrices and MLPs (e.g., inference-network transformations of summary statistics), it might be faster to always use the CPU (at least for certain batchsize ranges).
- SimpleChains.jl: are the user-friendly constructors for each estimator type correctly converted to `SimpleChainsLayers`?

### Refactoring/API improvements
- 🟡 Automatically and reliably infer the number of summaries from an arbitrary `summary_network`, so that the user need not specify it when constructing an estimator.
   * This can be easily done for the common cases (Chain, DeepSet), with an `@info` given to tell the user what we inferred and to change it if it is wrong. For other cases, just error and tell the user to specify the number of summaries explicitly.
- Clean and improve the plotting code/logic.
- 🟡 Improve console output during training (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif), which uses [this](https://github.com/CarloLucibello/Tsunami.jl/blob/main/src/ProgressMeter/ProgressMeter.jl) code based on [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl/issues)).
- Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl/Lux.jl.
- Consider [StatefulLuxLayer](https://lux.csail.mit.edu/stable/manual/flux_lux_interop) as a replacement for `LuxEstimator`. (Currently, it has the same problem as `TrainState`, `model` needs to be an `AbstractLuxLayer`, but perhaps this can be relaxed.)

### Testing
- Turn some of the docstring examples into [doctests](https://documenter.juliadocs.org/stable/man/doctests/) for automatic checking of examples and to prevent examples becoming outdated.
- Improve code coverage (including extensions).
- Automatic type-stability testing using [JET.jl](https://github.com/aviatesk/JET.jl).
- Automatic quality testing with [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl).

### General
- The package would benefit by leveraging the SciML ecosystem (see, e.g., [here](https://docs.sciml.ai/DiffEqFlux/dev/)), for instance, using [Neural Ordinary Differential Equations](https://docs.sciml.ai/DiffEqFlux/dev/) for fast surrogate simulators; summary networks (i.e., data encoders); and approximate distributions for posterior estimation.

---

### 🟡 Breaking changes to decide upon before version 1.0

These changes would alter the fields of estimator objects, making it more difficult to load estimators saved in previous versions (although user-facing API would remain unchanged).

- Add a `base_distribution` field in `NormalisingFlow` (default standard Normal).

- Might be helpful to store the loss function in `PointEstimator` objects. 
   * Mainly useful for knowing post-training how the estimator was trained, and for computing the risk at the assessment stage (however, this is a minor convenience, we could also just add a `loss` argument to `assess`).
   * Otherwise, could just save the loss function as metadata during training (`assess` could then load it from `tmp` or `savepath`).

- Might be helpful to store the number of parameters `num_parameters` in the estimator object. 
   * This would be useful for basic checks which would lead to better/more intuitive error messages. This also makes sense with the new summary network decomposition, since these constructors already require `num_parameters`.
   * Main drawback is that it bloats the struct slightly; and estimators don't necessarily need to be initialised with `num_parameters` explicitly given, in which case this field would then be empty.


