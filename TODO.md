# TODO List

A checklist of planned tasks, improvements, and ideas for the package. Feel free to update this file as tasks are completed, added, or changed. Tasks marked 🔴 or 🟡 are high or medium priority; unmarked tasks are lower priority.

---

### Features

**Backend**
- 🔴 Support for [Lux.jl](https://lux.csail.mit.edu/stable/).
- Support for [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) to train small neural networks quickly on the CPU (see [this](@ref) blog post; see also the [Lux docs](https://lux.csail.mit.edu/stable/api/Lux/interop#Lux-Models-to-Simple-Chains) for converting Lux models to simple chains).

**Training**
- 🟡 Sequential training methods.
- Support for reading data from disk during training, to handle data sets that are too large to fit in memory.
- 🟡 Post-training calibration for better inference (especially PointEstimator and RatioEstimator, but can easily do this for general estimators).

**Estimator types & methods**
- 🟡 Improve inference methods with RatioEstimator (alternative to grid-based sampling, e.g., MCMC).
- 🟡 Methods for high-dimensional parameter vectors (e.g., [telescoping ratio estimation](https://arxiv.org/abs/2510.04042)).
- Model selection/comparison: see [here](https://bayesflow.org/main/api/bayesflow.approximators.ModelComparisonApproximator.html#bayesflow.approximators.ModelComparisonApproximator), [this paper](https://arxiv.org/abs/2004.10629), and [this paper](https://arxiv.org/pdf/2503.23156).
- Hierarchical models: see [this paper](https://arxiv.org/abs/2408.13230) and [this paper](https://arxiv.org/abs/2505.14429).
- Additional [approximate distributions](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/approximatedistributions/) for full posterior inference.
- Explicit learning of summary statistics (see [Zammit-Mangion et al., 2025, Sec. 4](https://arxiv.org/pdf/2404.12484)).
- Ensemble methods with general estimator types (e.g., PosteriorEstimator, RatioEstimator).

**Inference & diagnostics**
- 🟡 Summary-statistic-based model-misspecification detection: see [here](https://bayesflow.org/main/api/bayesflow.diagnostics.summary_space_comparison.html) and the references therein. Once implemented, document by adding a subsection "Detecting model misspecification", or similar, at the end of the page "API/Inference with observed data".
- Parameter bounds when doing posterior inference (see [#38](https://github.com/msainsburydale/NeuralEstimators.jl/issues/38)).
- Incorporate [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) (possibly as an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions)) to expand bootstrap functionality.
- assess.jl/inference.jl for more general parameter shapes (currently assumes the parameters are stored as a matrix).

**Summary network architecture**
- Add several ready-to-go summary networks (e.g., for gridded data, time-series, etc.).
- Functions for automated neural architecture search (see, e.g., [this paper](https://www.jmlr.org/papers/volume20/18-598/18-598.pdf)) using for example, [evolutionary algorithms](https://en.wikipedia.org/wiki/Neural_architecture_search#Evolution) or [Bayesian optimization](https://en.wikipedia.org/wiki/Neural_architecture_search#Bayesian_optimization).
- Automatically and reliably infer the number of summaries from an arbitrary `summary_network`, so that the user need not specify it when constructing an estimator.

### Documentation
- 🔴 Lux.jl: Once we've added support for Lux, update the documentation to reflect that either Flux or Lux can be used (use `codegroup`s in the examples to choose which deep-learning package and to define the neural networks).
- Update/improve the logo and home page (see, e.g., [here](https://beautiful.makie.org/dev/)).
- Citations: Use proper citation manager (see [here](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/citations)).
- Examples: Add [`::: tabs`](https://luxdl.github.io/DocumenterVitepress.jl/dev/manual/markdown-examples#Tabs) in the assessment stage to show the various diagnostic plots (recovery plots for point estimates; SBC and posterior contraction for posterior samples).
-  Examples: Don't use DeepSets in CNN (just reference DeepSet); also think about this for GNN. Maybe retain DeepSet so that we can illustrate variable grid sizes using `GlobalMeanPool` (not clear how variable grid sizes could be handled during training otherwise)? Could do a `::: codegroup` to illustrate both networks (could also do this for the univariate data example, showing both DeepSet and regular MLP).
- Example: Sequence (e.g., time-series) input using recurrent neural networks (RNNs).
- Example: Spatio-temporal data.
- Example: Nonstationary spatial data with image-to-image networks (see, e.g., [LatticeVision](https://arxiv.org/abs/2505.09803)).
- Example: Discrete parameters (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)). (Might need extra functionality for this.)
- Example: Lévy Processes using DeepSet (see [here](https://arxiv.org/abs/2505.01639)).
- Clean Advanced usage; move "expert summary statistics", "censored data", and "missing data" to the examples section (each with their own example page), and merge "Variable sample sizes" into the section on replciated data (possibly as a "Bonus" subsection at the end).
- Some of the API pages could be organised better with subsections.
- Add a gif to the README (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif)).
- GitHub: Remove the workshop branch and update the tutorial.

### Performance
- 🔴 [Lux.jl](https://lux.csail.mit.edu/stable/) with Reactant.jl might be faster.
- 🟡 Prcompilation to reduce time-to-first-X (see, e.g., [here](https://github.com/SciML/DiffEqFlux.jl/blob/master/src/precompilation.jl)). 
- Improve the efficiency of the code where possible. See the general [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) that could apply, and the [Flux performance tips](https://fluxml.ai/Flux.jl/stable/guide/performance/).
- Some operations involving only matrices and MLPs (e.g., inference network transformations of summary statistics) should default to using the CPU.
- Find and remove type instabilities (test using [JET.jl](https://github.com/aviatesk/JET.jl)).
- SimpleChains.jl could be utilized for faster training of "small" feedforward networks on the CPU.

### Refactoring/API improvements
- Clean and improve the plotting code/logic.
- Improve console output during training (see, e.g., [here](https://github.com/CarloLucibello/Tsunami.jl/blob/main/docs/src/assets/readme_training.gif), which uses [this](https://github.com/CarloLucibello/Tsunami.jl/blob/main/src/ProgressMeter/ProgressMeter.jl) code based on [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl/issues)).
- Move [DeepSet](https://msainsburydale.github.io/NeuralEstimators.jl/dev/API/architectures/#NeuralEstimators.DeepSet) to Flux.jl/Lux.jl.

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


