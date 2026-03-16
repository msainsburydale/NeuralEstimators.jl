# Contributing to NeuralEstimators

Thank you for your interest in contributing to **NeuralEstimators**! We welcome contributions of all sizes, from fixing typos to implementing new estimator types. This document describes the code structure, development workflow, and conventions for submitting contributions.

---

## Table of Contents

1. [Code Structure](#code-structure)
1. [Key Abstractions](#key-abstractions)
1. [Development Workflow](#development-workflow)
1. [Adding a New Estimator](#adding-a-new-estimator)
1. [Adding a New Approximate Distribution](#adding-a-new-approximate-distribution)
1. [Adding a New Architecture](#adding-a-new-architecture)
1. [Adding a New Documentation Example](#adding-a-new-documentation-example)

---

## Code Structure

All source files live under `src/`. The entry point is `NeuralEstimators.jl`, which declares the module, imports dependencies, defines exports, and `include`s every subfile.

```
src/
├── NeuralEstimators.jl          # Module entry point: imports, exports, includes
│
├── Estimators/                  # Neural estimator types
│   ├── Estimators.jl            # Abstract types (NeuralEstimator, BayesEstimator) and summary-network helpers
│   ├── PointEstimator.jl        # Point estimates 
│   ├── QuantileEstimator.jl     # Posterior quantile estimation
│   ├── PosteriorEstimator.jl    # Full posterior approximation
│   ├── RatioEstimator.jl        # Likelihood-to-evidence ratio estimation
│   ├── Ensemble.jl              # Ensemble of estimators
│
├── ApproximateDistributions/    # Approximate distributions used by PosteriorEstimator
│   ├── ApproximateDistributions.jl  # Abstract type (ApproximateDistribution)
│   ├── GaussianMixture.jl       
│   ├── NormalisingFlow.jl       
│
├── train.jl                     # Training function (train)
├── assess.jl                    # Post-training assessment (assess, Assessment, etc.)
├── Parameters.jl                # AbstractParameterSet and utilities
├── DataSet.jl                   # DataSet struct and utilities
├── Architectures.jl             # Neural-network building blocks (DeepSet, MLP, etc.)
├── inference.jl                 # Post-training inference (estimate, sampleposterior, etc.)
├── summarystatistics.jl         # Expert summary statistics (samplesize, etc.)
├── losses.jl                    # Non-standard loss functions (quantileloss, tanhloss, etc.)
├── missingdata.jl               # Missing data support (EM algorithm, masking)
├── utility.jl                   # Utility functions
```

### Including Files

New `.jl` files in `src/` must be explicitly `include`d in `NeuralEstimators.jl`. Files inside `src/Estimators/` and `src/ApproximateDistributions/` are auto-included in alphabetical order. Any new file added to these subfolders will therefore be picked up automatically, but its public symbols should still be exported from `NeuralEstimators.jl`.

---

## Key Abstractions

### `NeuralEstimator` and `BayesEstimator`

Defined in `Estimators/Estimators.jl`, these are the root abstract types of the estimator hierarchy:

```
NeuralEstimator
├── BayesEstimator
│   ├── PointEstimator
│   └── QuantileEstimator (and variants)
├── PosteriorEstimator
├── RatioEstimator
├── Ensemble
```

Every concrete estimator wraps one or more Flux neural networks and must be callable on data `Z` (and possibly parameters `θ`).

### `ApproximateDistribution`

An abstract type (defined in `ApproximateDistributions.jl`) for families of parametric distributions used to approximate the posterior distribution by objects of type `PosteriorEstimator`. 

### `AbstractParameterSet`

An abstract type (defined in `Parameters.jl`) for user-defined structs that hold parameters and any precomputed intermediate quantities needed for simulation. 

### `DataSet`

A struct (defined in `DataSet.jl`) that couples raw data `Z` with a matrix `S` of precomputed expert summary statistics (`K` columns, one per data set). Passing a `DataSet` object to any estimator causes the learned summary statistics from the summary network to be concatenated with `S` before being passed to the inference network. If `S` is not provided, `DataSet(Z)` behaves identically to passing `Z` directly. The internal forward-pass helper `_summarystatistics` handles the concatenation.

### `train` and `assess`

`train` (in `train.jl`) is the universal training interface. It accepts any `NeuralEstimator` and dispatches on the estimator type to select the appropriate loss. `assess` (in `assess.jl`) evaluates a trained estimator and returns an `Assessment` object that supports downstream visualizations and diagnostics (e.g., `risk`, `bias`, `rmse`, and `coverage`.)

---

## Development Workflow

1. **Fork and clone** the repository.
1. **Activate the package environment** from the repository root:
   ```julia
   using Pkg; Pkg.activate("."); Pkg.instantiate()
   ```
   or
   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```
1. **Make your changes** in the appropriate source file. For iterative development, it is often easier to prototype interactively first: load `NeuralEstimators` in a Julia session, import any internal functions you need to extend (e.g., `import NeuralEstimators: sampleposterior`), and define or modify methods on the fly without restarting the session. Once things are working, move the definitions into the appropriate source file.
1. **Export new public symbols** in `NeuralEstimators.jl`.
1. **Write or update documentation** in the `docs` directory (see [docs/README.md](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/README.md)) and add a docstring on every exported symbol (use `@doc raw"""..."""` to allow unescaped LaTeX).
1. **Write or update tests** in the `test/` directory.
1. **Run tests** with:
   ```julia
   Pkg.test("NeuralEstimators")
   ```
   or
   ```bash
   julia --project=. -e "using Pkg; Pkg.test()"
   ```
1. **Open a pull request** describing your changes. 

Please try to follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/) where possible.

---

## Adding a New Estimator

The recommended workflow for developing a new estimator type is to first prototype interactively, which allows you to rapidly iterate and test changes on the fly without restarting the session, and then consolidate into the package once things are working.

**Interactive prototyping:**
1. Define a custom struct subtyping `NeuralEstimator`, for example, `struct MyEstimator <: NeuralEstimator`. This struct will have fields storing the neural networks and anything else needed for inference.
1. Define convenience constructors. 
1. Define the forward pass of the estimator: `(estimator::MyEstimator)(input) = ...`  
   * When applying the summary network to data, use `_summarystatistics(estimator, Z)` rather than `estimator.summary_network(Z)` to automatically cater for `DataSet` objects.
1. `import` and extend the relevant functions listed below.
1. Train and assess the estimator to verify correctness.
1. Write a docstring for the estimator with a self-contained example that can be copy-pasted and run.

**Consolidating into the package:**
1. Create `src/Estimators/MyEstimator.jl` and move all of the above definitions into it.
1. Export `MyEstimator` in `NeuralEstimators.jl`.

**Which methods to define:**

After running `import NeuralEstimators: <function name>`:

- Training: 
   - `_loss(estimator::MyEstimator, loss)`: by default, returns `loss` unchanged. Define `_loss(estimator::MyEstimator, loss = nothing)` to enforce a specific objective regardless of what the user provides.

   - `_inputoutput(estimator, Z, θ)`: by default, returns `(Z, θ)` as the `(input, output)` pair, where `Z` is the simulated data and `θ` is the true parameters, so that training minimises `loss(estimator(Z), θ)`. Override if the estimator requires a different input/output structure.

   - `_risk(estimator, loss, dataset, device, optimiser)`: by default, iterates over `(input, output)` batches in `dataset` and computes `loss(estimator(input), output)`. Override only if the required computation cannot be expressed through `_loss` or `_inputoutput` alone.
- Inference: extend existing functions (e.g., `sampleposterior`) or define new ones as needed. 
- Assessment: if your estimator returns point estimates via `estimate` or posterior
  samples via `sampleposterior`, it will work with `assess` automatically (just
  add it to the relevant `Union` in `assess.jl`). Otherwise, implement a custom
  `assess` method.

---

## Adding a New Approximate Distribution

**Interactive prototyping:**
1. Define a custom struct subtyping `ApproximateDistribution`, for example, `struct MyDist <: ApproximateDistribution`. The fields of this struct should store anything needed to parameterise the distribution, including any neural network(s) used to condition the distribution on the learned summary statistics (e.g., an MLP mapping summary statistics to distributional parameters, or coupling blocks that accept summary statistics as a conditioning input as in a normalising flow).
1. Define a convenience constructor with the signature: `MyDist(num_parameters::Integer, num_summaries::Integer; kwargs...)`. This ensures compatibility with `PosteriorEstimator`'s convenience constructor, which calls `q(num_parameters, num_summaries; kwargs...)` internally.
1. Import the following functions: `import NeuralEstimators: logdensity, sampleposterior, numdistributionalparams`.
1. Define methods:
- `logdensity(q::MyDist, θ::AbstractMatrix, t::AbstractMatrix)`: log-density of `q` evaluated at parameters `θ`, given summary statistics `t`. Required for training.
- `sampleposterior(q::MyDist, t::AbstractMatrix, N::Integer)`: draws `N` posterior samples from `q` given summary statistics `t`, where each column of `t` corresponds to an independent data set. Required for inference.
- `numdistributionalparams(q::MyDist)`: number of distributional parameters. Optional.
1. Test the distribution by plugging it into a `PosteriorEstimator`, then training and assessing the estimator to verify correctness.
1. Write a docstring with a self-contained example that can be copy-pasted and run.

**Consolidating into the package:**
1. Create `src/ApproximateDistributions/MyDist.jl` and move all of the above definitions into it.
1. Export `MyDist` in `NeuralEstimators.jl`.


---

## Adding a New Architecture

In general, we leave the construction of neural network architectures to the user: the package is intentionally agnostic to the specific architecture used, and most standard Flux layers and containers work out of the box. 

That said, reusable building blocks that are broadly applicable and not already available in Flux (e.g., `DeepSet`, `MLP`) can be added to `Architectures.jl`. Follow the Flux layer pattern (a struct with fields for parameters, and a callable method), and export the type and any associated constructor helpers from `NeuralEstimators.jl`.

---

## Adding a New Documentation Example

Documentation examples live in `docs/src/examples/`. Each example is a self-contained `.md` file that walks through the full workflow for a specific type of data, as the data structure determines the appropriate neural network architecture.

**To add a new example:**

1. Create a new file `docs/src/examples/data_<name>.md` and write the example following the general workflow:
   - Introductory paragraph describing the statistical model and parameters.
   - `## Package dependencies` — list all required packages, including the GPU code-group block.
   - `## Sampling parameters` — define a `sampler` function.
   - `## Simulating data` — define a `simulator` function.
   - `## Constructing the neural network` — construct the summary network.
   - `## Constructing the neural estimator` — use the code-group block for `PointEstimator`, `PosteriorEstimator`, and `RatioEstimator`.
   - `## Training the estimator` — call `train`.
   - `## Assessing the estimator` — call `assess` and show diagnostics.
   - `## Applying the estimator to observed data` — use the code-group block for the three estimator types.
   
   See `docs/src/examples/data_replicated.md` for a worked example to follow as a template.
2. Register the new file in `makedocs` in `docs/make.jl` by adding it to the `"Examples"` section of the `pages` argument.
3. Build and preview the documentation locally (see [docs/README.md](https://github.com/msainsburydale/NeuralEstimators.jl/blob/main/docs/README.md)).