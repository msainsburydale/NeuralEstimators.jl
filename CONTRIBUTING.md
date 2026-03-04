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

---

## Code Structure

All source files live under `src/`. The entry point is `NeuralEstimators.jl`, which declares the module, imports dependencies, defines exports, and `include`s every subfile.

```
src/
├── NeuralEstimators.jl          # Module entry point: imports, exports, includes
│
├── Estimators/                  # Neural estimator types
│   ├── Estimators.jl            # Abstract types: NeuralEstimator, BayesEstimator
│   ├── PointEstimator.jl        # Point estimates 
│   ├── QuantileEstimator.jl     # Posterior quantile estimation
│   ├── PosteriorEstimator.jl    # Full posterior approximation
│   ├── RatioEstimator.jl        # Likelihood-to-evidence ratio estimation
│   ├── Ensemble.jl              # Ensemble of estimators
│
├── ApproximateDistributions/    # Approximate posterior distributions used by PosteriorEstimator
│   ├── GaussianMixture.jl       
│   ├── NormalisingFlow.jl       
│
├── Parameters.jl                # ParameterConfigurations abstract type and utilities
├── Architectures.jl             # Neural-network building blocks (DeepSet, MLP, etc.)
├── train.jl                     # Training functions (train)
├── assess.jl                    # Estimator assessment (assess, Assessment, risk, bias, rmse, etc.)
├── inference.jl                 # Post-training inference (estimate, sampleposterior, bootstrap, etc.)
├── summarystatistics.jl         # Expert summary statistics (samplesize, samplecorrelation, etc.)
├── losses.jl                    # Non-standard loss functions (quantileloss, tanhloss, etc.)
├── missingdata.jl               # Missing data support (EM algorithm, masking)
├── utility.jl                   # Miscellaneous utility functions
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

An abstract type (defined in `ApproximateDistributions.jl`) for families of parameteric distributions used to approximate the posterior distribution by objects of type `PosteriorEstimator`. 

### `ParameterConfigurations`

An abstract type (defined in `Parameters.jl`) for user-defined structs that hold parameter matrices and any precomputed intermediate quantities needed for simulation. The only required field is `θ`, a `d × K` matrix of parameter vectors. Implement `subsetparameters` if your type stores additional fields that should be subsetted consistently.

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

The recommended workflow for developing a new estimator type is to first prototype interactively, and then consolidate into the package once things are working.

**Interactive prototyping:**
1. Once you've started Julia and loaded `NeuralEstimators`, define a custom struct subtyping `NeuralEstimator`, for example, `struct MyEstimator <: NeuralEstimator`. This struct will have fields storing the neural network and anything else needed for inference.
1. Define the forward pass of the estimator: `(estimator::MyEstimator)(input) = ...`
1. Import any public or internal functions you need to extend (see below for which ones are needed): `import NeuralEstimators: _loss, _inputoutput`.
1. Extend the relevant functions interactively. This allows you to rapidly iterate and test changes on the fly without restarting the session.
1. Train and assess the estimator interactively to verify correctness.
1. Write a docstring for the estimator with a self-contained example that can be copy-pasted and run.

**Consolidating into the package:**
1. Create `src/Estimators/MyEstimator.jl` and move all of the above definitions into it.
1. Export `MyEstimator` in `NeuralEstimators.jl`.

**Which methods to extend:**
- Training: 
   - `_loss(estimator, loss)`: By default, returns the user-specified `loss` unchanged. Define `_loss(estimator::MyEstimator, loss = nothing)` to override this and enforce a specific objective regardless of what the user provides. Note that the two-argument form is required for compatibility with how the function is called in `train()`.

   - `_inputoutput(estimator, Z, θ)`: By default, returns `(Z, θ)` as the `(input, output)` pair, where `Z` is the simulated data and `θ` is the true parameters, so that training minimises `loss(estimator(Z), θ)`. Override this method if the estimator requires a different input/output structure.

   - `_risk(estimator, loss, dataset, device, optimiser)`: By default, iterates over `(input, output)` batches in `dataset` and returns the empirical risk based on `loss(estimator(input), output)`. Override this method only if the required computation cannot be expressed through `_loss()` or `_inputoutput()` alone.
- Inference: Extend existing functions (e.g., `sampleposterior`) or define new ones if needed. 
- Assessment: If your estimator returns point estimates via `estimate()` or posterior
  samples via `sampleposterior()`, it will work with `assess()` automatically — just
  add it to the relevant `Union` in `assess.jl`. Otherwise, implement a custom
  `assess()` method.

---

## Adding a New Approximate Distribution

The recommended workflow for developing a new approximate distribution is to first prototype interactively, and then consolidate into the package once things are working.

**Interactive prototyping:**
1. Once you've started Julia and loaded `NeuralEstimators`, define a custom struct subtyping `ApproximateDistribution`, for example, `struct MyDist <: ApproximateDistribution`. This struct will have fields storing anything needed to parameterise the distribution.
1. Import any functions you need to extend: `import NeuralEstimators: logdensity, sampleposterior`.
1. Extend the relevant functions interactively (see below). This allows you to rapidly iterate and test changes on the fly without restarting the session.
1. Test the distribution by plugging it into a `PosteriorEstimator` and training interactively to verify correctness.
1. Write a docstring with a self-contained example that can be copy-pasted and run.

**Consolidating into the package:**
1. Create `src/ApproximateDistributions/MyDist.jl` and move all of the above definitions into it.
1. Export `MyDist` in `NeuralEstimators.jl`.

**Which methods to extend:**
- `logdensity(q::MyDist, θ::AbstractMatrix, t::AbstractMatrix)`: The log-density of the approximate distribution `q` evaluated at parameters `θ`, given summary statistics `t`. Required for training.

- `sampleposterior(q::MyDist, t::AbstractMatrix, N::Integer)`: Draw `N` posterior samples from `q` given summary statistics `t`. Required for inference.
- `numdistributionalparams(q::MyDist)`: The number of distributional parameters; used to automatically construct the MLP that maps summary statistics to distributional parameters inside `PosteriorEstimator`.

---

## Adding a New Architecture

In general, we leave the construction of neural network architectures to the user: the package is intentionally agnostic to the specific architecture used, and most standard Flux layers and containers work out of the box. 

That said, reusable building blocks that are broadly applicable and not already available in Flux (e.g., `DeepSet`, `MLP`) can be added to `Architectures.jl`. Follow the Flux layer pattern (a struct with fields for parameters, and a callable method), and export the type and any associated constructor helpers from `NeuralEstimators.jl`.