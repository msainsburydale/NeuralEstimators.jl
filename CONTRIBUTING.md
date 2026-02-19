# Contributing to NeuralEstimators

Thank you for your interest in contributing to **NeuralEstimators**! This document describes the code structure, development workflow, and how to submit contributions. 

---

## Table of Contents

1. [Code Structure](#code-structure)
1. [Key Abstractions](#key-abstractions)
1. [Development Workflow](#development-workflow)
1. [Coding Conventions](#coding-conventions)
1. [Adding a New Estimator](#adding-a-new-estimator)
1. [Adding a New Architecture](#adding-a-new-architecture)
1. [Adding a New Approximate Distribution](#adding-a-new-approximate-distribution)
1. [Submitting a Contribution](#submitting-a-contribution)

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

New `.jl` files in `src/` must be explicitly `include`d in `NeuralEstimators.jl`. Files inside `src/Estimators/` are auto-included in alphabetical order (except `Ensemble.jl`, which is always loaded first). Any new file added to that subfolder will therefore be picked up automatically, but its public symbols should still be exported from `NeuralEstimators.jl`.

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

### `ParameterConfigurations`

An abstract type (defined in `Parameters.jl`) for user-defined structs that hold parameter matrices and any precomputed intermediate quantities needed for simulation. The only required field is `θ`, a `d × K` matrix of parameter vectors. Implement `subsetparameters` if your type stores additional fields that should be subsetted consistently.

### `ApproximateDistribution`

An abstract type (defined in `ApproximateDistributions.jl`) used by `PosteriorEstimator`. 

### `train` and `assess`

`train` (in `train.jl`) is the universal training interface. It accepts any `NeuralEstimator` and dispatches on the estimator type to select the appropriate loss. `assess` (in `assess.jl`) evaluates a trained estimator and returns an `Assessment` object that supports downstream computations such as `risk`, `bias`, `rmse`, and `coverage`.

---

## Development Workflow

1. **Fork and clone** the repository.
2. **Activate the package environment** from the repository root:
   ```julia
   using Pkg; Pkg.activate("."); Pkg.instantiate()
   ```
3. **Make your changes** in the appropriate source file (see the structure above).
4. **Export new public symbols** in `NeuralEstimators.jl`.
5. **Write or update tests** in the `test/` directory.
6. **Run tests** with:
   ```julia
   Pkg.test("NeuralEstimators")
   ```
   or
   ```bash
   julia --project=. -e "using Pkg; Pkg.test()"
   ```
7. **Open a pull request** describing your changes (see [Submitting a Contribution](#submitting-a-contribution)).

---

## Coding Conventions

- **Style:** Follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/).
- **Types:** Use `Float32` (via `randn32`, `rand32`, etc.) throughout, as Flux defaults to single precision. Avoid introducing unnecessary `Float64` allocations in hot paths.
- **Docstrings:** Every exported symbol must have a docstring in the standard Julia format. Include at least one runnable `# Examples` block. For mathematical content, use `@doc raw"""..."""` to allow unescaped LaTeX.
- **Deprecations:** When renaming or removing a public symbol, add a deprecated alias in `deprecated.jl` and do not delete the old export immediately.
- **GPU compatibility:** When relevant, try to make new code agnostic to the device (CPU, GPU).

---

## Adding a New Estimator

1. Create a file `src/Estimators/MyEstimator.jl`.
1. Define a struct that subtypes `NeuralEstimator` (or `BayesEstimator` if it targets a Bayes risk):
   ```julia
   struct MyEstimator{N} <: NeuralEstimator
       ::N
   end
   ```
1. If the estimator requires a custom training loss, add a method to `train` in `train.jl` (or extend the loss dispatch there).
1. Export the type in `NeuralEstimators.jl`.
1. Add an `assess` method in `assess.jl` if the estimator produces outputs that are not scalar point estimates (e.g., intervals, posterior samples).
1. Write a docstring with a self-contained example that can be copy-pasted and run.

---

## Adding a New Architecture

New neural network building blocks belong in `Architectures.jl`. Follow the Flux layer pattern (a struct with fields for parameters, and a callable method). Export the type and any associated constructor helpers from `NeuralEstimators.jl`.

---

## Adding a New Approximate Distribution

1. Create a file `src/ApproximateDistributions/MyDist.jl`.
1. Define a struct that subtypes `ApproximateDistribution`:
   ```julia
   struct MyDist{A} <: ApproximateDistribution
       ::A
   end
   ```
1. Implement the methods:
   ```julia
   logdensity(q::MyDist, θ::AbstractMatrix, tz::AbstractMatrix)  # returns 1 × K matrix
   sampleposterior(q::MyDist, tz::AbstractMatrix, N::Integer)    # returns Vector of d × N matrices
   ```
1. Optionally implement `numdistributionalparams(q::MyDist)` if the number of distributional parameters can be determined statically.
1. Export the type in `NeuralEstimators.jl` and add a docstring with a working example.

Note that, analogously to `Estimators/`, files in `ApproximateDistributions/` are auto-included alphabetically, so no manual `include` is needed — just ensure the new file is in the folder and the symbol is exported.

---

## Submitting a Contribution

- (Optional) Open a **GitHub Issue** before starting significant work, so the approach can be discussed.
- Write a PR description explaining the motivation, the change, and any limitations.
- Ensure all tests pass and that new functionality is documented, tested, and exported.
- If your contribution adds a dependency, discuss it first — new dependencies are added conservatively.

We appreciate contributions of all sizes, from fixing typos in docstrings to implementing new estimator types. Thank you for helping improve NeuralEstimators!