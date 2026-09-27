# Tests for NeuralEstimators.jl

Run tests with:

```julia
Pkg.test("NeuralEstimators")
```
or
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

The suite consists of `general.jl` (utilities, data containers, components) and `backends.jl` (estimators across the Flux, Lux and SimpleChains backends, devices and AD types). To run only one of them, pass its name as a test argument:

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args = ["general"])'
```

## Checking code coverage locally

Coverage is reported on [Codecov](https://app.codecov.io/gh/msainsburydale/NeuralEstimators.jl) for every pull request. Before opening one, please check locally that the tests exercise any new code you have written.

1. Run the tests with coverage tracking (optionally restricted to one test file, as above):

   ```bash
   julia --project=. -e 'using Pkg; Pkg.test(coverage = true)'
   ```

   This writes a `*.cov` file next to each source file that was executed.

2. Summarise the coverage and write an LCOV file (CoverageTools.jl only needs to be installed once, e.g., in your global environment):

   ```julia
   using CoverageTools
   cov = vcat(process_folder("src"), process_folder("ext"))
   LCOV.writefile("lcov.info", cov)

   # Overall coverage
   covered, total = get_summary(cov)
   covered / total

   # Coverage of a single file
   get_summary(only(filter(fc -> fc.filename == "src/train.jl", cov)))
   ```

3. Inspect the uncovered lines. In VS Code, the [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) extension reads `lcov.info` and highlights covered and uncovered lines in the editor. Alternatively, `genhtml lcov.info -o coverage` (from [lcov](https://github.com/linux-test-project/lcov), e.g., `brew install lcov`) builds an HTML report in `coverage/` (open `coverage/index.html` in a web browser, e.g., by running `open coverage/index.html`).

4. Remove the coverage files when you are done (they are ignored by git, but otherwise accumulate across runs):

   ```bash
   find src ext test -name "*.jl.*.cov" -delete
   ```

Notes:
- Code in package extensions (`ext/`) is only covered if the packages that trigger the extension are loaded in the tests (e.g., the AdvancedHMC extension requires `using AdvancedHMC, ForwardDiff, LogDensityProblems`); add any new trigger packages to `test/Project.toml`.
- GPU-only code paths are not covered unless a functional GPU is available (the CI runners have none).
- Julia sometimes attributes coverage to lines such as the closing `end` of a loop inconsistently; small discrepancies of this kind can be ignored.
