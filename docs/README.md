# Instructions for contributing to the documentation

The [package documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) is built using [Documenter.jl](https://documenter.juliadocs.org/stable/). Source files for the documentation are located in [docs/src](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/src) and as docstrings within the codebase for each function or type provided by the package. 

### Workflow for contributing

1.	**Download the source code** by cloning the package repository from GitHub.
2.	**Edit the documentation** by modifying the relevant source files or docstrings.
3.	**Build the documentation locally** to preview your changes. To build the documentation locally, run the following command from the root folder:
```bash
julia --project=. docs/make.jl && julia -e 'using LiveServer; serve(dir="docs/build")'
```
4.	**Push changes** to the main branch. The updated documentation will be automatically built and deployed.

