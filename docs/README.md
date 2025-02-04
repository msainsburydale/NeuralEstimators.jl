# Instructions for contributing to the documentation

The [package documentation](https://msainsburydale.github.io/NeuralEstimators.jl/dev/) is built using [Documenter.jl](https://documenter.juliadocs.org/stable/). Source files for the documentation are located in [docs/src](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/docs/src) and as the docstrings for each function or type defined in [src](https://github.com/msainsburydale/NeuralEstimators.jl/tree/main/src). 

### Workflow for contributing

1.	**Download the source code**: Clone the package repository from GitHub, for instance, by running the following command:
```bash
git clone https://github.com/msainsburydale/NeuralEstimators.jl.git
```
2.	**Edit the documentation**: Modify the relevant source files or docstrings.
3.	**Build the documentation locally**: Preview your changes by building the documentation locally. This can be done by installing the Julia packages `Documenter` and `LiveServer`, and then running the following command from the root folder:
```bash
julia --project=. docs/make.jl && julia -e 'using LiveServer; serve(dir="docs/build")'
```
4.	**Push changes**: Once satisfied with your changes, `git commit` and `git push` to the main branch. The updated documentation will be automatically built and deployed.

