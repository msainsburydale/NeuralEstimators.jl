push!(LOAD_PATH,"../src/")
using Documenter, NeuralEstimators

makedocs(
    sitename="NeuralEstimators.jl",
    # format = Documenter.LaTeX(),
    # format = Documenter.LaTeX(platform = "none"), # extracting the .tex file can be useful for bug fixing
    pages = [
           "index.md",
           "methodology.md",
           "Workflow" => [
                "workflow/overview.md",
                "workflow/examples.md",
                "workflow/advancedusage.md"
           ],
           "API" => [
                "API/core.md",
                "API/architectures.md",
                "API/approximatedistributions.md",
                "API/loss.md",
                "API/simulation.md",
                "API/utility.md",
                "API/index.md"
           ]
       ]
)

deploydocs(
  deps = nothing, make = nothing,
  repo = "github.com/msainsburydale/NeuralEstimators.jl.git",
  target = "build",
  branch = "gh-pages",
  devbranch = "main"
)
