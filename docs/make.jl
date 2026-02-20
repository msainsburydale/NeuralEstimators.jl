push!(LOAD_PATH, "../src/")
using Documenter, NeuralEstimators

# Install the packages required by the package extensions
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
# Pkg.add(["AlgebraOfGraphics", "CairoMakie"])
Pkg.instantiate()
using AlgebraOfGraphics, CairoMakie

makedocs(
    # modules = modules,
    sitename = "NeuralEstimators.jl",
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
