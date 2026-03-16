push!(LOAD_PATH, "../src/")
using Documenter, DocumenterVitepress, NeuralEstimators

# Install the packages required by the package extensions
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.add(["Makie"])
Pkg.instantiate()
using Makie

makedocs(
    sitename = "NeuralEstimators.jl",
    pages = [
        "Home" => "index.md",
        "Methodology" => "methodology.md",
        "Workflow overview" => "overview.md",
        "Examples" => [
            "examples/data_replicated.md",
            "examples/data_gridded.md",
            "examples/data_irregularspatial.md"
        ],
        "Advanced usage" => "examples/advancedusage.md",
        "API" => [
            "Parameters and data" => "API/parametersdata.md",
            "API/estimators.md",
            "API/training.md",
            "API/assessment.md",
            "API/inference.md",
            "API/architectures.md",
            "API/approximatedistributions.md",
            "API/lossfunctions.md",
            "API/miscellaneous.md",
            "API/index.md"
        ]
    ],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/msainsburydale/NeuralEstimators.jl",
        devbranch = "main",
        devurl = "dev")
)

deploydocs(
    deps = nothing, make = nothing,
    repo = "github.com/msainsburydale/NeuralEstimators.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main"
)
