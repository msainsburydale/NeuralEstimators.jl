push!(LOAD_PATH,"../src/")
using Documenter, NeuralEstimators

makedocs(
    sitename="NeuralEstimators.jl",
    pages = [
           "Home" => "index.md",
           "framework.md",
           "Workflow" => [
                "Examples" => "workflow/examples.md",
                "Advanced usage" => "workflow/advancedusage.md"
           ],
           "API" => [
                "Core functions" => "API/core.md",
                "Model-specific functions" => "API/simulation.md",
                "Utility functions" => "API/utility.md",
                "Index" => "API/index.md"
           ]
       ]
)
