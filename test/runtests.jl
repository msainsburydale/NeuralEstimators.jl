# Run a subset of the test files with, e.g., Pkg.test(test_args = ["general"])
if isempty(ARGS) || "general" in ARGS
    include("general.jl")
end
if isempty(ARGS) || "backends" in ARGS
    include("backends.jl")
end
