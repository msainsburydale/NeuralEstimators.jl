push!(LOAD_PATH,"../src/")
using Documenter, NeuralEstimators

makedocs(
    sitename="NeuralEstimators.jl",
    pages = [
           "Home" => "index.md",
           "backgroundtheory.md",
           "Workflow" => [
                "Overview" => "workflow/overview.md",
                "Simple example" => "workflow/simple.md",
                "More complicated example" => "workflow/complex.md",
                "Advanced usage" => "workflow/advanced.md"
           ],
           "API" => [
                "Core functions" => "API/core.md",
                "Data simulation" => "API/simulation.md",
                "Utility functions" => "API/utility.md",
                "Index" => "API/index.md"
           ]
       ]
)




# Template for more complicated model:
# <!-- We describe the workflow with a simple estimation task: Estimating μ from N(μ, σ) data, where σ = 1 is known. Our first step is to define a `struct` with a field `θ`, along with any other information that will be needed for data simulation; in this case, the known standard deviation, σ.
# ```
# struct Parameters
# 	θ
# 	σ
# end
# ```
# Storing parameter information a `struct` is useful for storing intermediates objects needed for data simulation, such as Cholesky factors, and for implementing variants of on-the-fly and just-in-time simulation.
#
# The next step is to define an object (of any type) that can be used to sample parameters and pass on information to the data simulation function. Here, we define the prior distribution, Ω, of θ. We also store information that is needed for data simulation; in this case, the known standard deviation, σ.
# ```
# μ₀ = 0             # prior mean
# σ₀ = 0.5           # prior standard deviation
# Ω = Normal(μ₀, σ₀) # prior distribution
#
# ξ = (Ω = Ω, σ = 1)
# ```
#
# We then define a method `sample(Ω, K)` that returns K samples of θ as a p × K matrix, where p is the dimension of θ (in this case, p = 1). The return type should be an instance of the `struct` defined above.
# ```
# function sample(ξ, K::Integer)
# 	θ = rand(ξ.Ω, K)
# 	θ = reshape(θ, 1, :) # convert from column vector to p x K matrix
# 	Parameters(θ, ξ.σ)
# end
# ```
#
# Next, we implicitly define the statistical model by providing a method for `simulate()`, which defines data simulation conditional on θ. The method must take two arguments; an instance of the `struct` defined above and `m`, the sample size. There is some flexibility in the permitted type of `m` (e.g., `Integer`, `IntegerRange`, etc.), but `simulate()` must return a vector of (multi-dimensional) arrays, where each array is associated with one parameter vector.
# ```
# function simulate(params::Parameters, m::Integer)
# 	n = 1
# 	σ = params.σ
# 	μ = vec(params.θ)
# 	Z = [rand(Normal(μ, σ), n, 1, m) for μ ∈ θ] # extra 1 needed for consistency with Flux.jl model
# end
# ```
