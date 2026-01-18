module NeuralEstimatorsOptimExt

using NeuralEstimators
using Optim
import NeuralEstimators: _optimdensity

function _optimdensity(θ₀, logprior::Function, est)
    θ₀ = Float32.(θ₀)       # convert for efficiency and to avoid warnings

    objective(θ) = -first(logprior(θ) + est(Z, θ)) # closure that will be minimised

    # Gradient using reverse-mode automatic differentiation with Zygote
    # ∇objective(θ) = gradient(θ -> objective(θ), θ)[1]
    # θ̂ = Optim.optimize(objective, ∇objective, θ₀, Optim.LBFGS(); inplace = false) |> Optim.minimizer

    # Gradient using finite differences
    # θ̂ = Optim.optimize(objective, θ₀, Optim.LBFGS()) |> Optim.minimizer

    # Gradient-free NelderMead algorithm (find that this is most stable)
    θ̂ = Optim.optimize(objective, θ₀, Optim.NelderMead()) |> Optim.minimizer
end

end
