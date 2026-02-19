@doc raw"""
	PiecewiseEstimator <: NeuralEstimator
	PiecewiseEstimator(estimators::Vector{BayesEstimator}, changepoints::Vector{Integer})
Creates a piecewise estimator
([Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec. 2.2.2)
from a collection of neural Bayes `estimators` and sample-size `changepoints`.

This allows different estimators to be applied to different ranges of sample sizes. For instance, you may wish to use one estimator for small samples and another for larger ones. Given changepoints $m_1 < m_2 < \dots < m_{l-1}$, the piecewise estimator selects from $l$ trained estimators based on the observed sample size $m$ as follows:
```math
\hat{\boldsymbol{\theta}}(\boldsymbol{Z})
=
\begin{cases}
\hat{\boldsymbol{\theta}}_1(\boldsymbol{Z}) & m \leq m_1,\\
\hat{\boldsymbol{\theta}}_2(\boldsymbol{Z}) & m_1 < m \leq m_2,\\
\quad \vdots \\
\hat{\boldsymbol{\theta}}_l(\boldsymbol{Z}) & m > m_{l-1}.
\end{cases}
```
where $\hat{\boldsymbol{\theta}}_1(\cdot)$ is a neural Bayes estimator trained to be near-optimal over the range of sample sizes in which it is applied. 

Although this strategy requires training multiple neural networks, it is computationally efficient in practice when combined with pre-training (see [Sainsbury-Dale at al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec 2.3.3), which can be automated using [`trainmultiple()`](@ref). 

# Examples
```julia
using NeuralEstimators, Flux

n = 2    # bivariate data
d = 3    # dimension of parameter vector 
w = 128  # width of each hidden layer

# Small-sample estimator
ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, d));
θ̂₁ = PointEstimator(DeepSet(ψ₁, ϕ₁))

# Large-sample estimator
ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, d));
θ̂₂ = PointEstimator(DeepSet(ψ₂, ϕ₂))

# Piecewise estimator with changepoint m=30
θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], 30)

# Apply the (untrained) piecewise estimator to data
Z = [rand(n, m) for m ∈ (10, 50)]
estimate(θ̂, Z)
```
"""
struct PiecewiseEstimator{E, C} <: NeuralEstimator
    estimators::E
    changepoints::C
    function PiecewiseEstimator(estimators, changepoints)
        if isa(changepoints, Number)
            changepoints = [changepoints]
        end
        @assert all(isinteger.(changepoints)) "`changepoints` should contain integers"
        if length(changepoints) != length(estimators) - 1
            error("The length of `changepoints` should be one fewer than the number of `estimators`")
        elseif !issorted(changepoints)
            error("`changepoints` should be in ascending order")
        else
            E = typeof(estimators)
            C = typeof(changepoints)
            new{E, C}(estimators, changepoints)
        end
    end
end
function (estimator::PiecewiseEstimator)(Z)
    changepoints = [estimator.changepoints..., Inf]
    m = numberreplicates(Z)
    θ̂ = map(eachindex(Z)) do i
        # find which estimator to use and then apply it
        mᵢ = m[i]
        j = findfirst(mᵢ .<= changepoints)
        estimator.estimators[j](Z[[i]])
    end
    return stackarrays(θ̂)
end
Base.show(io::IO, estimator::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(estimator.estimators)) estimators and sample size change-points: $(estimator.changepoints)")
