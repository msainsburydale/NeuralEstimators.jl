"""
	Ensemble <: NeuralEstimator
	Ensemble(estimators)
	Ensemble(architecture::Function, J::Integer)
	(ensemble::Ensemble)(Z; aggr = mean)
Defines an ensemble of `estimators` which, when applied to data `Z`, returns the mean (or another summary defined by `aggr`) of the individual estimates (see, e.g., [Sainsbury-Dale et al., 2025, Sec. S5](https://doi.org/10.48550/arXiv.2501.04330)).

The ensemble can be initialised with a collection of trained `estimators` and then
applied immediately to observed data. Alternatively, the ensemble can be
initialised with a collection of untrained `estimators`
(or a function defining the architecture of each estimator, and the number of estimators in the ensemble),
trained with `train()`, and then applied to observed data. In the latter case, where the ensemble is trained directly,
if `savepath` is specified both the ensemble and component estimators will be saved.

Note that `train()` currently acts sequentially on the component estimators, using the `Adam` optimiser.

The ensemble components can be accessed by indexing the ensemble; the number of component estimators can be obtained using `length()`.

See also [`Parallel`](https://fluxml.ai/Flux.jl/stable/reference/models/layers/#Flux.Parallel), which can be used to mimic ensemble methods with an appropriately chosen `connection`. 

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|θ ~ N(θ, 1) with θ ~ N(0, 1)
d = 1     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
m = 30    # number of independent replicates in each data set
sampler(K) = randn32(d, K)
simulator(θ, m) = [μ .+ randn32(n, m) for μ ∈ eachcol(θ)]

# Neural-network architecture of each ensemble component
function architecture()
	ψ = Chain(Dense(n, 64, relu), Dense(64, 64, relu))
	ϕ = Chain(Dense(64, 64, relu), Dense(64, d))
	network = DeepSet(ψ, ϕ)
	PointEstimator(network)
end

# Initialise ensemble with three component estimators 
ensemble = Ensemble(architecture, 3)
ensemble[1]      # access component estimators by indexing
ensemble[1:2]    # indexing with an iterable collection returns the corresponding ensemble 
length(ensemble) # number of component estimators

# Training
ensemble = train(ensemble, sampler, simulator, m = m, epochs = 5)

# Assessment
θ = sampler(1000)
Z = simulator(θ, m)
assessment = assess(ensemble, θ, Z)
rmse(assessment)

# Apply to data
ensemble(Z)
```
"""
struct Ensemble{T <: NeuralEstimator} <: NeuralEstimator
    estimators::Vector{T}
end
Ensemble(architecture::Function, J::Integer) = Ensemble([architecture() for j = 1:J])

function (ensemble::Ensemble)(Z; aggr = mean)
    # Collect each estimator’s output
    θ̂s = [estimator(Z) for estimator in ensemble.estimators]

    # Stack into 3D array (d × n × m) where m = number of estimators
    θ̂ = stackarrays(θ̂s, merge = false)

    # Aggregate elementwise
    if aggr === mean
        θ̂ = mean(θ̂; dims = 3)
    else
        #NB mapslices doesn't work with Zygote, so use mean as the default
        θ̂ = mapslices(aggr, cpu(θ̂); dims = 3)
    end

    return dropdims(θ̂; dims = 3)
end

Base.getindex(e::Ensemble, i::Integer) = e.estimators[i]
Base.getindex(e::Ensemble, indices::AbstractVector{<:Integer}) = Ensemble(e.estimators[indices])
Base.getindex(e::Ensemble, indices::UnitRange{<:Integer}) = Ensemble(e.estimators[indices])
Base.length(e::Ensemble) = length(e.estimators)
Base.eachindex(e::Ensemble) = eachindex(e.estimators)
Base.show(io::IO, ensemble::Ensemble) = print(io, "\nEnsemble with $(length(ensemble.estimators)) component estimators")


function train(ensemble::Ensemble, args...; kwargs...)
    kwargs = (; kwargs...)
    savepath = haskey(kwargs, :savepath) ? kwargs.savepath : nothing
    verbose = haskey(kwargs, :verbose) ? kwargs.verbose : true
    optimiser = haskey(kwargs, :optimiser) ? kwargs.optimiser : nothing
    estimators = map(enumerate(ensemble.estimators)) do (i, estimator)
        verbose && @info "Training estimator $i of $(length(ensemble))"
        if !isnothing(savepath)
            kwargs = merge(kwargs, (savepath = joinpath(savepath, "estimator$i"),))
        end
        if !isnothing(optimiser) # catch errors caused by constructing the optimiser from the Ensemble object
            lr = try
                findlr(optimiser)
            catch
                ;
                5e-4
            end
            kwargs = merge(kwargs, (optimiser = Flux.setup(Adam(lr), estimator),))
        end
        train(estimator, args...; kwargs...)
    end
    ensemble = Ensemble(estimators)

    if !isnothing(savepath)
        if !ispath(savepath)
            mkpath(savepath)
        end
        model_state = Flux.state(cpu(ensemble))
        @save joinpath(savepath, "ensemble.bson") model_state
    end

    return ensemble
end
