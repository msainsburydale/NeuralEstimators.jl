"""
    train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {P, T}
    train(estimator, θ_train::P, θ_val::P, simulator::Function; ...) where {P, T}
    train(estimator, sampler::Function, simulator::Function; ...)
Trains a neural `estimator`.

The methods cater for different variants of "on-the-fly" simulation.
Specifically, a `sampler` can be provided to continuously sample new parameter
vectors from the prior, and a `simulator` can be provided to continuously
simulate new data conditional on the parameters. If
provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data
(`Z_train` and `Z_val`), they will be held fixed during training.

The validation parameters and data are always held fixed.

The trained estimator is always returned on the CPU. 

# Keyword arguments common to all methods:
- `loss = mae` (applicable only to [`PointEstimator`](@ref)): loss function used to train the neural network. 
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
- `stopping_epochs = 5`: cease training if the risk does not improve in this number of epochs.
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
- `optimiser = Optimisers.setup(Adam(5e-4), estimator)`: any [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) optimisation rule for updating the neural-network parameters. When the training data and/or parameters are held fixed, one may wish to employ regularisation to prevent overfitting; for example, `optimiser = Optimisers.setup(OptimiserChain(WeightDecay(1e-4), Adam()), estimator)`, which corresponds to L₂ regularisation with penalty coefficient λ=10⁻⁴. 
- `lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule}`: defines the learning-rate schedule for adaptively changing the learning rate during training. Accepts either a [ParameterSchedulers.jl](https://fluxml.ai/ParameterSchedulers.jl/dev/) object or `nothing` for a fixed learning rate. By default, it uses [`CosAnneal`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/cyclic/#ParameterSchedulers.CosAnneal) with a maximum set to the initial learning rate from `optimiser`, a minimum of zero, and a period equal to the number of epochs. The learning rate is updated at the end of each epoch. 
- `freeze_summary_network = false`: if `true` and the estimator has a `summary_network` field, freezes the summary network parameters during training (i.e., only the inference network is updated). In this case, the summary statistics for a given instance of simulated data are computed only once, giving a significant speedup. This is useful for transfer learning, where a pretrained summary network is held fixed while a new inference network is trained for a different model or estimator type.
- `use_gpu = true`: flag indicating whether to use a GPU if one is available.
- `adtype::AbstractADType = AutoZygote()`: the automatic differentiation backend used to compute gradients during training. The default uses [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/). Alternatively, `AutoEnzyme()` can be used to enable [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/), which can be faster and more memory efficient, and supports mutation and scalar indexing (requires `using Enzyme`).
- `savepath::Union{Nothing, String} = tempdir()`: path to save information generated during training. If `nothing`, nothing is saved. Otherwise, the following files are always saved to both `savepath` and `tempdir()` (the latter for convenient within-session access via [`loadrisk`](@ref), [`plotrisk`](@ref), and [`loadoptimiser`](@ref)):
  - `loss_per_epoch.csv`: training and validation risk at each epoch, in the first and second columns respectively.
  - `best_optimiser.bson`: optimiser state corresponding to the best validation risk.
  - `final_optimiser.bson`: optimiser state at the final epoch.

  If additionally `savepath != tempdir()`, the following files are also saved to `savepath`:
  - `best_network.bson`: neural-network parameters corresponding to the best validation risk.
  - `final_network.bson`: neural-network parameters at the final epoch.
  - `train_time.csv`: total training time in seconds.
- `risk_history::Union{Nothing, Matrix} = nothing`: a matrix with two columns containing the training and validation risk from a previous call to `train()`, used to initialise the risk history when continuing training. Can be loaded from a previous call to `train` using [`loadrisk`](@ref).
- `verbose = true`: flag indicating whether information, including empirical risk values and timings, should be printed to the console during training.

# Keyword arguments common to `train(estimator, sampler, simulator)` and `train(estimator, θ_train, θ_val, simulator)`:
- `simulator_args = ()`: positional arguments passed to the simulator as `simulator(θ, simulator_args...)`.
- `simulator_kwargs::NamedTuple = (;)`: keyword arguments passed to the simulator as `simulator(...; simulator_kwargs...)`.
- `epochs_per_Z_refresh = 1`: the number of passes to make through the training set before the training data are refreshed.
- `simulate_just_in_time = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

# Keyword arguments unique to `train(estimator, sampler, simulator)`:
- `sampler_args = ()`: positional arguments passed to the parameter sampler as `sampler(K, sampler_args...)`.
- `sampler_kwargs::NamedTuple = (;)`: keyword arguments passed to the parameter sampler as `sampler(...; sampler_kwargs...)`.
- `K = 10000`: number of parameter vectors in the training set.
- `K_val = K ÷ 2` number of parameter vectors in the validation set.
- `epochs_per_θ_refresh = 1`: the number of passes to make through the training set before the training parameters are refreshed. Must be a multiple of `epochs_per_Z_refresh`. Can also be provided as `epochs_per_theta_refresh`.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²), priors μ ~ U(0, 1) and σ ~ U(0, 1)
d, m = 2, 50 # number of unknown parameters and number replicates in each data set
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Summary network
num_summaries = 3d
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator
estimator = PointEstimator(summary_network, d; num_summaries = num_summaries)

# Training: simulation on-the-fly
K = 1000
estimator  = train(estimator, sampler, simulator, K = K)

# Plot the risk history (using any plotting backend)
using Plots
unicodeplots()
plotrisk()

# Training: simulation on-the-fly with fixed parameters 
θ_train = sampler(K)
θ_val   = sampler(K)
estimator = train(estimator, θ_train, θ_val, simulator, optimiser = loadoptimiser(), risk_history = loadrisk(), freeze_summary_network = true)

# Training: fixed parameters and fixed data
Z_train   = simulator(θ_train)
Z_val     = simulator(θ_val)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val, optimiser = loadoptimiser(), risk_history = loadrisk(), freeze_summary_network = true)
```
"""
function train end

# TODO with summary statistics only, might be faster to always use the CPU?

# Computes the risk for a general loss function in a memory-safe manner, updating the
# neural-network parameters using stochastic gradient descent if optimiser is specified
# function _risk(estimator, loss, data_loader, device, optimiser = nothing, adtype = AutoZygote())
#     sum_loss = 0.0f0
#     K = 0
#     for (input, output) in data_loader
#         input, output = input |> device, output |> device
#         if !isnothing(optimiser)
#             # NB computing and storing the loss in this way is computationally efficient, but it means that
#             # the final training risk that we report for each epoch is slightly inaccurate
#             # (since the neural-network parameters are updated after each batch)
#             ls, ∇ = Flux.withgradient(estimator -> loss(estimator(input), output), adtype, estimator)
#             Optimisers.update!(optimiser, estimator, ∇[1])
#         else
#             ls = loss(estimator(input), output)
#         end
#         # Convert average loss to a sum and add to total
#         num_obs = numobs(input)
#         sum_loss += ls * num_obs
#         K += num_obs
#     end

#     return cpu(sum_loss/K)
# end

function _risk(estimator, loss, data_loader, device, optimiser = nothing, adtype = AutoZygote())
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data_loader
        input, output = input |> device, output |> device
        if !isnothing(optimiser)
            # NB computing and storing the loss in this way is computationally efficient, but it means that
            # the final training risk that we report for each epoch is slightly inaccurate
            # (since the neural-network parameters are updated after each batch)
            ls, ∇ = _riskvalue_and_gradient(estimator, loss, input, output, adtype)
            Optimisers.update!(optimiser, estimator, ∇)
        else
            ls = _riskvalue(estimator, loss, input, output)
        end
        # Convert average loss to a sum and add to total
        num_obs = numobs(input)
        sum_loss += ls * num_obs
        K += num_obs
    end

    return cpu(sum_loss/K)
end

function _riskvalue_and_gradient end
_riskvalue(estimator, loss, input, output) = loss(estimator(input), output)

# For generic estimators, use the user-specified loss function
_loss(estimator, loss) = loss

# Constructs inputs and outputs (default simulated data and corresponding true parameters, respectively)
_inputoutput(estimator, Z, θ) = (Z, θ)

function _dataloader(estimator, Z, θ, batchsize)
    data = _inputoutput(estimator, Z, _stripnames(_extractθ(θ)))
    _DataLoader(data, batchsize)
end

# Thin wrapper around DataLoader with sensible training defaults
# NB: redirect_stderr suppresses batchsize warning from DataLoader
function _DataLoader(data, batchsize::Integer; shuffle = true, partial = false)
    oldstd = stdout
    redirect_stderr(devnull)
    data_loader = DataLoader(f32(data), batchsize = batchsize, shuffle = shuffle, partial = partial)
    redirect_stderr(oldstd)
    return data_loader
end

function _findlr(opt)
    if opt isa Optimisers.Leaf
        rule = opt.rule
        if hasproperty(rule, :eta)
            return rule.eta
        elseif rule isa OptimiserChain
            for subrule in rule.opts
                if hasproperty(subrule, :eta)
                    return subrule.eta
                end
            end
        end
    elseif opt isa AbstractArray || opt isa Tuple
        for subopt in opt
            eta = _findlr(subopt)
            if !isnothing(eta)
                return eta
            end
        end
    elseif opt isa NamedTuple
        for (_, subopt) in pairs(opt)
            eta = _findlr(subopt)
            if !isnothing(eta)
                return eta
            end
        end
    end
    return nothing
end

function _saveloss(loss_per_epoch, savepath)
    isnothing(savepath) && return
    for path in unique([savepath, tempdir()])
        !ispath(path) && mkpath(path)
        CSV.write(joinpath(path, "loss_per_epoch.csv"), Tables.table(loss_per_epoch), header = false)
    end
end

function _saveoptimiser(optimiser, savepath; best::Bool = true)
    isnothing(savepath) && return
    file = best ? "best_optimiser.bson" : "final_optimiser.bson"
    for path in unique([savepath, tempdir()])
        !ispath(path) && mkpath(path)
        optimiser = cpu(optimiser)
        @save joinpath(path, file) optimiser
    end
end


function _saveestimator end

function _savefinal(estimator, optimiser, train_time, savepath, verbose)
    _saveestimator(estimator, savepath; best = false)
    _saveoptimiser(optimiser, savepath; best = false)
    _forcegc(verbose)
    verbose && println("Finished training in $(train_time) seconds")
    if !isnothing(savepath) && savepath != tempdir()
        CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
    end
end

function _checkargs(batchsize, epochs, stopping_epochs, risk_history)
    @assert batchsize > 0
    @assert epochs > 0
    @assert stopping_epochs > 0
    @assert isnothing(risk_history) || size(risk_history, 2) == 2 "`risk_history` must have exactly two columns (training risk and validation risk)"
end

"""
    loadoptimiser(savepath::String = tempdir(); best::Bool = true)
Loads the optimiser saved during the most recent call to `train()`.

By default, loads from the temporary directory used during the current session.
If a `savepath` was provided to `train()`, that path can be passed here to
reload the optimiser from a previous session. 

By default, loads the optimiser corresponding to the best network (as measured
by validation loss). Set `best = false` to load the optimiser from the final
epoch instead, which can be useful for resuming training from exactly where it
left off.

The returned optimizer can be passed directly to `train()` via the `optimiser` 
keyword argument to resume training with the correct optimiser state.
"""
function loadoptimiser(savepath::String = tempdir(); best::Bool = true)
    file = best ? "best_optimiser.bson" : "final_optimiser.bson"
    path = joinpath(savepath, file)
    @assert isfile(path) "No optimiser state found at $(path). Please ensure that `train()` has been called, or provide the correct `savepath`."
    return load(path, @__MODULE__)[:optimiser]
end

"""
	loadrisk(savepath::String = tempdir())
Loads the training and validation risk history saved during the most recent call
to `train()`. By default, loads from the temporary directory used during the
current session. If a `savepath` was provided to `train()`, that path can be
passed here to reload risk history from a previous session.

Returns a matrix with two columns: training risk (column 1) and validation risk (column 2).

See also [`plotrisk`](@ref).
"""
function loadrisk(savepath::String = tempdir())
    path = joinpath(savepath, "loss_per_epoch.csv")
    @assert isfile(path) "No risk history found at $(path). Please ensure that `train()` has been called, or provide the correct `savepath`."
    df = CSV.read(path, DataFrame, header = false)
    return Matrix{Float32}(df)
end

"""
	plotrisk(savepath::String = tempdir())
Plots the training and validation risk as a function of epoch, loaded from
`savepath`. By default, loads from the temporary directory used during the
current session. If a `savepath` was provided to `train()`, that path can be
passed here to plot risk history from a previous session.

Requires a plotting package (e.g., `using Plots`) to be loaded.

See also [`loadrisk`](@ref).
"""
plotrisk(savepath::String = tempdir()) = _plotrisk(savepath)
_plotrisk(savepath) = error("plotrisk requires a plotting package to be loaded. Please load a supported plotting package (e.g., `using Plots`) and try again.")
# Note that defining a generic function here and defining the method in the extension also works,
# but the error that is thrown when a plotting package is not loaded is not particularly informative.