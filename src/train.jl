#TODO Remove adtype argument from _risk once we've sorted out Reactant compilation
#TODO clean up saving/loading... think we want to save the best parameters and optimisers separately:
# optimizer.bson: optimizer rule + optimizer state (continued training)
# parameters.bson: neural-network parameters (and states) (continued training + loading in different session)
"""
    train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {P, T}
    train(estimator, θ_train::P, θ_val::P, simulator; ...) where P
    train(estimator, sampler, simulator; ...)
Trains a neural `estimator`.

The methods cater for different variants of "on-the-fly" simulation.
Specifically, a callable `sampler` can be provided to continuously sample new parameters, 
and a callable `simulator` can be provided to continuously simulate new data conditional on the parameters. If
provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data
(`Z_train` and `Z_val`), they will be held fixed during training.

The validation parameters and data are always held fixed.

The trained estimator is always returned on the CPU. 

# Keyword arguments common to all methods:
- `loss = mae` (applicable only to [`PointEstimator`](@ref)): loss function used to train the neural network. 
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
- `stopping_epochs = 5`: cease training if the risk does not improve in this number of epochs.
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
- `optimiser::Optimisers.AbstractRule = Adam(5e-4)`: any [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) optimisation rule for updating the neural-network parameters. When the training data or parameters are fixed, one may wish to use regularisation to help prevent overfitting; see [Regularisation](@ref).
- `lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule}`: defines the learning-rate schedule for adaptively changing the learning rate during training. Accepts either a [ParameterSchedulers.jl](https://fluxml.ai/ParameterSchedulers.jl/dev/) object or `nothing` for a fixed learning rate. By default, it uses [`CosAnneal`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/cyclic/#ParameterSchedulers.CosAnneal) with a maximum set to the initial learning rate from `optimiser`, a minimum of zero, and a period equal to the number of epochs. The learning rate is updated at the end of each epoch. 
- `freeze_summary_network = false`: if `true` and the estimator has a `summary_network` field, freezes the summary network parameters during training (i.e., only the inference network is updated). In this case, the summary statistics for a given instance of simulated data are computed only once, giving a significant speedup. This is useful for transfer learning, where a pretrained summary network is held fixed while a new inference network is trained for a different model or estimator type.
- `device = nothing`: the device used for computation, e.g., `cpu_device()`, `gpu_device()`, or `reactant_device()` (the latter requires Lux.jl). If `nothing`, the device is inferred from `use_gpu`. Takes priority over `use_gpu`.
- `use_gpu::Bool = true`: flag indicating whether to use a GPU if one is available. Ignored if `device` is provided.
- `adtype::AbstractADType = AutoZygote()`: the automatic differentiation backend used to compute gradients during training. The default uses [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/). Alternatively, `AutoEnzyme()` can be used to enable [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/), which can be faster and more memory efficient, and supports mutation and scalar indexing (requires `using Enzyme`).
- `savepath::Union{Nothing, String} = tempdir()`: path to save information generated during training. Saving is disabled if `savepath = nothing`. Otherwise, the following files are always saved to both `savepath` and `tempdir()`:
  - `loss_per_epoch.csv`: training and validation risk at each epoch, in the first and second columns respectively.
  - `best_optimizer.bson`: optimiser and optimiser state corresponding to the best validation risk.
  - `final_optimizer.bson`: optimiser and optimiser state at the final epoch.

  If additionally `savepath != tempdir()`, the following files are also saved to `savepath`:
  - `best_trainstate.bson`: neural-network parameters, optimiser, and optimiser state corresponding to the best validation risk.
  - `final_trainstate.bson`: neural-network parameters, optimiser, and optimiser state at the final epoch.
  - `train_time.csv`: total training time in seconds.
- `risk_history::Union{Nothing, Matrix} = nothing`: a matrix with two columns containing the training and validation risk from a previous call to `train()`, used to initialise the risk history when continuing training. Can be loaded from a previous call to `train` using [`loadrisk`](@ref).
- `verbose = true`: flag indicating whether information, including empirical risk values and timings, should be printed to the console during training.

# Keyword arguments common to `train(estimator, sampler, simulator)` and `train(estimator, θ_train, θ_val, simulator)`:
- `simulator_args = ()`: positional arguments passed to `simulator`.
- `simulator_kwargs::NamedTuple = (;)`: keyword arguments passed to `simulator`.
- `epochs_per_Z_refresh = 1`: the number of passes to make through the training set before the training data are refreshed.
- `simulate_just_in_time = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

# Keyword arguments unique to `train(estimator, sampler, simulator)`:
- `sampler_args = ()`: positional arguments passed to `sampler`.
- `sampler_kwargs::NamedTuple = (;)`: keyword arguments passed to `sampler`.
- `K = 10000`: number of parameter vectors in the training set.
- `K_val = K ÷ 2` number of parameter vectors in the validation set.
- `epochs_per_θ_refresh = 1`: the number of passes to make through the training set before the training parameters are refreshed. Must be a multiple of `epochs_per_Z_refresh`.

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

function getestimator end
function _construct_train_state end
function _risk end
function _train_step end

function train(estimator::NeuralEstimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; optimiser::Optimisers.AbstractRule = Adam(5e-4), kwargs...) where {P, T}
    trainstate = _construct_train_state(estimator, optimiser)
    trainstate = train(trainstate, θ_train, θ_val, Z_train, Z_val; kwargs...)
    getestimator(trainstate)
end

function train(estimator::NeuralEstimator, θ_train::P, θ_val::P, simulator; optimiser::Optimisers.AbstractRule = Adam(5e-4), kwargs...) where {P}
    trainstate = _construct_train_state(estimator, optimiser)
    trainstate = train(trainstate, θ_train, θ_val, simulator; kwargs...)
    getestimator(trainstate)
end

function train(estimator::NeuralEstimator, sampler, simulator; optimiser::Optimisers.AbstractRule = Adam(5e-4), kwargs...)
    trainstate = _construct_train_state(estimator, optimiser)
    trainstate = train(trainstate, sampler, simulator; kwargs...)
    getestimator(trainstate)
end

#TODO with frozen summary networks, might be faster to always use the CPU once the summary statistics have been computed
_freeze_summary_network!(trainstate) = Optimisers.freeze!(trainstate.optimizer_state.summary_network)
_thaw!(trainstate) = Optimisers.thaw!(trainstate.optimizer_state)

_trainstate_to_device(trainstate, device) = device(trainstate)

function _resolve_adtype(trainstate, device, adtype, verbose = true)

    # Hard errors
    if trainstate isa FluxTrainState && device isa ReactantDevice
        error("reactant_device() is not supported with Flux; switch to Lux to use Reactant/XLA.")
    end

    if isnothing(adtype) # Set default adtype if not provided
        adtype = if device isa ReactantDevice
            AutoReactant()
        elseif trainstate isa FluxTrainState || device isa CUDADevice
            AutoZygote()
        else
            AutoEnzyme()  # Lux + CPU
        end
    else # Soft adjustments
        if device isa ReactantDevice && !(adtype isa AutoReactant)
            @info "Setting adtype = AutoReactant() since device is a ReactantDevice."
            adtype = AutoReactant()
        elseif adtype isa AutoReactant && !(device isa ReactantDevice)
            adtype = trainstate isa FluxTrainState || device isa CUDADevice ? AutoZygote() : AutoEnzyme()
            @info "Setting adtype = $(nameof(typeof(adtype)))() since AutoReactant requires reactant_device(), but device = $(nameof(typeof(device)))() was provided."
        elseif device isa CUDADevice && adtype isa AutoEnzyme
            @info "Setting adtype = AutoZygote() since AutoEnzyme is not yet fully supported with gpu_device()."
            if trainstate isa FluxTrainState
                @info "For improved performance, consider using Lux.jl + Reactant.jl."
            else
                @info "For improved performance, consider using Reactant.jl: run Reactant.set_default_backend(\"gpu\"), then set device = reactant_device() and adtype = AutoReactant()."
            end
            adtype = AutoZygote()
        end
    end

    verbose && @info "Automatic differentiation: $(nameof(typeof(adtype)))"

    return adtype
end

function train(trainstate, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
    batchsize::Integer = 32,
    epochs::Integer = 100,
    loss = mae,
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    device = nothing,
    use_gpu::Bool = true,
    adtype::Union{AbstractADType, Nothing} = nothing,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = ParameterSchedulers.CosAnneal(_findlr(trainstate), zero(_findlr(trainstate)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false
) where {P, T}

    # Determine device
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = verbose)

    # Determine adtype and check deep-learning backend + adtype + device are compatible
    adtype = _resolve_adtype(trainstate, device, adtype, verbose)

    # Move trainstate to device and extract the current estimator
    trainstate = _trainstate_to_device(trainstate, device)
    estimator = getestimator(trainstate)

    if estimator isa PosteriorEstimator && estimator.q isa Gaussian && nameof(typeof(device)) === :ReactantDevice
        throw(ArgumentError("Gaussian approximate distribution is not supported with ReactantDevice. If this affects your use case, please contact the package maintainer."))
    end

    train_time = 0.0

    # Precompute summary statistics when the summary network is frozen
    if freeze_summary_network
        if !_has_summary_network(estimator)
            @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
            freeze_summary_network = false
        elseif device isa ReactantDevice
            @warn "`freeze_summary_network = true` is not supported with `reactant_device()` and will be ignored. If this affects your use case, please contact the package maintainer."
            freeze_summary_network = false
        elseif estimator isa PointEstimator && _is_identity(estimator.inference_network)
            @warn "`freeze_summary_network = true` has no effect when the inference network is the identity function; returning the estimator unchanged."
            return trainstate
        else
            _freeze_summary_network!(trainstate)
            verbose && print("Computing summary statistics...")
            t = @elapsed begin
                #TODO device management here
                Z_train = Summaries(summarystatistics(estimator, Z_train; device = device, batchsize = batchsize))
                Z_val = Summaries(summarystatistics(estimator, Z_val; device = device, batchsize = batchsize))
            end
            verbose && println(" Finished in $(round(t, digits = 3)) seconds")
            train_time += t
        end
    end

    verbose && println("Constructing the training set...")
    train_set = _dataloader(estimator, Z_train, θ_train, batchsize)

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    verbose && print("Computing the initial validation risk...")
    min_val_risk, trainstate = _risk(trainstate, loss, val_set, device, adtype)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    trainstate_best = deepcopy(trainstate)
    early_stopping_counter = 0

    # ---- End common setup ----

    train_time += @elapsed for epoch = 1:epochs
        GC.gc(false)

        # For each batch update trainstate and compute the training loss
        epoch_time = @elapsed train_risk, trainstate = _train_step(trainstate, loss, train_set, device, adtype)
        epoch_time += @elapsed val_risk, _ = _risk(trainstate, loss, val_set, device, adtype)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $(lpad(epoch, ndigits(epochs)))  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(trainstate))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # Update the learning rate
        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            trainstate = Optimisers.adjust!(trainstate, next_lr)
        end

        # Save info and check early stopping
        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _save_trainstate(trainstate, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            trainstate_best = deepcopy(trainstate)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    _thaw!(trainstate)
    _save_trainstate(trainstate, savepath; best = false)

    _forcegc(verbose)
    verbose && println("Finished training in $(train_time) seconds")
    if !isnothing(savepath) && savepath != tempdir()
        CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
    end

    return _trainstate_to_device(trainstate_best, cpu_device())
end

function train(trainstate, θ_train::P, θ_val::P, simulator;
    simulator_args = (), m = nothing, # trailing deprecated argument
    simulator_kwargs::NamedTuple = (;),
    batchsize::Integer = 32,
    epochs_per_Z_refresh::Integer = 1,
    epochs::Integer = 100,
    loss = mae,
    savepath::Union{Nothing, String} = tempdir(),
    simulate_just_in_time::Bool = false,
    stopping_epochs::Integer = 5,
    device = nothing,
    use_gpu::Bool = true,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(_findlr(trainstate), zero(_findlr(trainstate)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false,
    adtype::Union{AbstractADType, Nothing} = nothing
) where {P}
    if !isnothing(m)
        @warn "`m` is deprecated, use `simulator_args` instead"
        simulator_args = (m,)
    end

    @assert epochs_per_Z_refresh > 0
    if simulate_just_in_time && epochs_per_Z_refresh != 1
        @error "We cannot simulate the data just-in-time if we aren't refreshing the data every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`"
    end

    # Determine device
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = verbose)

    # Determine adtype and check deep-learning backend + adtype + device are compatible
    adtype = _resolve_adtype(trainstate, device, adtype, verbose)

    # Move trainstate to device and extract the current estimator
    trainstate = _trainstate_to_device(trainstate, device)
    estimator = getestimator(trainstate)

    if estimator isa PosteriorEstimator && estimator.q isa Gaussian && nameof(typeof(device)) === :ReactantDevice
        throw(ArgumentError("Gaussian approximate distribution is not supported with ReactantDevice. If this affects your use case, please contact the package maintainer."))
    end

    verbose && print("Simulating validation data...")
    train_time = @elapsed Z_val = simulator(θ_val, simulator_args...; simulator_kwargs...)
    verbose && println(" Simulated in $(round(train_time, digits = 3)) seconds")

    if freeze_summary_network
        if !_has_summary_network(estimator)
            @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
            freeze_summary_network = false
        elseif device isa ReactantDevice
            @warn "`freeze_summary_network = true` is not supported with `reactant_device()` and will be ignored. If this affects your use case, please contact the package maintainer."
            freeze_summary_network = false
        elseif estimator isa PointEstimator && _is_identity(estimator.inference_network)
            @warn "`freeze_summary_network = true` has no effect when the inference network is the identity function; returning the estimator unchanged."
            return trainstate
        else
            _freeze_summary_network!(trainstate)
            verbose && print("Computing summary statistics...")
            t = @elapsed Z_val = Summaries(summarystatistics(estimator, Z_val; device = device, batchsize = batchsize))
            verbose && println(" Finished in $(round(t, digits = 3)) seconds")
            train_time += t
        end
    end

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # We may store Z_train in its entirety either to reduce simulation overhead or we are
    # not refreshing Z_train every epoch so we need it for subsequent epochs
    store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    verbose && print("Computing the initial validation risk...")
    min_val_risk, trainstate = _risk(trainstate, loss, val_set, device, adtype)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    trainstate_best = deepcopy(trainstate)
    early_stopping_counter = 0

    # ---- End common setup ----

    local train_set
    train_time += @elapsed for epoch = 1:epochs
        GC.gc(false)
        epoch_time = 0.0

        if store_entire_train_set
            # Simulate new training data if needed
            if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
                verbose && print("Simulating training data...")
                train_set = nothing
                GC.gc(false)
                t = @elapsed Z_train = simulator(θ_train, simulator_args...; simulator_kwargs...)
                verbose && println(" Finished in $(round(t, digits = 3)) seconds")
                epoch_time += t
                if freeze_summary_network
                    epoch_time += @elapsed Z_train = Summaries(summarystatistics(estimator, Z_train; use_gpu = use_gpu, batchsize = batchsize))
                end
                train_set = _dataloader(estimator, Z_train, θ_train, batchsize)
            end
            # Update estimator and compute the training risk
            epoch_time += @elapsed train_risk, trainstate = _train_step(trainstate, loss, train_set, device, adtype)
        else
            # Update estimator and compute the training risk
            train_risk = []
            t = 0.0
            for θ ∈ _DataLoader(θ_train, batchsize)
                t += @elapsed Z = simulator(θ, simulator_args...; simulator_kwargs...)
                set = _dataloader(estimator, Z, θ, batchsize)
                epoch_time += @elapsed rsk, trainstate = _train_step(trainstate, loss, set, device, adtype)

                push!(train_risk, rsk)
            end
            verbose && println("Total simulation time: $(round(t, digits = 3)) seconds")
            epoch_time += t
            train_risk = mean(train_risk) #TODO mean of means ≠ grand mean
        end

        # Compute and report the validation risk
        epoch_time += @elapsed val_risk, _ = _risk(trainstate, loss, val_set, device, adtype)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $(lpad(epoch, ndigits(epochs)))  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(trainstate))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # Update the learning rate
        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            trainstate = Optimisers.adjust!(trainstate, next_lr)
        end

        # Save info and check early stopping
        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _save_trainstate(trainstate, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            trainstate_best = deepcopy(trainstate)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    _thaw!(trainstate)
    _save_trainstate(trainstate, savepath; best = false)

    _forcegc(verbose)
    verbose && println("Finished training in $(train_time) seconds")
    if !isnothing(savepath) && savepath != tempdir()
        CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
    end

    return _trainstate_to_device(trainstate_best, cpu_device())
end

function train(trainstate, sampler, simulator;
    K::Integer = 10_000,
    K_val::Integer = K ÷ 2 + 1,
    sampler_args = (),
    sampler_kwargs::NamedTuple = (;),
    simulator_args = (), m = nothing, # trailing deprecated argument
    simulator_kwargs::NamedTuple = (;),
    epochs_per_θ_refresh::Integer = 1, epochs_per_theta_refresh::Integer = 1,
    epochs_per_Z_refresh::Integer = 1,
    simulate_just_in_time::Bool = false,
    loss = mae,
    batchsize::Integer = 32,
    epochs::Integer = 100,
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    device = nothing,
    use_gpu::Bool = true,
    adtype::Union{AbstractADType, Nothing} = nothing,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(_findlr(trainstate), zero(_findlr(trainstate)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false
)
    if !isnothing(m)
        @warn "`m` is deprecated, use `simulator_args` instead"
        simulator_args = (m,)
    end

    @assert epochs_per_θ_refresh == 1 || epochs_per_theta_refresh == 1 "Only one of `epochs_per_θ_refresh` or `epochs_per_theta_refresh` should be provided"
    if epochs_per_theta_refresh != 1
        epochs_per_θ_refresh = epochs_per_theta_refresh
    end

    @assert K > 0
    @assert epochs_per_Z_refresh > 0
    @assert epochs_per_θ_refresh > 0
    @assert epochs_per_θ_refresh % epochs_per_Z_refresh == 0 "`epochs_per_θ_refresh` must be a multiple of `epochs_per_Z_refresh`"

    store_entire_train_set = epochs_per_Z_refresh > 1 || !simulate_just_in_time

    # Number of batches of θ in each epoch
    num_batches = ceil(Int, K / batchsize)

    # Determine device
    device = _resolvedevice(device = device, use_gpu = use_gpu, verbose = verbose)

    # Determine adtype and check deep-learning backend + adtype + device are compatible
    adtype = _resolve_adtype(trainstate, device, adtype, verbose)

    # Move trainstate to device and extract the current estimator
    trainstate = _trainstate_to_device(trainstate, device)
    estimator = getestimator(trainstate)

    if estimator isa PosteriorEstimator && estimator.q isa Gaussian && nameof(typeof(device)) === :ReactantDevice
        throw(ArgumentError("Gaussian approximate distribution is not supported with ReactantDevice. If this affects your use case, please contact the package maintainer."))
    end

    verbose && print("Sampling validation parameters...")
    train_time = @elapsed θ_val = sampler(K_val, sampler_args...; sampler_kwargs...)
    verbose && println(" Finished in $(round(train_time, digits = 3)) seconds")

    verbose && print("Simulating validation data...")
    t = @elapsed Z_val = simulator(θ_val, simulator_args...; simulator_kwargs...)
    verbose && println(" Finished in $(round(t, digits = 3)) seconds")
    train_time += t

    if freeze_summary_network
        if !_has_summary_network(estimator)
            @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
            freeze_summary_network = false
        elseif device isa ReactantDevice
            @warn "`freeze_summary_network = true` is not supported with `reactant_device()` and will be ignored. If this affects your use case, please contact the package maintainer."
            freeze_summary_network = false
        elseif estimator isa PointEstimator && _is_identity(estimator.inference_network)
            @warn "`freeze_summary_network = true` has no effect when the inference network is the identity function; returning the estimator unchanged."
            return trainstate
        else
            _freeze_summary_network!(trainstate)
            verbose && print("Computing summary statistics...")
            t = @elapsed Z_val = Summaries(summarystatistics(estimator, Z_val; device = device, batchsize = batchsize))
            verbose && println(" Finished in $(round(t, digits = 3)) seconds")
            train_time += t
        end
    end

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    verbose && print("Computing the initial validation risk...")
    min_val_risk, trainstate = _risk(trainstate, loss, val_set, device, adtype)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    trainstate_best = deepcopy(trainstate)
    early_stopping_counter = 0

    # ---- End common setup ----

    # For loops create a new scope for the variables that are not present in the
    # enclosing scope, and such variables get a new binding in each iteration of
    # the loop; circumvent this by declaring local variables
    local θ_train
    local train_set
    train_time += @elapsed for epoch ∈ 1:epochs
        GC.gc(false)

        epoch_time = 0.0

        if store_entire_train_set

            # Simulate new training data if needed
            if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0

                # Possibly also refresh the parameter set
                if epoch == 1 || (epoch % epochs_per_θ_refresh) == 0
                    verbose && print("Refreshing the training parameters...")
                    θ_train = nothing
                    GC.gc(false)
                    t = @elapsed θ_train = sampler(K, sampler_args...; sampler_kwargs...)
                    verbose && println(" Finished in $(round(t, digits = 3)) seconds")
                end

                verbose && print("Refreshing the training data...")
                train_set = nothing
                GC.gc(false)
                t = @elapsed Z_train = simulator(θ_train, simulator_args...; simulator_kwargs...)
                verbose && println(" Finished in $(round(t, digits = 3)) seconds")
                epoch_time += t
                if freeze_summary_network
                    epoch_time += @elapsed Z_train = Summaries(summarystatistics(estimator, Z_train; use_gpu = use_gpu, batchsize = batchsize))
                end
                train_set = _dataloader(estimator, Z_train, θ_train, batchsize)
            end

            # For each batch, update estimator and compute the training risk
            epoch_time += @elapsed train_risk, trainstate = _train_step(trainstate, loss, train_set, device, adtype)

        else
            # Full simulation on the fly and just-in-time sampling
            # Precomputation is incompatible with just-in-time simulation since
            # each batch is simulated and immediately consumed.
            train_risk = []
            epoch_time += @elapsed for _ ∈ 1:num_batches
                θ = sampler(batchsize, sampler_args...; sampler_kwargs...)
                Z = simulator(θ, simulator_args...; simulator_kwargs...)
                dat = _dataloader(estimator, Z, θ, batchsize)
                rsk, trainstate = _train_step(trainstate, loss, dat, device, adtype)
                push!(train_risk, rsk)
            end
            train_risk = mean(train_risk) #TODO mean of means ≠ grand mean
        end

        epoch_time += @elapsed val_risk, _ = _risk(trainstate, loss, val_set, device, adtype)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $(lpad(epoch, ndigits(epochs)))  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(trainstate))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # Update the learning rate
        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            trainstate = Optimisers.adjust!(trainstate, next_lr)
        end

        # Save info and check early stopping
        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _save_trainstate(trainstate, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            trainstate_best = deepcopy(trainstate)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    _thaw!(trainstate)
    _save_trainstate(trainstate, savepath; best = false)

    _forcegc(verbose)
    verbose && println("Finished training in $(train_time) seconds")
    if !isnothing(savepath) && savepath != tempdir()
        CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
    end

    return _trainstate_to_device(trainstate_best, cpu_device())
end

"""
    _save_trainstate(trainstate, savepath; best::Bool = true)

Saves the training state to disk. The model parameters, states, optimizer, and optimiser state are saved separately as a BSON file. 
"""
function _save_trainstate end

# For generic estimators, use the user-specified loss function
_loss(estimator, loss) = loss

# Constructs inputs and outputs (default simulated data and corresponding true parameters, respectively)
_inputoutput(estimator, Z, θ) = (Z, θ)

function _dataloader(estimator, Z, θ, batchsize)
    data = _inputoutput(estimator, Z, _stripnames(_extractθ(θ)))
    _DataLoader(data, batchsize)
end

_dataloader(estimator::LuxEstimator, Z, θ, batchsize) = _dataloader(estimator.estimator, Z, θ, batchsize)

# Thin wrapper around DataLoader with sensible training defaults
# NB: redirect_stderr suppresses batchsize warning from DataLoader
function _DataLoader(data, batchsize::Integer; shuffle = true, partial = false)
    oldstd = stdout
    redirect_stderr(devnull)
    data_loader = DataLoader(f32(data), batchsize = batchsize, shuffle = shuffle, partial = partial)
    redirect_stderr(oldstd)
    return data_loader
end

# Learning rate from an optimiser rule
_findlr(trainstate) = _findlr(trainstate.optimizer)
function _findlr(rule::Optimisers.AbstractRule)
    hasproperty(rule, :eta) && return Float64(rule.eta)
    hasproperty(rule, :opt) && return _findlr(rule.opt)       # ReactantOptimiser
    hasproperty(rule, :optimizer) && return _findlr(rule.optimizer)  # other wrappers
    return nothing
end
function _findlr(rule::OptimiserChain)
    for opt in rule.opts
        lr = _findlr(opt)
        isnothing(lr) || return lr
    end
    @warn "Could not determine learning rate from OptimiserChain"
    return nothing
end

function _saveloss(loss_per_epoch, savepath)
    isnothing(savepath) && return
    for path in unique([savepath, tempdir()])
        !ispath(path) && mkpath(path)
        CSV.write(joinpath(path, "loss_per_epoch.csv"), Tables.table(loss_per_epoch), header = false)
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

The returned optimizer can be passed directly to `train()` via the keyword argument `optimiser`.
"""
function loadoptimiser(savepath::String = tempdir(); best::Bool = true)
    prefix = best ? "best" : "final"
    file = "$(prefix)_optimizer.bson"
    path = joinpath(savepath, file)
    @assert isfile(path) "No optimiser state found at $(path). Please ensure that `train()` has been called or check that the correct `savepath` has been provided."
    return load(path, @__MODULE__)[:optimizer]
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
