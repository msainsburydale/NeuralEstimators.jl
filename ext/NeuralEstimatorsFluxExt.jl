module NeuralEstimatorsFluxExt

using NeuralEstimators
using NeuralEstimators: mae, _findlr, _loss, _inputoutput, _dataloader, _DataLoader, _saveloss, _saveoptimiser, _savefinal, _checkargs, _getdevice, cpu, gpu, numobs, Shortcut, _precomputesummaries, _risk
using Flux
using Optimisers
using ADTypes
using BSON
using BSON: @save, load
using Printf: @sprintf
using ParameterSchedulers
using Statistics: mean
import NeuralEstimators: MLP, ResidualBlock, train, _testmode!, _saveestimator, _riskvalue_and_gradient, _gaussianmixture_final_layer

# ---------------------- Simple functions ---------------------

_testmode!(args...) = Flux.testmode!(args...)


# ---------------------- Architectures ---------------------

# TODO Define _Chain(), _Dense(), _Parallel(), _Shortcut(), Conv(), BatchNorm(), SkipConnection() in the core package which are overloaded in the extensions. If both Lux and Flux loaded, can dispatch to the correct method based on a simple struct sentinel

function NeuralEstimators._gaussianmixture_final_layer(out::Integer, num_components::Integer, d::Integer)
    Parallel(vcat,
        Chain(Dense(out, num_components), softmax), # ∑wⱼ = 1
        Dense(out, d * num_components, identity),   # μ ∈ ℝ
        Dense(out, d * num_components, softplus)    # σ > 0
    )
end

function MLP(in::Integer, out::Integer; depth::Integer = 2, width::Integer = 128, activation::Function = relu, output_activation = identity, final_layer = nothing)
    @assert depth > 0
    @assert width > 0

    layers = []
    push!(layers, Dense(in => width, activation))
    if depth > 1
        push!(layers, [Dense(width => width, activation) for _ ∈ 2:depth]...)
    end
    push!(layers, Dense(width => out, output_activation))
    if !isnothing(final_layer)
        push!(layers, final_layer)
    end

    return MLP(Chain(layers...))
end

(b::ResidualBlock)(x) = relu.(b.block(x))
function ResidualBlock(filter, channels; stride = 1)
    layer = Chain(
        Conv(filter, channels; stride = stride, pad = 1, bias = false),
        BatchNorm(channels[2], relu),
        Conv(filter, channels[2]=>channels[2]; pad = 1, bias = false),
        BatchNorm(channels[2])
    )

    if stride == 1 && channels[1] == channels[2]
        # dimensions match, can add input directly to output
        connection = +
    else
        #TODO options for different dimension matching (padding vs. projection)
        # Projection connection using 1x1 convolution
        connection = Shortcut(
            Chain(
            Conv((1, 1), channels; stride = stride, bias = false),
            BatchNorm(channels[2])
        )
        )
    end

    ResidualBlock(SkipConnection(layer, connection))
end

# ---------------------- Training methods ---------------------

function train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
    batchsize::Integer = 32,
    epochs::Integer = 100,
    loss = mae,
    optimiser = Optimisers.setup(Adam(5e-4), estimator),
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    use_reactant::Bool = false,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = ParameterSchedulers.CosAnneal(_findlr(optimiser), zero(_findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false,
    adtype::AbstractADType = AutoZygote()
) where {P, T}

    device = _getdevice(use_gpu, use_reactant; verbose = verbose)

    if freeze_summary_network && !hasfield(typeof(estimator), :summary_network)
        @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
        freeze_summary_network = false
    end

    train_time = 0.0

    # Precompute summary statistics before the epoch loop when the summary network
    # is frozen and the training data are fixed. Because the summary network
    # parameters never change, applying it to Z_train and Z_val yields the same
    # result every epoch — so we pay that cost once here and wrap the matrices in
    # Summaries to signal to _summarystatistics that no further network pass is needed.
    # This gives a large speedup when the summary network is expensive relative to
    # the inference network.
    if freeze_summary_network
        verbose && print("Computing summary statistics...")
        t = @elapsed begin
            Z_train = Summaries(_precomputesummaries(estimator, Z_train; use_gpu = use_gpu, batchsize = batchsize))
            Z_val = Summaries(_precomputesummaries(estimator, Z_val; use_gpu = use_gpu, batchsize = batchsize))
        end
        verbose && println(" Finished in $(round(t, digits = 3)) seconds")
        train_time += t
    end

    verbose && println("Constructing the training set...")
    train_set = _dataloader(estimator, Z_train, θ_train, batchsize)

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    estimator = estimator |> device
    optimiser = optimiser |> device

    freeze_summary_network && Optimisers.freeze!(optimiser.summary_network)

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
    early_stopping_counter = 0

    # ---- End common setup ----

    train_time += @elapsed for epoch = 1:epochs
        GC.gc(false)

        # For each batch update estimator and compute the training loss
        epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser, adtype)
        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # Update the learning rate
        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        # Save information and check early stopping
        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _saveestimator(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    Optimisers.thaw!(optimiser)
    _savefinal(estimator, optimiser, train_time, savepath, verbose)

    return cpu(estimator_best)
end

function train(estimator, θ_train::P, θ_val::P, simulator;
    simulator_args = (), m = nothing, # trailing deprecated argument
    simulator_kwargs::NamedTuple = (;),
    batchsize::Integer = 32,
    epochs_per_Z_refresh::Integer = 1,
    epochs::Integer = 100,
    loss = mae,
    optimiser = Optimisers.setup(Adam(5e-4), estimator),
    savepath::Union{Nothing, String} = tempdir(),
    simulate_just_in_time::Bool = false,
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    use_reactant::Bool = false,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(_findlr(optimiser), zero(_findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false,
    adtype::AbstractADType = AutoZygote()
) where {P}
    
    device = _getdevice(use_gpu, use_reactant; verbose = verbose)

    if freeze_summary_network && !hasfield(typeof(estimator), :summary_network)
        @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
        freeze_summary_network = false
    end

    if !isnothing(m)
        @warn "`m` is deprecated, use `simulator_args` instead"
        simulator_args = (m,)
    end

    @assert epochs_per_Z_refresh > 0
    if simulate_just_in_time && epochs_per_Z_refresh != 1
        @error "We cannot simulate the data just-in-time if we aren't refreshing the data every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`"
    end

    verbose && print("Simulating validation data...")
    train_time = @elapsed Z_val = simulator(θ_val, simulator_args...; simulator_kwargs...)
    verbose && println(" Simulated in $(round(train_time, digits = 3)) seconds")

    if freeze_summary_network
        verbose && print("Computing summary statistics...")
        t = @elapsed Z_val = Summaries(_precomputesummaries(estimator, Z_val; use_gpu = use_gpu, batchsize = batchsize))
        verbose && println(" Finished in $(round(t, digits = 3)) seconds")
        train_time += t
    end

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # We may store Z_train in its entirety either because we
    # want to avoid the overhead of simulating continuously or we are
    # not refreshing Z_train every epoch so we need it for subsequent epochs
    store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    estimator = estimator |> device
    optimiser = optimiser |> device

    freeze_summary_network && Optimisers.freeze!(optimiser.summary_network)

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
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
                    epoch_time += @elapsed Z_train = Summaries(_precomputesummaries(estimator, Z_train; use_gpu = use_gpu, batchsize = batchsize))
                end
                train_set = _dataloader(estimator, Z_train, θ_train, batchsize)
            end
            # Update estimator and compute the training risk
            epoch_time += @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser, adtype)
        else
            # Update estimator and compute the training risk
            train_risk = []
            t = 0.0
            for θ ∈ _DataLoader(θ_train, batchsize)
                t += @elapsed Z = simulator(θ, simulator_args...; simulator_kwargs...)
                set = _dataloader(estimator, Z, θ, batchsize)
                epoch_time += @elapsed rsk = _risk(estimator, loss, set, device, optimiser, adtype)
                push!(train_risk, rsk)
            end
            verbose && println("Total simulation time: $(round(t, digits = 3)) seconds")
            epoch_time += t
            train_risk = mean(train_risk)
        end

        # Compute the validation risk and report to the user
        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _saveestimator(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    Optimisers.thaw!(optimiser)
    _savefinal(estimator, optimiser, train_time, savepath, verbose)

    return cpu(estimator_best)
end

function train(estimator, sampler, simulator;
    sampler_args = (), ξ = nothing, xi = nothing, # trailing deprecated arguments
    sampler_kwargs::NamedTuple = (;),
    simulator_args = (), m = nothing, # trailing deprecated argument
    simulator_kwargs::NamedTuple = (;),
    epochs_per_θ_refresh::Integer = 1, epochs_per_theta_refresh::Integer = 1,
    epochs_per_Z_refresh::Integer = 1,
    simulate_just_in_time::Bool = false,
    loss = mae,
    optimiser = Optimisers.setup(Adam(5e-4), estimator),
    batchsize::Integer = 32,
    epochs::Integer = 100,
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    use_reactant::Bool = false,
    verbose::Bool = true,
    K::Integer = 10_000,
    K_val::Integer = K ÷ 2 + 1,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(_findlr(optimiser), zero(_findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing,
    freeze_summary_network::Bool = false,
    adtype::AbstractADType = AutoZygote()
)
    device = _getdevice(use_gpu, use_reactant; verbose = verbose)

    if freeze_summary_network && !hasfield(typeof(estimator), :summary_network)
        @warn "`freeze_summary_network = true` has no effect for estimators without a `summary_network` field"
        freeze_summary_network = false
    end

    @assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
    if !isnothing(xi)
        ξ = xi
    end
    if !isnothing(ξ)
        @warn "`ξ` is deprecated, use `sampler_args` instead"
        # sampler_args = isempty(sampler_args) ? (ξ,) : ((ξ,), sampler_args...)
        # sampler_args = ((ξ,), sampler_args...)
        sampler_args = (ξ,)
    end
    if !isnothing(m)
        @warn "`m` is deprecated, use `simulator_args` instead"
        # simulator_args = (m, simulator_args...)
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

    verbose && print("Sampling validation parameters...")
    train_time = @elapsed θ_val = sampler(K_val, sampler_args...; sampler_kwargs...)
    verbose && println(" Finished in $(round(train_time, digits = 3)) seconds")

    verbose && print("Simulating validation data...")
    t = @elapsed Z_val = simulator(θ_val, simulator_args...; simulator_kwargs...)
    verbose && println(" Finished in $(round(t, digits = 3)) seconds")
    train_time += t

    if freeze_summary_network
        verbose && print("Computing summary statistics...")
        t = @elapsed Z_val = Summaries(_precomputesummaries(estimator, Z_val; use_gpu = use_gpu, batchsize = batchsize))
        verbose && println(" Finished in $(round(t, digits = 3)) seconds")
        train_time += t
    end

    verbose && println("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    estimator = estimator |> device
    optimiser = optimiser |> device

    freeze_summary_network && Optimisers.freeze!(optimiser.summary_network)

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = ParameterSchedulers.Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
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
                    epoch_time += @elapsed Z_train = Summaries(_precomputesummaries(estimator, Z_train; use_gpu = use_gpu, batchsize = batchsize))
                end
                train_set = _dataloader(estimator, Z_train, θ_train, batchsize)
            end

            # For each batch, update estimator and compute the training risk
            epoch_time += @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser, adtype)

        else
            # Full simulation on the fly and just-in-time sampling
            # Precomputation is incompatible with just-in-time simulation since
            # each batch is simulated and immediately consumed.
            train_risk = []
            epoch_time += @elapsed for _ ∈ 1:num_batches
                θ = sampler(batchsize, sampler_args...; sampler_kwargs...)
                Z = simulator(θ, simulator_args...; simulator_kwargs...)
                dat = _dataloader(estimator, Z, θ, batchsize)
                rsk = _risk(estimator, loss, dat, device, optimiser, adtype)
                push!(train_risk, rsk)
            end
            train_risk = mean(train_risk)
        end

        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" _findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _saveestimator(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    Optimisers.thaw!(optimiser)
    _savefinal(estimator, optimiser, train_time, savepath, verbose)

    return cpu(estimator_best)
end

function _riskvalue_and_gradient(estimator, loss, input, output, adtype)
    ls, ∇ = Flux.withgradient(estimator -> loss(estimator(input), output), adtype, estimator)
    return ls, ∇[1]
end


function _saveestimator(estimator, savepath; best::Bool = true)
    isnothing(savepath) && return
    if savepath != tempdir()
        file = best ? "best_network.bson" : "final_network.bson"
        model_state = Flux.state(cpu(estimator))
        !ispath(savepath) && mkpath(savepath)
        @save joinpath(savepath, file) model_state
    end
end


end
