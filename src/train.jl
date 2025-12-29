"""
	train(estimator, sampler::Function, simulator::Function; ...)
	train(estimator, θ_train::P, θ_val::P, simulator::Function; ...) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}
Trains a neural `estimator`.

The methods cater for different variants of "on-the-fly" simulation.
Specifically, a `sampler` can be provided to continuously sample new parameter
vectors from the prior, and a `simulator` can be provided to continuously
simulate new data conditional on the parameters. If
provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data
(`Z_train` and `Z_val`), they will be held fixed during training.

In all methods, the validation parameters and data are held fixed.

The estimator is returned on the CPU so that it can be easily saved post training. 

# Keyword arguments common to all methods:
- `loss = mae` (applicable only to [`PointEstimator`](@ref)): loss function used to train the neural network. In addition to the standard loss functions provided by `Flux` (e.g., `mae`, `mse`), see [Loss functions](@ref) for further options. 
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
- `stopping_epochs = 5`: cease training if the risk does not improve in this number of epochs.
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
- `optimiser = Flux.setup(Adam(5e-4), estimator)`: any [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) optimisation rule for updating the neural-network parameters. When the training data and/or parameters are held fixed, one may wish to employ regularisation to prevent overfitting; for example, `optimiser = Flux.setup(OptimiserChain(WeightDecay(1e-4), Adam()), estimator)`, which corresponds to L₂ regularisation with penalty coefficient λ=10⁻⁴. 
- `lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule}`: defines the learning-rate schedule for adaptively changing the learning rate during training. Accepts either a [ParameterSchedulers.jl](https://fluxml.ai/ParameterSchedulers.jl/dev/) object or `nothing` for a fixed learning rate. By default, it uses [`CosAnneal`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/cyclic/#ParameterSchedulers.CosAnneal) with a maximum set to the initial learning rate from `optimiser`, a minimum of zero, and a period equal to the number of epochs. The learning rate is updated at the end of each epoch. 
- `use_gpu = true`: flag indicating whether to use a GPU if one is available.
- `savepath::Union{Nothing, String} = nothing`: path to save the trained estimator and other information; if `nothing` (default), nothing is saved. Otherwise, the neural-network parameters (i.e., the weights and biases) will be saved during training as `bson` files; the risk function evaluated over the training and validation sets will also be saved, in the first and second columns of `loss_per_epoch.csv`, respectively; the best parameters (as measured by validation risk) will be saved as `best_network.bson`.
- `verbose = true`: flag indicating whether information, including empirical risk values and timings, should be printed to the console during training.

# Keyword arguments common to `train(estimator, sampler, simulator)` and `train(estimator, θ_train, θ_val, simulator)`:
- `m = nothing`: arguments to the simulator (e.g., the number of replicates in each data set). The simulator is called as `simulator(θ, m)` if `m` is given and as `simulator(θ)` otherwise. 
- `epochs_per_Z_refresh = 1`: the number of passes to make through the training set before the training data are refreshed.
- `simulate_just_in_time = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

# Keyword arguments unique to `train(estimator, sampler, simulator)`:
- `K = 10000`: number of parameter vectors in the training set.
- `K_val = K ÷ 5` number of parameter vectors in the validation set.
- `ξ = nothing`: an arbitrary collection of objects that, if provided, will be passed to the parameter sampler as `sampler(K, ξ)`; otherwise, the parameter sampler will be called as `sampler(K)`. Can also be provided as `xi`.
- `epochs_per_θ_refresh = 1`: the number of passes to make through the training set before the training parameters are refreshed. Must be a multiple of `epochs_per_Z_refresh`. Can also be provided as `epochs_per_theta_refresh`.

# Examples
```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²) with priors μ ~ N(0, 1) and σ ~ U(0, 1)
function sampler(K)
	μ = randn(K) # Gaussian prior
	σ = rand(K)  # Uniform prior
	θ = vcat(μ', σ')
	return θ
end
function simulator(θ, m)
	[ϑ[1] .+ ϑ[2] * randn(1, m) for ϑ ∈ eachcol(θ)]
end

# Neural network 
d = 2     # dimension of the parameter vector θ
n = 1     # dimension of each independent replicate of Z
w = 128   # width of each hidden layer 
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, d))
network = DeepSet(ψ, ϕ)

# Initialise the estimator
estimator = PointEstimator(network)

# Number of independent replicates to use during training
m = 15

# Training: simulation on-the-fly
estimator  = train(estimator, sampler, simulator, m = m)

# Training: simulation on-the-fly with fixed parameters
K = 10000
θ_train = sampler(K)
θ_val   = sampler(K)
estimator = train(estimator, θ_train, θ_val, simulator, m = m)

# Training: fixed parameters and fixed data
Z_train   = simulator(θ_train, m)
Z_val     = simulator(θ_val, m)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val)
```
"""
function train end

# NB to follow the naming convention, batchsize and savepath should be batch_size and save_path

function findlr(opt)
    if opt isa Optimisers.Leaf
        return opt.rule.eta
    elseif opt isa AbstractArray || opt isa Tuple
        for subopt in opt
            eta = findlr(subopt)
            if !isnothing(eta)
                return eta
            end
        end
    elseif opt isa NamedTuple
        for (_, subopt) in pairs(opt)
            eta = findlr(subopt)
            if !isnothing(eta)
                return eta
            end
        end
    end
    return nothing
end

function _train(estimator, sampler, simulator;
    m = nothing,
    ξ = nothing, xi = nothing,
    epochs_per_θ_refresh::Integer = 1, epochs_per_theta_refresh::Integer = 1,
    epochs_per_Z_refresh::Integer = 1,
    simulate_just_in_time::Bool = false,
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    batchsize::Integer = 32,
    epochs::Integer = 100,
    savepath::Union{String, Nothing} = nothing,
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    K::Integer = 10_000,
    K_val::Integer = K ÷ 5 + 1,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false)
)

    # Check duplicated arguments that are needed so that the R interface uses ASCII characters only
    @assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
    @assert epochs_per_θ_refresh == 1 || epochs_per_theta_refresh == 1 "Only one of `epochs_per_θ_refresh` or `epochs_per_theta_refresh` should be provided"
    if !isnothing(xi)
        ξ = xi
    end
    if epochs_per_theta_refresh != 1
        epochs_per_θ_refresh = epochs_per_theta_refresh
    end

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)

    @assert K > 0
    @assert epochs_per_θ_refresh > 0
    @assert epochs_per_θ_refresh % epochs_per_Z_refresh == 0 "`epochs_per_θ_refresh` must be a multiple of `epochs_per_Z_refresh`"

    if !isnothing(savepath)
        loss_path = joinpath(savepath, "loss_per_epoch.bson")
        if isfile(loss_path)
            rm(loss_path)
        end
        if !ispath(savepath)
            mkpath(savepath)
        end
    end

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && println("Sampling the validation set...")
    θ_val = isnothing(ξ) ? sampler(K_val) : sampler(K_val, ξ)
    val_set = _constructset(estimator, simulator, θ_val, m, batchsize)

    # Initialise the loss per epoch matrix
    verbose && print("Computing the initial validation risk...")
    val_risk = _risk(estimator, loss, val_set, device)
    loss_per_epoch = [val_risk val_risk;]
    verbose && println(" Initial validation risk = $val_risk")

    # Save initial estimator
    !isnothing(savepath) && _savestate(estimator, savepath, 0)

    # Number of batches of θ in each epoch
    batches = ceil((K / batchsize))

    store_entire_train_set = epochs_per_Z_refresh > 1 || !simulate_just_in_time

    # If provided, convert the learning-rate schedule to an iterable
    if !isnothing(lr_schedule)
        lr_schedule = Stateful(lr_schedule)
    end

    # For loops create a new scope for the variables that are not present in the
    # enclosing scope, and such variables get a new binding in each iteration of
    # the loop; circumvent this by declaring local variables.
    local estimator_best = deepcopy(estimator)
    local θ_train
    local train_set
    local min_val_risk = val_risk # minimum validation loss, monitored for early stopping
    local early_stopping_counter = 0
    train_time = @elapsed for epoch ∈ 1:epochs
        if store_entire_train_set

            # Simulate new training data if needed
            if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0

                # Possibly also refresh the parameter set
                if epoch == 1 || (epoch % epochs_per_θ_refresh) == 0
                    verbose && print("Refreshing the training parameters...")
                    θ_train = nothing
                    @sync gc()
                    t = @elapsed θ_train = isnothing(ξ) ? sampler(K) : sampler(K, ξ)
                    verbose && println(" Finished in $(round(t, digits = 3)) seconds")
                end

                verbose && print("Refreshing the training data...")
                train_set = nothing
                @sync gc()
                t = @elapsed train_set = _constructset(estimator, simulator, θ_train, m, batchsize)
                verbose && println(" Finished in $(round(t, digits = 3)) seconds")
            end

            # For each batch, update estimator and compute the training risk
            epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)

        else
            # Full simulation on the fly and just-in-time sampling
            train_risk = []
            epoch_time = @elapsed for _ ∈ 1:batches
                θ = isnothing(ξ) ? sampler(batchsize) : sampler(batchsize, ξ)
                set = _constructset(estimator, simulator, θ, m, batchsize)
                rsk = _risk(estimator, loss, set, device, optimiser)
                push!(train_risk, rsk)
            end
            train_risk = mean(train_risk)
        end

        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")
        !isnothing(savepath) && @save loss_path loss_per_epoch

        # If the current risk is better than the previous best, save estimator and
        # update the minimum validation risk; otherwise, add to the early stopping counter
        if val_risk <= min_val_risk
            !isnothing(savepath) && _savestate(estimator, savepath, epoch)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end
    end

    # save key information and the best estimator
    !isnothing(savepath) && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
    !isnothing(savepath) && _savebestmodel(savepath)

    return cpu(estimator_best)
end

function _train(estimator, θ_train::P, θ_val::P, simulator;
    m = nothing,
    batchsize::Integer = 32,
    epochs_per_Z_refresh::Integer = 1,
    epochs::Integer = 100,
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    savepath::Union{String, Nothing} = nothing,
    simulate_just_in_time::Bool = false,
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false)
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
    if simulate_just_in_time && epochs_per_Z_refresh != 1
        @error "We cannot simulate the data just-in-time if we aren't refreshing the data every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`"
    end

    if !isnothing(savepath)
        loss_path = joinpath(savepath, "loss_per_epoch.bson")
        if isfile(loss_path)
            rm(loss_path)
        end
        if !ispath(savepath)
            mkpath(savepath)
        end
    end

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && println("Simulating validation data...")
    val_set = _constructset(estimator, simulator, θ_val, m, batchsize)
    verbose && print("Computing the initial validation risk...")
    val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $val_risk")

    # Initialise the loss per epoch matrix and save the initial estimator
    loss_per_epoch = [val_risk val_risk;]
    !isnothing(savepath) && _savestate(estimator, savepath, 0)

    # If provided, convert the learning-rate schedule to an iterable
    if !isnothing(lr_schedule)
        lr_schedule = Stateful(lr_schedule)
    end

    # We may simulate Z_train in its entirety either because (i) we
    # want to avoid the overhead of simulating continuously or (ii) we are
    # not refreshing Z_train every epoch so we need it for subsequent epochs.
    # Either way, store this decision in a variable.
    store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

    local estimator_best = deepcopy(estimator)
    local train_set
    local min_val_risk = val_risk
    local early_stopping_counter = 0
    train_time = @elapsed for epoch = 1:epochs
        sim_time = 0.0
        if store_entire_train_set
            # Simulate new training data if needed
            if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
                verbose && print("Simulating training data...")
                train_set = nothing
                @sync gc()
                sim_time = @elapsed train_set = _constructset(estimator, simulator, θ_train, m, batchsize)
                verbose && println(" Finished in $(round(sim_time, digits = 3)) seconds")
            end
            # Update estimator and compute the training risk
            epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)
        else
            # Update estimator and compute the training risk
            epoch_time = 0.0
            train_risk = []

            for θ ∈ _ParameterLoader(θ_train, batchsize = batchsize)
                sim_time += @elapsed set = _constructset(estimator, simulator, θ, m, batchsize)
                epoch_time += @elapsed rsk = _risk(estimator, loss, set, device, optimiser)
                push!(train_risk, rsk)
            end
            verbose && println("Total time spent simulating data: $(round(sim_time, digits = 3)) seconds")
            train_risk = mean(train_risk)
        end
        epoch_time += sim_time

        # Compute the validation risk and report to the user
        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # save the loss every epoch in case training is prematurely halted
        !isnothing(savepath) && @save loss_path loss_per_epoch

        # If the current risk is better than the previous best, save estimator and
        # update the minimum validation risk
        if val_risk <= min_val_risk
            !isnothing(savepath) && _savestate(estimator, savepath, epoch)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end
    end

    # save key information and save the best estimator as best_network.bson.
    !isnothing(savepath) && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
    !isnothing(savepath) && _savebestmodel(savepath)

    return cpu(estimator_best)
end

function _train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
    batchsize::Integer = 32,
    epochs::Integer = 100,
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    savepath::Union{String, Nothing} = nothing,
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false)
) where {T, P <: Union{Tuple, AbstractMatrix, ParameterConfigurations}}
    @assert batchsize > 0
    @assert epochs > 0
    @assert stopping_epochs > 0

    if !isnothing(savepath)
        loss_path = joinpath(savepath, "loss_per_epoch.bson")
        if isfile(loss_path)
            rm(loss_path)
        end
        if !ispath(savepath)
            mkpath(savepath)
        end
    end

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && print("Computing the initial validation risk...")
    val_set = _constructset(estimator, Z_val, θ_val, batchsize)
    val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $val_risk")

    verbose && print("Computing the initial training risk...")
    train_set = _constructset(estimator, Z_train, θ_train, batchsize)
    initial_train_risk = _risk(estimator, loss, train_set, device)
    verbose && println(" Initial training risk = $initial_train_risk")

    # Initialise the loss per epoch matrix and save the initial estimator
    loss_per_epoch = [initial_train_risk val_risk;]
    !isnothing(savepath) && _savestate(estimator, savepath, 0)

    # If provided, convert the learning-rate schedule to an iterable
    if !isnothing(lr_schedule)
        lr_schedule = Stateful(lr_schedule)
    end

    local estimator_best = deepcopy(estimator)
    local min_val_risk = val_risk
    local early_stopping_counter = 0
    train_time = @elapsed for epoch = 1:epochs

        # For each batch update estimator and compute the training loss
        epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)
        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # save the loss every epoch in case training is prematurely halted
        !isnothing(savepath) && @save loss_path loss_per_epoch

        # If the current loss is better than the previous best, save estimator and update the minimum validation risk
        if val_risk <= min_val_risk
            !isnothing(savepath) && _savestate(estimator, savepath, epoch)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end
    end

    # save key information
    !isnothing(savepath) && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
    !isnothing(savepath) && _savebestmodel(savepath)

    return cpu(estimator_best)
end

# General fallback
train(args...; kwargs...) = _train(args...; kwargs...)

# Wrapper functions for specific types of neural estimators
function train(estimator::Union{IntervalEstimator, QuantileEstimatorDiscrete}, args...; kwargs...)

    # Get the keyword arguments
    kwargs = (; kwargs...)

    # Define the loss function based on the given probabiltiy levels
    τ = f32(estimator.probs)
    # Determine if we need to move τ to the GPU
    use_gpu = haskey(kwargs, :use_gpu) ? kwargs.use_gpu : true
    device = _checkgpu(use_gpu, verbose = false)
    τ = device(τ)
    # Define the loss function
    qloss = (estimator, θ) -> quantileloss(estimator, θ, τ)

    # Notify the user if "loss" is in the keyword arguments
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the quantile loss is always used"
    end
    # Add our quantile loss to the list of keyword arguments
    kwargs = merge(kwargs, (loss = qloss,))

    # Train the estimator
    _train(estimator, args...; kwargs...)
end

function train(estimator::QuantileEstimatorContinuous, args...; kwargs...)
    # We define the loss function in the method _risk(estimator::QuantileEstimatorContinuous)
    # Here, just notify the user if they've assigned a loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the quantile loss is always used"
    end
    _train(estimator, args...; kwargs...)
end

function train(estimator::RatioEstimator, args...; kwargs...)

    # Get the keyword arguments and assign the loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the binary cross-entropy loss is always used"
    end
    _train(estimator, args...; kwargs...)
end

function train(estimator::PosteriorEstimator, args...; kwargs...)

    # Get the keyword arguments and assign the loss function
    kwargs = (; kwargs...)
    if haskey(kwargs, :loss)
        @info "The keyword argument `loss` is not required when training a $(typeof(estimator)), since in this case the KL divergence is always used"
    end
    kwargs = merge(kwargs, (loss = (q, θ) -> -mean(q),))
    _train(estimator, args...; kwargs...)
end

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

# ---- Lower level functions ----

# Wrapper function that constructs a set of input and outputs (usually simulated data and corresponding true parameters)
function _constructset(estimator, simulator::Function, θ::P, m, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = isnothing(m) ? simulator(θ) : simulator(θ, m)
    _constructset(estimator, Z, θ, batchsize)
end
function _constructset(estimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))
    _DataLoader((Z, θ), batchsize)
end
function _constructset(estimator::QuantileEstimatorDiscrete, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    i = estimator.i
    if isnothing(i)
        input = Z
        output = θ
    else
        @assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (estimator.i = $(estimator.i))"
        θᵢ = θ[i:i, :]
        θ₋ᵢ = θ[Not(i), :]
        input = (Z, θ₋ᵢ) # "Tupleise" the input
        output = θᵢ
    end

    _DataLoader((input, output), batchsize)
end
function _constructset(estimator::QuantileEstimatorContinuous, Zτ, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    θ = f32(_extractθ(θ))
    Z, τ = Zτ
    Z = f32(Z)
    τ = f32(τ)

    i = estimator.i
    if isnothing(i)
        input = (Z, τ)
        output = θ
    else
        @assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (estimator.i = $(estimator.i))"
        θᵢ = θ[i:i, :]
        θ₋ᵢ = θ[Not(i), :]
        # Combine each θ₋ᵢ with the corresponding vector of
        # probability levels, which requires repeating θ₋ᵢ appropriately
        θ₋ᵢτ = map(eachindex(τ)) do k
            τₖ = τ[k]
            θ₋ᵢₖ = repeat(θ₋ᵢ[:, k:k], inner = (1, length(τₖ)))
            vcat(θ₋ᵢₖ, τₖ')
        end
        input = (Z, θ₋ᵢτ)   # "Tupleise" the input
        output = θᵢ
    end

    _DataLoader((input, output), batchsize)
end

function _constructset(estimator::RatioEstimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    # Create independent pairs
    K = numobs(Z)
    θ̃ = subsetparameters(θ, shuffle(1:K)) #NB can use getobs here instead of subsetparameters
    Z̃ = Z # NB memory inefficient to replicate the data in this way

    # Combine dependent and independent pairs
    θ = hcat(θ, θ̃)
    if Z isa AbstractVector
        Z = vcat(Z, Z̃)
    elseif Z isa AbstractMatrix
        Z = hcat(Z, Z̃)
    else # general combine along the observation dimension... 
        # NB most of the scenarios are covered above, so the following isn't really tested
        Z = getobs(joinobs(Z, Z̃), 1:2K)
    end

    # Create class labels for output
    labels = [:dependent, :independent]
    output = onehotbatch(repeat(labels, inner = K), labels)[1:1, :]

    # Shuffle everything in case batching isn't shuffled properly downstrean
    idx = shuffle(1:2K)
    Z = getobs(Z, idx)
    θ = getobs(θ, idx)
    output = output[1:1, idx]

    # Combine data and parameters into a single tuple
    input = (Z, θ)

    _DataLoader((input, output), batchsize)
end

function _constructset(estimator::PosteriorEstimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = f32(Z)
    θ = f32(_extractθ(θ))

    input = (Z, θ) # combine data and parameters into a single tuple
    output = θ # irrelevant what we use here, just a placeholder

    _DataLoader((input, output), batchsize)
end

# Computes the risk function in a memory-safe manner, optionally updating the
# neural-network parameters using stochastic gradient descent
function _risk(estimator, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
        k = size(output)[end]
        if !isnothing(optimiser)
            # NB storing the loss in this way is efficient, but it means that
            # the final training risk that we report for each epoch is slightly inaccurate
            # (since the neural-network parameters are updated after each batch)
            ls, ∇ = Flux.withgradient(estimator -> loss(estimator(input), output), estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        else
            ls = loss(estimator(input), output)
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end

    return cpu(sum_loss/K)
end

function _risk(estimator::RatioEstimator, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
        k = size(output)[end]
        loss_fn = est -> Flux.logitbinarycrossentropy(est.network(input), output)
        if !isnothing(optimiser)
            ls, ∇ = Flux.withgradient(loss_fn, estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        else
            ls = loss_fn(estimator)
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end

    return cpu(sum_loss/K)
end

function _risk(estimator::QuantileEstimatorContinuous, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        k = size(output)[end]
        input, output = input |> device, output |> device

        if isnothing(estimator.i)
            Z, τ = input
            input1 = Z
            input2 = permutedims.(τ)
            input = (input1, input2)
            τ = reduce(hcat, τ)                # reduce from vector of vectors to matrix
        else
            Z, θ₋ᵢτ = input
            τ = [x[end, :] for x ∈ θ₋ᵢτ] # extract probability levels
            τ = reduce(hcat, τ)          # reduce from vector of vectors to matrix
        end

        # Repeat τ and θ to facilitate broadcasting and indexing
        # Note that repeat() cannot be differentiated by Zygote
        p = size(output, 1)
        @ignore_derivatives τ = repeat(τ, inner = (p, 1))
        @ignore_derivatives output = repeat(output, inner = (size(τ, 1) ÷ p, 1))

        if !isnothing(optimiser)
            ls, ∇ = Flux.withgradient(estimator -> quantileloss(estimator(input), output, τ), estimator)
            Flux.update!(optimiser, estimator, ∇[1])
        else
            ls = quantileloss(estimator(input), output, τ)
        end
        # Convert average loss to a sum and add to total
        sum_loss += ls * k
        K += k
    end
    return cpu(sum_loss/K)
end

# ---- Wrapper function for training multiple estimators over a range of sample sizes ----

"""
	trainmultiple(estimator, sampler::Function, simulator::Function, m::Vector{Integer}; ...)
	trainmultiple(estimator, θ_train, θ_val, simulator::Function, m::Vector{Integer}; ...)
	trainmultiple(estimator, θ_train, θ_val, Z_train, Z_val, m::Vector{Integer}; ...)
	trainmultiple(estimator, θ_train, θ_val, Z_train::V, Z_val::V; ...) where {V <: AbstractVector{AbstractVector{Any}}}
A wrapper around `train()` to construct multiple neural estimators for different sample sizes `m`.

The positional argument `m` specifies the desired sample sizes.
Each estimator is pre-trained with the estimator for the previous sample size (see [Sainsbury-Dale at al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522), Sec 2.3.3). For example, if `m = [m₁, m₂]`, the estimator for sample size `m₂` is
pre-trained with the estimator for sample size `m₁`.

The method for `Z_train` and `Z_val` subsets the data using
`subsetdata(Z, 1:mᵢ)` for each `mᵢ ∈ m`. The method for `Z_train::V` and
`Z_val::V` trains an estimator for each element of `Z_train::V` and `Z_val::V`
and, hence, it does not need to invoke `subsetdata()`, which can be slow or
difficult to define in some cases (e.g., for graphical data). Note that, in this
case, `m` is inferred from the data.

The keyword arguments inherit from `train()`. The keyword arguments `epochs`,
`batchsize`, `stopping_epochs`, and `optimiser` can each be given as vectors.
For example, if training two estimators, one may use a different number of
epochs for each estimator by providing `epochs = [epoch₁, epoch₂]`.

The function returns a vector of neural estimators, each corresponding to a sample size in `m`.

See also [PiecewiseEstimator](@ref).
"""
function trainmultiple end

function _trainmultiple(estimator; sampler = nothing, simulator = nothing, M = nothing, θ_train = nothing, θ_val = nothing, Z_train = nothing, Z_val = nothing, args...)
    @assert !(typeof(estimator) <: Vector) # check that estimator is not a vector of estimators, which is common error if one calls trainmultiple() on the output of a previous call to trainmultiple()

    kwargs = (; args...)
    verbose = _checkargs_trainmultiple(kwargs)

    @assert all(M .> 0)
    M = sort(M)
    E = length(M)
    estimators = [deepcopy(estimator) for _ ∈ 1:E]

    for i ∈ eachindex(estimators)
        mᵢ = M[i]
        verbose && @info "training with m=$(mᵢ)"

        # Pre-train if this is not the first estimator
        if i > 1
            Flux.loadmodel!(estimators[i], Flux.state(estimators[i - 1]))
        end

        # Modify/check the keyword arguments before passing them onto train
        kwargs = (; args...)
        if haskey(kwargs, :savepath) && !isnothing(kwargs.savepath)
            kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
        end
        kwargs = _modifyargs(kwargs, i, E)

        # Train the estimator, dispatching based on the given arguments
        if !isnothing(sampler)
            estimators[i] = train(estimators[i], sampler, simulator; m = mᵢ, kwargs...)
        elseif !isnothing(simulator)
            estimators[i] = train(estimators[i], θ_train, θ_val, simulator; m = mᵢ, kwargs...)
        else
            # subset the training and validation data to the current sample size, and then train 
            Z_trainᵢ = subsetdata(Z_train, 1:mᵢ)
            Z_valᵢ = subsetdata(Z_val, 1:mᵢ)
            estimators[i] = train(estimators[i], θ_train, θ_val, Z_train, Z_valᵢ; kwargs...)
        end
    end
    return estimators
end
trainmultiple(estimator, sampler, simulator, M; args...) = _trainmultiple(estimator, sampler = sampler, simulator = simulator, M = M; args...)
trainmultiple(estimator, θ_train::P, θ_val::P, simulator, M; args...) where {P <: Union{AbstractMatrix, ParameterConfigurations}} = _trainmultiple(estimator, θ_train = θ_train, θ_val = θ_val, simulator = simulator, M = M; args...)

# This method is for when the data can be easily subsetted
function trainmultiple(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}; args...) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}
    @assert length(unique(numberreplicates(Z_val))) == 1 "The elements of `Z_val` should be equally replicated: check with `numberreplicates(Z_val)`"
    @assert length(unique(numberreplicates(Z_train))) == 1 "The elements of `Z_train` should be equally replicated: check with `numberreplicates(Z_train)`"

    _trainmultiple(estimator, θ_train = θ_train, θ_val = θ_val, Z_train = Z_train, Z_val = Z_val, M = M; args...)
end

# This method is for when the data CANNOT be easily subsetted, so another layer of vectors is needed
function trainmultiple(estimator, θ_train::P, θ_val::P, Z_train::V, Z_val::V; args...) where {V <: AbstractVector{S}} where {S <: Union{V₁, Tuple{V₁, V₂}}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}
    @assert length(Z_train) == length(Z_val)

    @assert !(typeof(estimator) <: Vector) # check that estimator is not a vector of estimators, which is common error if one calls trainmultiple() on the output of a previous call to trainmultiple()

    E = length(Z_train) # number of estimators

    kwargs = (; args...)
    verbose = _checkargs_trainmultiple(kwargs)

    estimators = [deepcopy(estimator) for _ ∈ 1:E]

    for i ∈ eachindex(estimators)

        # Subset the training and validation data to the current sample size
        Z_trainᵢ = Z_train[i]
        Z_valᵢ = Z_val[i]

        mᵢ = extrema(unique(numberreplicates(Z_valᵢ)))
        if mᵢ[1] == mᵢ[2]
            mᵢ = mᵢ[1]
            verbose && @info "training with m=$(mᵢ)"
        else
            verbose && @info "training with m ∈ [$(mᵢ[1]), $(mᵢ[2])]"
            mᵢ = "$(mᵢ[1])-$(mᵢ[2])"
        end

        # Pre-train if this is not the first estimator
        if i > 1
            Flux.loadmodel!(estimators[i], Flux.state(estimators[i - 1]))
        end

        # Modify/check the keyword arguments before passing them onto train
        kwargs = (; args...)
        if haskey(kwargs, :savepath) && !isnothing(kwargs.savepath)
            kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
        end
        kwargs = _modifyargs(kwargs, i, E)

        # Train the estimator for the current sample size
        estimators[i] = train(estimators[i], θ_train, θ_val, Z_trainᵢ, Z_valᵢ; kwargs...)
    end

    return estimators
end

function _checkargs_trainmultiple(kwargs)
    @assert !haskey(kwargs, :m) "Please provide the number of independent replicates, `m`, as a positional argument (i.e., provide the argument simply as `trainmultiple(..., m)` rather than `trainmultiple(..., m = m)`)."
    verbose = haskey(kwargs, :verbose) ? kwargs.verbose : true
    return verbose
end

# ---- Miscellaneous helper functions ----

# E = number of estimators
function _modifyargs(kwargs, i, E)
    for arg ∈ [:epochs, :batchsize, :stopping_epochs]
        if haskey(kwargs, arg)
            field = getfield(kwargs, arg)
            if typeof(field) <: Vector # this check is needed because there is no method length(::Adam)
                @assert length(field) ∈ (1, E)
                if length(field) > 1
                    kwargs = merge(kwargs, NamedTuple{(arg,)}(field[i]))
                end
            end
        end
    end

    kwargs = Dict(pairs(kwargs)) # convert to Dictionary so that kwargs can be passed to train()
    return kwargs
end

function _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
    @assert batchsize > 0
    @assert epochs > 0
    @assert stopping_epochs > 0
    @assert epochs_per_Z_refresh > 0
end

function _savestate(estimator, savepath, epoch = "")
    if !ispath(savepath)
        mkpath(savepath)
    end
    model_state = Flux.state(cpu(estimator))
    file_name = epoch == "" ? "network.bson" : "network_epoch$epoch.bson"
    network_path = joinpath(savepath, file_name)
    @save network_path model_state
end

function _saveinfo(loss_per_epoch, train_time, savepath::String; verbose::Bool = true)
    verbose && println("Finished training in $(train_time) seconds")

    # Recall that we initialised the training loss to the initial validation
    # loss. Slightly better to just use the training loss from the second epoch:
    loss_per_epoch[1, 1] = loss_per_epoch[2, 1]

    # Save quantities of interest
    @save joinpath(savepath, "loss_per_epoch.bson") loss_per_epoch
    CSV.write(joinpath(savepath, "loss_per_epoch.csv"), Tables.table(loss_per_epoch), header = false)
    CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
end

"""
	_savebestmodel(path::String)

Given a `path` to a containing neural networks saved with names
`"network_epochx.bson"` and an object saved as `"loss_per_epoch.bson"`,
saves the weights of the best network (measured by validation loss) as
'best_network.bson'.
"""
function _savebestmodel(path::String)
    # Load the risk as a function of epoch
    loss_per_epoch = load(joinpath(path, "loss_per_epoch.bson"), @__MODULE__)[:loss_per_epoch]

    # Replace NaN with Inf so they won't interfere with finding the minimum risk
    loss_per_epoch .= ifelse.(isnan.(loss_per_epoch), Inf, loss_per_epoch)

    # Find the epoch in which the validation risk was minimised
    best_epoch = argmin(loss_per_epoch[:, 2])

    # Subtract 1 since the first row is the risk evaluated for the initial neural network, that
    # is, the network at epoch 0
    best_epoch -= 1

    # Save the best network
    load_path = joinpath(path, "network_epoch$(best_epoch).bson")
    save_path = joinpath(path, "best_network.bson")
    cp(load_path, save_path, force = true)

    return nothing
end
