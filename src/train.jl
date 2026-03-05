"""
    train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}
    train(estimator, θ_train::P, θ_val::P, simulator::Function; ...) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    train(estimator, sampler::Function, simulator::Function; ...)
Trains a neural `estimator`.

The methods cater for different variants of "on-the-fly" simulation.
Specifically, a `sampler` can be provided to continuously sample new parameter
vectors from the prior, and a `simulator` can be provided to continuously
simulate new data conditional on the parameters. If
provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data
(`Z_train` and `Z_val`), they will be held fixed during training.

The validation parameters and data are always held fixed, 

The trained estimator is always returned on the CPU. 

# Keyword arguments common to all methods:
- `loss = Flux.mae` (applicable only to [`PointEstimator`](@ref)): loss function used to train the neural network. 
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
- `stopping_epochs = 5`: cease training if the risk does not improve in this number of epochs.
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
- `optimiser = Flux.setup(Adam(5e-4), estimator)`: any [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) optimisation rule for updating the neural-network parameters. When the training data and/or parameters are held fixed, one may wish to employ regularisation to prevent overfitting; for example, `optimiser = Flux.setup(OptimiserChain(WeightDecay(1e-4), Adam()), estimator)`, which corresponds to L₂ regularisation with penalty coefficient λ=10⁻⁴. 
- `lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule}`: defines the learning-rate schedule for adaptively changing the learning rate during training. Accepts either a [ParameterSchedulers.jl](https://fluxml.ai/ParameterSchedulers.jl/dev/) object or `nothing` for a fixed learning rate. By default, it uses [`CosAnneal`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/cyclic/#ParameterSchedulers.CosAnneal) with a maximum set to the initial learning rate from `optimiser`, a minimum of zero, and a period equal to the number of epochs. The learning rate is updated at the end of each epoch. 
- `use_gpu = true`: flag indicating whether to use a GPU if one is available.
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

# Number of data sets in each epoch and number of independent replicates in each data set
K = 1000
m = 30

# Training: simulation on-the-fly
estimator  = train(estimator, sampler, simulator, simulator_args = m, K = K)

# Plot the risk history (using any plotting backend)
using Plots
unicodeplots()
plotrisk()

# Training: simulation on-the-fly with fixed parameters 
θ_train = sampler(K)
θ_val   = sampler(K)
estimator = train(estimator, θ_train, θ_val, simulator, simulator_args = m, optimiser = loadoptimiser(), risk_history = loadrisk())

# Training: fixed parameters and fixed data
Z_train   = simulator(θ_train, m)
Z_val     = simulator(θ_val, m)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val, optimiser = loadoptimiser(), risk_history = loadrisk())
```
"""
function train end

function train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
    batchsize::Integer = 32,
    epochs::Integer = 100,
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing
) where {T, P <: Union{Tuple, AbstractMatrix, ParameterConfigurations}}
    verbose && print("Constructing the training set...")
    train_set = _dataloader(estimator, Z_train, θ_train, batchsize)

    verbose && print("Constructing the validation set...")
    val_set = _dataloader(estimator, Z_val, θ_val, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
    early_stopping_counter = 0

    # ---- End common setup ----

    train_time = @elapsed for epoch = 1:epochs
        GC.gc(false)

        # For each batch update estimator and compute the training loss
        epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)
        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        # Update the learning rate
        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        # Save information and check early stopping
        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _savenetwork(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    _savefinal(estimator, optimiser, train_time, savepath, verbose)

    return cpu(estimator_best)
end

function train(estimator, θ_train::P, θ_val::P, simulator;
    simulator_args = (), m = nothing, # trailing deprecated argument
    simulator_kwargs::NamedTuple = (;),
    batchsize::Integer = 32,
    epochs_per_Z_refresh::Integer = 1,
    epochs::Integer = 100,
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    savepath::Union{Nothing, String} = tempdir(),
    simulate_just_in_time::Bool = false,
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing
) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    if !isnothing(m)
        @warn "`m` is deprecated, use `simulator_args` instead"
        simulator_args = (m,)
    end

    @assert epochs_per_Z_refresh > 0
    if simulate_just_in_time && epochs_per_Z_refresh != 1
        @error "We cannot simulate the data just-in-time if we aren't refreshing the data every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`"
    end

    # We may simulate Z_train in its entirety either because (i) we
    # want to avoid the overhead of simulating continuously or (ii) we are
    # not refreshing Z_train every epoch so we need it for subsequent epochs
    store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

    verbose && println("Simulating validation data and constructing the validation set...")
    val_set = _dataloader(estimator, simulator, θ_val, simulator_args, simulator_kwargs, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
    early_stopping_counter = 0

    # ---- End common setup ----

    local train_set
    train_time = @elapsed for epoch = 1:epochs
        GC.gc(false)

        sim_time = 0.0
        if store_entire_train_set
            # Simulate new training data if needed
            if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
                verbose && print("Simulating training data...")
                train_set = nothing
                GC.gc(false)
                sim_time = @elapsed train_set = _dataloader(estimator, simulator, θ_train, simulator_args, simulator_kwargs, batchsize)
                verbose && println(" Finished in $(round(sim_time, digits = 3)) seconds")
            end
            # Update estimator and compute the training risk
            epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)
        else
            # Update estimator and compute the training risk
            epoch_time = 0.0
            train_risk = []

            for θ ∈ _ParameterLoader(θ_train, batchsize = batchsize)
                sim_time += @elapsed set = _dataloader(estimator, simulator, θ, simulator_args, simulator_kwargs, batchsize)
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

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _savenetwork(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

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
    loss = Flux.Losses.mae,
    optimiser = Flux.setup(Adam(5e-4), estimator),
    batchsize::Integer = 32,
    epochs::Integer = 100,
    savepath::Union{Nothing, String} = tempdir(),
    stopping_epochs::Integer = 5,
    use_gpu::Bool = true,
    verbose::Bool = true,
    K::Integer = 10_000,
    K_val::Integer = K ÷ 2 + 1,
    lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = CosAnneal(findlr(optimiser), zero(findlr(optimiser)), epochs, false),
    risk_history::Union{Nothing, Matrix} = nothing
)
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

    verbose && println("Sampling validation parameters...")
    θ_val = sampler(K_val, sampler_args...; sampler_kwargs...)
    verbose && println("Simulating validation data and constructing the validation set...")
    val_set = _dataloader(estimator, simulator, θ_val, simulator_args, simulator_kwargs, batchsize)

    # ---- Common setup ----

    loss = _loss(estimator, loss)
    _checkargs(batchsize, epochs, stopping_epochs, risk_history)

    device = _checkgpu(use_gpu, verbose = verbose)
    estimator = estimator |> device
    optimiser = optimiser |> device

    verbose && print("Computing the initial validation risk...")
    min_val_risk = _risk(estimator, loss, val_set, device)
    verbose && println(" Initial validation risk = $min_val_risk")

    loss_per_epoch = [min_val_risk min_val_risk;]
    if !isnothing(risk_history)
        loss_per_epoch = vcat(risk_history, loss_per_epoch)
    end

    !isnothing(lr_schedule) && (lr_schedule = Stateful(lr_schedule))

    estimator_best = deepcopy(estimator)
    early_stopping_counter = 0

    # ---- End common setup ----

    # For loops create a new scope for the variables that are not present in the
    # enclosing scope, and such variables get a new binding in each iteration of
    # the loop; circumvent this by declaring local variables
    local θ_train
    local train_set
    train_time = @elapsed for epoch ∈ 1:epochs
        GC.gc(false)

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
                t = @elapsed train_set = _dataloader(estimator, simulator, θ_train, simulator_args, simulator_kwargs, batchsize)
                verbose && println(" Finished in $(round(t, digits = 3)) seconds")
            end

            # For each batch, update estimator and compute the training risk
            epoch_time = @elapsed train_risk = _risk(estimator, loss, train_set, device, optimiser)

        else
            # Full simulation on the fly and just-in-time sampling
            train_risk = []
            epoch_time = @elapsed for _ ∈ 1:num_batches
                θ = sampler(batchsize, sampler_args...; sampler_kwargs...)
                dat = _dataloader(estimator, simulator, θ, simulator_args, simulator_kwargs, batchsize)
                rsk = _risk(estimator, loss, dat, device, optimiser)
                push!(train_risk, rsk)
            end
            train_risk = mean(train_risk)
        end

        epoch_time += @elapsed val_risk = _risk(estimator, loss, val_set, device)
        loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
        verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Learning rate: $(@sprintf "%.2E" findlr(optimiser))  Epoch time: $(round(epoch_time, digits = 3)) seconds")

        if !isnothing(lr_schedule)
            next_lr = ParameterSchedulers.next!(lr_schedule)
            Optimisers.adjust!(optimiser, next_lr)
        end

        _saveloss(loss_per_epoch, savepath)
        if val_risk <= min_val_risk
            _savenetwork(estimator, savepath; best = true)
            _saveoptimiser(optimiser, savepath; best = true)
            min_val_risk = val_risk
            early_stopping_counter = 0
            estimator_best = deepcopy(estimator)
        else
            early_stopping_counter += 1
            early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
        end
    end

    _savefinal(estimator, optimiser, train_time, savepath, verbose)

    return cpu(estimator_best)
end

# For generic estimators, use the user-specified loss function
_loss(estimator, loss) = loss

# Constructs inputs and outputs (default simulated data and corresponding true parameters, respectively)
function _inputoutput(estimator, Z, θ::P) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    input = Z
    output = _extractθ(θ)
    return input, output
end

function _dataloader(estimator, simulator, θ::P, simulator_args, simulator_kwargs, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    Z = simulator(θ, simulator_args...; simulator_kwargs...)
    _dataloader(estimator, Z, θ, batchsize)
end

function _dataloader(estimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
    data = _inputoutput(estimator, Z, θ)
    _DataLoader(data, batchsize)
end

# Thin wrapper around Flux's DataLoader with sensible training defaults
# NB: redirect_stderr suppresses batchsize warning from DataLoader
function _DataLoader(data, batchsize::Integer; shuffle = true, partial = false)
    oldstd = stdout
    redirect_stderr(devnull)
    data_loader = DataLoader(f32(data), batchsize = batchsize, shuffle = shuffle, partial = partial)
    redirect_stderr(oldstd)
    return data_loader
end

# Computes the risk for a general loss function in a memory-safe manner, updating the
# neural-network parameters using stochastic gradient descent if optimiser is specified
function _risk(estimator, loss, data_loader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in data_loader
        input, output = input |> device, output |> device
        k = Flux.numobs(input)
        if !isnothing(optimiser)
            # NB storing the loss in this way is computationally efficient, but it means that
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

function _savenetwork(estimator, savepath; best::Bool = true)
    isnothing(savepath) && return
    if savepath != tempdir()
        file = best ? "best_network.bson" : "final_network.bson"
        model_state = Flux.state(cpu(estimator))
        !ispath(savepath) && mkpath(savepath)
        @save joinpath(savepath, file) model_state
    end
end

function _savefinal(estimator, optimiser, train_time, savepath, verbose)
    _savenetwork(estimator, savepath; best = false)
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

# ---- Wrapper function for training multiple estimators over a range of sample sizes ----

#NB I think this should be deprecated, adds code complexity for something that is not often used, or can be easily implemented.
# Otherwise, it could be made more generally useful (i.e., implement pretraining more generally).

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
    num_estimators = length(M)
    estimators = [deepcopy(estimator) for _ ∈ 1:num_estimators]

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
        kwargs = _modifyargs(kwargs, i, num_estimators)

        # Train the estimator, dispatching based on the given arguments
        if !isnothing(sampler)
            estimators[i] = train(estimators[i], sampler, simulator; simulator_args = mᵢ, kwargs...)
        elseif !isnothing(simulator)
            estimators[i] = train(estimators[i], θ_train, θ_val, simulator; simulator_args = mᵢ, kwargs...)
        else
            # subset the training and validation data to the current sample size, and then train 
            Z_trainᵢ = subsetdata(Z_train, 1:mᵢ)
            Z_valᵢ = subsetdata(Z_val, 1:mᵢ)
            estimators[i] = train(estimators[i], θ_train, θ_val, Z_trainᵢ, Z_valᵢ; kwargs...)
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

# This method is for when the data cannot be easily subsetted, so another layer of vectors is needed
function trainmultiple(estimator, θ_train::P, θ_val::P, Z_train::V, Z_val::V; args...) where {V <: AbstractVector{S}} where {S <: Union{V₁, Tuple{V₁, V₂}}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}
    @assert length(Z_train) == length(Z_val)

    @assert !(typeof(estimator) <: Vector) # check that estimator is not a vector of estimators, which is common error if one calls trainmultiple() on the output of a previous call to trainmultiple()

    num_estimators = length(Z_train) # number of estimators

    kwargs = (; args...)
    verbose = _checkargs_trainmultiple(kwargs)

    estimators = [deepcopy(estimator) for _ ∈ 1:num_estimators]

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
        kwargs = _modifyargs(kwargs, i, num_estimators)

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

function _modifyargs(kwargs, i, num_estimators)
    for arg ∈ [:epochs, :batchsize, :stopping_epochs]
        if haskey(kwargs, arg)
            field = getfield(kwargs, arg)
            if typeof(field) <: Vector # this check is needed because there is no method length(::Adam)
                @assert length(field) ∈ (1, num_estimators)
                if length(field) > 1
                    kwargs = merge(kwargs, NamedTuple{(arg,)}(field[i]))
                end
            end
        end
    end

    kwargs = Dict(pairs(kwargs)) # convert to Dictionary so that kwargs can be passed to train()
    return kwargs
end
