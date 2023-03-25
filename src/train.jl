"""
	train(θ̂, P, simulator::Function; ...)
	train(θ̂, θ_train::P, θ_val::P, simulator::Function; ...)
	train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...)

Train a neural estimator with architecture `θ̂`.

The methods cater for different forms of on-the-fly simulation. The method that
takes a constructor
`P <: Union{AbstractMatrix, ParameterConfigurations}` and a function `simulator`
for sampling parameters and simulating data, respectively, allows for
both the parameters and the data to be simulated on-the-fly. Note that
`simulator` is called as `simulator(θ, m)`, where `θ` is a set of parameters
and `m` is the sample size (see keyword arguments below).
If provided with specific instances of parameters (`θ_train` and `θ_val`) or
data (`Z_train` and `Z_val`), they will be held fixed during training.

In all methods, the validation set is held fixed to reduce noise when
evaluating the validation risk function, which is used to monitor the
performance of the estimator during training.

If the number of replicates in `Z_train` is a multiple of the
number of replicates for each element of `Z_val`, the training data will be
recycled throughout training. For example, if each
element of `Z_train` consists of 50 replicates, and each
element of `Z_val` consists of 10 replicates, the first epoch uses the first
10 replicates in `Z_train`, the second epoch uses the next 10 replicates, and so
on, until the sixth epoch again uses the first 10 replicates. Note that this
requires the data to be subsettable with the function `subsetdata`.

# Keyword arguments

Arguments common to all methods:
- `loss = mae`: the loss function, which should return the average loss when applied to multiple replicates.
- `epochs::Integer = 100`
- `batchsize::Integer = 32`
- `optimiser = ADAM(1e-4)`
- `savepath::String = ""`: path to save the trained estimator and other information; if savepath is an empty string (default), nothing is saved.
- `stopping_epochs::Integer = 5`: cease training if the risk doesn't improve in this number of epochs.
- `use_gpu::Bool = true`
- `verbose::Bool = true`

Arguments common to `train(θ̂, P, simulator)` and `train(θ̂, θ_train, θ_val, simulator)`:
- `m`: sample sizes (either an `Integer` or a collection of `Integers`).
- `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
- `simulate_just_in_time::Bool = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

Arguments unique to `train(θ̂, P, simulator)`:
- `K::Integer = 10000`: number of parameter vectors in the training set; the size of the validation set is `K ÷ 5`.
- `ξ = nothing`: an arbitrary collection of objects that are fixed (e.g., distance matrices); if `ξ` is provided, the constructor `P` is called as `P(K, ξ)`.

# Examples
```
#TODO
```
"""
function train end


function train(θ̂, P, simulator::Function;
	m,
	ξ = nothing,
	# epochs_per_θ_refresh::Integer = 1, # how often to refresh the training parameters; must be a multiple of `epochs_per_Z_refresh`.
	epochs_per_Z_refresh::Integer = 1,
	#simulate_just_in_time::Bool = false, # NB not needed now that we don't currently cater for epochs_per_θ_refresh > 1
	loss = Flux.Losses.mae,
	optimiser          = ADAM(1e-4),
    batchsize::Integer = 32,
    epochs::Integer    = 100,
	savepath::String   = "",
	stopping_epochs::Integer = 5,
    use_gpu::Bool      = true,
	verbose::Bool      = true,
	K::Integer         = 10_000
	)

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)

	@assert P <: Union{AbstractMatrix, ParameterConfigurations}
	@assert K > 0
	# @assert epochs_per_θ_refresh % epochs_per_Z_refresh  == 0 "`epochs_per_θ_refresh` must be a multiple of `epochs_per_Z_refresh`"

	savebool = savepath != "" # turn off saving if savepath is an empty string
	if savebool
		loss_path = joinpath(savepath, "loss_per_epoch.bson")
		if isfile(loss_path) rm(loss_path) end
		if !ispath(savepath) mkpath(savepath) end
	end

	device = _checkgpu(use_gpu, verbose = verbose)
    θ̂ = θ̂ |> device
    γ = Flux.params(θ̂)

	verbose && println("Simulating validation parameters and validation data...")
	θ_val = isnothing(ξ) ? P(K ÷ 5 + 1) : P(K ÷ 5 + 1, ξ)
	Z_val = _simulate(simulator, θ_val, m)
	Z_val = _quietDataLoader(Z_val, batchsize)

	# Initialise the loss per epoch matrix.
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	loss_per_epoch   = [initial_val_risk initial_val_risk;]
	verbose && println(" Initial validation risk = $initial_val_risk")

	# Save the initial θ̂. This is to prevent bugs in the case that
	# the initial loss does not improve
	savebool && _saveweights(θ̂, savepath, 0)

	# Number of batches of θ to use for each epoch
	batches = ceil((K / batchsize))
	@assert batches > 0

	# For loop creates a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.

	local min_val_risk = initial_val_risk # minimum validation loss; monitor this for early stopping
	local early_stopping_counter = 0
	train_time = @elapsed for epoch ∈ 1:epochs

		# For each batch, update θ̂ and compute the training loss
		train_loss = zero(initial_val_risk)
		epoch_time = @elapsed for _ ∈ 1:batches
			parameters = isnothing(ξ) ? P(batchsize) : P(batchsize, ξ)
			Z = simulate(parameters, m)
			θ = _extractθ(parameters)
			train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
		end
		train_loss = train_loss / (batchsize * batches) # convert to an average

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")
		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			savebool && _saveweights(θ̂, savepath, epoch)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


function train(θ̂, θ_train::P, θ_val::P, simulator::Function;
		m,
		batchsize::Integer = 32,
		epochs_per_Z_refresh::Integer = 1,
		epochs::Integer  = 100,
		loss             = Flux.Losses.mae,
		optimiser        = ADAM(1e-4),
		savepath::String = "",
		simulate_just_in_time::Bool = false,
		stopping_epochs::Integer = 5,
		use_gpu::Bool    = true,
		verbose::Bool    = true
		) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	_checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
	if simulate_just_in_time && epochs_per_Z_refresh != 1 @error "We cannot simulate the data just-in-time if we aren't refreshing it every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`" end

	savebool = savepath != "" # turn off saving if savepath is an empty string
	if savebool
		loss_path = joinpath(savepath, "loss_per_epoch.bson")
		if isfile(loss_path) rm(loss_path) end
		if !ispath(savepath) mkpath(savepath) end
	end

	device = _checkgpu(use_gpu, verbose = verbose)
    θ̂ = θ̂ |> device
    γ = Flux.params(θ̂)

	verbose && println("Simulating validation data...")
	Z_val = _simulate(simulator, θ_val, m)
	Z_val = _quietDataLoader(Z_val, batchsize)
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	verbose && println(" Initial validation risk = $initial_val_risk")

	# Initialise the loss per epoch matrix (NB just using validation for both for now)
	loss_per_epoch = [initial_val_risk initial_val_risk;]

	# Save the initial θ̂. This is to prevent bugs in the case that the initial
	# risk does not improve
	savebool && _saveweights(θ̂, savepath, 0)

	# We may simulate Z_train in its entirety either because (i) we
	# want to avoid the overhead of simulating continuously or (ii) we are
	# not refreshing Z_train every epoch so we need it for subsequent epochs.
	# Either way, store this decision in a variable.
	store_entire_Z_train = !simulate_just_in_time || epochs_per_Z_refresh != 1

	# for loops create a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.
	local Z_train
	local min_val_risk = initial_val_risk # minimum validation loss; monitor this for early stopping
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		train_loss = zero(initial_val_risk)

		if store_entire_Z_train

			# Simulate new training data if needed
			if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
				verbose && print("Simulating training data...")
				Z_train = nothing
				@sync gc()
				t = @elapsed Z_train = _simulate(simulator, θ_train, m)
				Z_train = _quietDataLoader(Z_train, batchsize)
				verbose && println(" Finished in $(round(t, digits = 3)) seconds")
			end

			# For each batch, update θ̂ and compute the training loss
			epoch_time = @elapsed for (Z, θ) in Z_train
			   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end

		else

			# For each batch, update θ̂ and compute the training loss
			epoch_time_simulate = 0.0
			epoch_time    = 0.0
			for parameters ∈ _ParameterLoader(θ_train, batchsize = batchsize)
				epoch_time_simulate += @elapsed Z = simulate(parameters, m)
				θ = _extractθ(parameters)
				epoch_time += @elapsed train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
			verbose && println("Total time spent simulating data: $(round(epoch_time_simulate, digits = 3)) seconds")
			epoch_time += epoch_time_simulate

		end
		train_loss = train_loss / size(θ_train, 2)

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			savebool && _saveweights(θ̂, savepath, epoch)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


function train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
		batchsize::Integer = 32,
		epochs::Integer  = 100,
		loss             = Flux.Losses.mae,
		optimiser        = ADAM(1e-4),
		savepath::String = "",
		stopping_epochs::Integer = 5,
		use_gpu::Bool    = true,
		verbose::Bool    = true
		) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

	@assert batchsize > 0
	@assert epochs > 0
	@assert stopping_epochs > 0

	# Determine if we we need to subset the data.
	# Start by assuming we will not subset the data:
	subsetbool = false
	m = unique(numberreplicates(Z_val))
	M = unique(numberreplicates(Z_train))
	if length(m) == 1 && length(M) == 1 # the data need to be equally replicated in order to subset
		M = M[1]
		m = m[1]
		# The number of replicates in the training data, M, need to be a
		# multiple of the number of replicates in the validation data, m.
		# Also, only subset the data if m ≂̸ M (the subsetting is redundant otherwise).
		subsetbool = M % m == 0 && m != M
	end

	savebool = savepath != "" # turn off saving if savepath is an empty string
	if savebool
		loss_path = joinpath(savepath, "loss_per_epoch.bson")
		if isfile(loss_path) rm(loss_path) end
		if !ispath(savepath) mkpath(savepath) end
	end

	device = _checkgpu(use_gpu, verbose = verbose)
    θ̂ = θ̂ |> device
    γ = Flux.params(θ̂)

	verbose && print("Computing the initial validation risk...")
	Z_val = _quietDataLoader((Z_val, _extractθ(θ_val)), batchsize)
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	verbose && println(" Initial validation risk = $initial_val_risk")

	verbose && print("Computing the initial training risk...")
	tmp = subsetbool ? subsetdata(Z_train, 1:m) : Z_train
	tmp = _quietDataLoader((tmp, _extractθ(θ_train)), batchsize)
	initial_train_risk = _lossdataloader(loss, tmp, θ̂, device)
	verbose && println(" Initial training risk = $initial_train_risk")

	# Initialise the loss per epoch matrix
	loss_per_epoch = [initial_train_risk initial_val_risk;]

	# Save the initial θ̂
	savebool && _saveweights(θ̂, savepath, 0)

	# Training data recycles every x epochs
	x = M ÷ m
	replicates = repeat([(1:m) .+ i*m for i ∈ 0:(x - 1)], outer = ceil(Integer, epochs/x))

	local min_val_risk = initial_val_risk
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		train_loss = zero(initial_train_risk)

		# For each batch update θ̂ and compute the training loss
		Z_train_current = subsetbool ? subsetdata(Z_train, replicates[epoch]) : Z_train
		Z_train_current = _quietDataLoader((Z_train_current, _extractθ(θ_train)), batchsize)
		epoch_time = @elapsed for (Z, θ) in Z_train_current
		   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
		end
		train_loss = train_loss / size(θ_train, 2)

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			savebool && _saveweights(θ̂, savepath, epoch)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


# ---- Wrapper functions for training multiple estimators over a range of sample sizes ----

#TODO not ideal that M is a capital letter, which is usually reserved for types.
"""
	trainx(θ̂, P, simulator, M; ...)
	trainx(θ̂, θ_train, θ_val, simulator, M; ...)
	trainx(θ̂, θ_train, θ_val, Z_train::T, Z_val::T, M; ...)
	trainx(θ̂, θ_train, θ_val, Z_train::V, Z_val::V; ...) where {V <: AbstractVector{AbstractVector{Any}}}

A wrapper around `train` to construct neural estimators for different sample sizes.

The collection `M` specifies the desired sample sizes.
Each estimator is pre-trained with the estimator for the previous sample size.
For example, if `M = [m₁, m₂]`, the estimator for sample size `m₂` is
pre-trained with the estimator for sample size `m₁`. The method for `Z_train::T`
and `Z_val::T` subsets the data using `subsetdata(Z, 1:mᵢ)` for each `mᵢ ∈ M`.

The method for `Z_train::V` and `Z_val::V` trains an estimator for each
element of `Z_train` and `Z_val`; hence, it does not need to invoke `subsetdata`,
which can be slow or difficult to define in some cases (e.g., for graphical data).

The keyword arguments inherit from `train`, and certain keyword arguments
can be given as vectors. For example, if we are training two estimators, we can
use a different number of epochs by providing `epochs = [e₁, e₂]`. Other
arguments that allow vectors are `batchsize`, `stopping_epochs`, and `optimiser`.

# Examples
```
#TODO
```
"""
function trainx end


function trainx(θ̂, P, simulator::Function, M; args...)

	kwargs = (;args...)
	verbose = _checkargs_trainx(kwargs)

	@assert all(M .> 0)
	M = sort(M)
	E = length(M)

	# Create a copy of θ̂ each sample size
	estimators = _deepcopyestimator(θ̂, kwargs, E)

	for i ∈ eachindex(estimators)

		mᵢ = M[i]
		verbose && @info "training with m=$(mᵢ)"

		# Pre-train if this is not the first estimator
		if i > 1 Flux.loadparams!(estimators[i], Flux.params(estimators[i-1])) end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# train
		estimators[i] = train(estimators[i], P, simulator; m = mᵢ, kwargs...)
	end

	return estimators
end


function trainx(θ̂, θ_train::P, θ_val::P, simulator::Function, M; args...)  where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	kwargs = (;args...)
	verbose = _checkargs_trainx(kwargs)

	@assert all(M .> 0)
	M = sort(M)
	E = length(M) # number of estimators

	# Create a copy of θ̂ each sample size
	estimators = _deepcopyestimator(θ̂, kwargs, E)

	for i ∈ eachindex(estimators)

		mᵢ = M[i]
		verbose && @info "training with m=$(mᵢ)"

		# Pre-train if this is not the first estimator
		if i > 1 Flux.loadparams!(estimators[i], Flux.params(estimators[i-1])) end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# train
		estimators[i] = train(estimators[i], θ_train, θ_val, simulator; m = mᵢ, kwargs...)
	end

	return estimators
end


function trainx(θ̂, θ_train::P, θ_val::P, Z_train::V, Z_val::V; args...) where {V <: AbstractVector{S}} where {S <: AbstractVector{T}}  where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

	@assert length(Z_train) == length(Z_val)
	E = length(Z_train) # number of estimators

	kwargs = (;args...)
	verbose = _checkargs_trainx(kwargs)

	# Create a copy of θ̂ each sample size
	estimators = _deepcopyestimator(θ̂, kwargs, E)

	for i ∈ eachindex(estimators)

		# Subset the training and validation data to the current sample size
		Z_trainᵢ = Z_train[i]
		Z_valᵢ   = Z_val[i]

		mᵢ = extrema(unique(numberreplicates(Z_valᵢ)))
		if mᵢ[1] == mᵢ[2]
			mᵢ = mᵢ[1]
			verbose && @info "training with m=$(mᵢ)"
		else
			verbose && @info "training with m ∈ [$(mᵢ[1]), $(mᵢ[2])]"
			mᵢ = "$(mᵢ[1])-$(mᵢ[2])"
		end

		# Pre-train if this is not the first estimator
		if i > 1 Flux.loadparams!(estimators[i], Flux.params(estimators[i-1])) end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# train
		estimators[i] = train(estimators[i], θ_train, θ_val, Z_trainᵢ, Z_valᵢ; kwargs...)
	end

	return estimators
end


function trainx(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}; args...)  where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}

	@assert length(unique(numberreplicates(Z_val))) == 1 "The elements of `Z_val` should be equally replicated: check with `numberreplicates(Z_val)`"
	@assert length(unique(numberreplicates(Z_train))) == 1 "The elements of `Z_train` should be equally replicated: check with `numberreplicates(Z_train)`"

	kwargs = (;args...)
	verbose = _checkargs_trainx(kwargs)

	@assert all(M .> 0)
	M = sort(M)
	E = length(M) # number of estimators

	# Create a copy of θ̂ each sample size
	estimators = _deepcopyestimator(θ̂, kwargs, E)

	for i ∈ eachindex(estimators)

		mᵢ = M[i]
		verbose && @info "training with m=$(mᵢ)"

		# Pre-train if this is not the first estimator
		if i > 1 Flux.loadparams!(estimators[i], Flux.params(estimators[i-1])) end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# Subset the validation data to the current sample size, and then train
		Z_valᵢ = subsetdata(Z_val, 1:mᵢ)
		estimators[i] = train(estimators[i], θ_train, θ_val, Z_train, Z_valᵢ; kwargs...)
	end

	return estimators
end



# ---- Helper functions ----

function _deepcopyestimator(θ̂, kwargs, E)
	# If we are using the GPU, we first need to move θ̂ to the GPU before copying it
	use_gpu = haskey(kwargs, :use_gpu) ? kwargs.use_gpu : true
	device  = _checkgpu(use_gpu, verbose = false)
	θ̂ = θ̂ |> device
	estimators = [deepcopy(θ̂) for _ ∈ 1:E]
	return estimators
end

# E = number of estimators
function _modifyargs(kwargs, i, E)
	for arg ∈ [:epochs, :batchsize, :stopping_epochs, :optimiser]
		if haskey(kwargs, arg)
			field = getfield(kwargs, arg)
			@assert length(field) ∈ (1, E)
			if length(field) > 1
				kwargs = merge(kwargs, NamedTuple{(arg,)}(field[i]))
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


function _checkargs_trainx(kwargs)
	@assert !haskey(kwargs, :m) "`m` should not be provided; use the positional argument `M`"
	verbose = haskey(kwargs, :verbose) ? kwargs.verbose : true
	return verbose
end

# Computes the loss function in a memory-safe manner
function _lossdataloader(loss, data_loader::DataLoader, θ̂, device)
    ls  = 0.0f0
    num = 0
    for (Z, θ) in data_loader
        Z, θ = Z |> device, θ |> device

		# Assuming loss returns an average, convert it to a sum
		b = length(Z)
        ls  += loss(θ̂(Z), θ) * b
        num +=  b
    end

    return cpu(ls / num)
end


function _saveweights(θ̂, savepath, epoch = "")
	if !ispath(savepath) mkpath(savepath) end
	weights = Flux.params(cpu(θ̂)) # return to cpu before serialization
	filename = epoch == "" ? "network.bson" : "network_epoch$epoch.bson"
	networkpath = joinpath(savepath, filename)
	@save networkpath weights
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

function _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)

	Z, θ = Z |> device, θ |> device

	# Compute gradients in such a way that the training loss is also saved.
	# This is equivalent to: gradients = gradient(() -> loss(θ̂(Z), θ), γ)
	ls, back = Zygote.pullback(() -> loss(θ̂(Z), θ), γ)
	gradients = back(one(ls))
	update!(optimiser, γ, gradients)

	# Assuming that loss returns an average, convert it to a sum.
	ls = ls * size(θ)[end]
	return ls
end

function _updatebatch!(θ̂::GNNEstimator, Z, θ, device, loss, γ, optimiser)

	m = numberreplicates(Z)
	Z = Flux.batch(Z)
	Z, θ = Z |> device, θ |> device

	# Compute gradients in such a way that the training loss is also saved.
	# This is equivalent to: gradients = gradient(() -> loss(θ̂(Z), θ), γ)
	ls, back = Zygote.pullback(() -> loss(θ̂(Z, m), θ), γ) # NB here we also pass m to θ̂, since Flux.batch() cannot be differentiated
	gradients = back(one(ls))
	update!(optimiser, γ, gradients)

	# Assuming that loss returns an average, convert it to a sum.
	ls = ls * size(θ)[end]
	return ls
end


# Wrapper function that returns simulated data and the true parameter values
_simulate(simulator::Function, params::P, m) where {P <: Union{AbstractMatrix, ParameterConfigurations}} = (simulator(params, m), _extractθ(params))
