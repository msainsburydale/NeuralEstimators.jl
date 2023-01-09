"""
Generic function for training a neural estimator.

The methods are designed to cater for different forms of "on-the-fly simulation"
(see the online documentation). In all methods, the validation data are held
fixed so that the validation risk function, which is used to monitor the
performance of the estimator during training, is not subject to noise.

Note that `train` is a mutating function, but the suffix `!` is omitted to avoid clashes with the
`Flux` function, `train!`.

# Keyword arguments

Arguments common to all methods:
- `loss = mae`: the loss function, which should return the average loss when applied to multiple replicates.
- `epochs::Integer = 100`
- `batchsize::Integer = 32`
- `optimiser = ADAM(1e-4)`
- `savepath::String = ""`: path to save the trained `θ̂` and other information; if savepath is an empty string (default), nothing is saved.
- `stopping_epochs::Integer = 5`: cease training if the risk doesn't improve in `stopping_epochs` epochs.
- `use_gpu::Bool = true`
- `verbose::Bool = true`

Arguments common to `train(θ̂, P)` and `train(θ̂, θ_train, θ_val)`:
- `m`: sample sizes (either an `Integer` or a collection of `Integers`).
- `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
- `simulate_just_in_time::Bool = false`: should we simulate the data "just-in-time"?

Arguments unique to `train(θ̂, P)`:
- `K::Integer = 10_000`: number of parameter vectors in the training set; the size of the validation set is `K ÷ 5`.
- `epochs_per_θ_refresh::Integer = 1`: how often to refresh the training parameters; this must be a multiple of `epochs_per_Z_refresh`.
- `ξ = nothing`: invariant model information; if `ξ` is provided, the constructor `P` is called as `P(K, ξ)`.
"""
function train end


"""
	train(θ̂, P; <keyword args>)

Train the neural estimator `θ̂` by providing a constructor, `P`, where `P` is
a subtype of `AbstractMatrix` or `ParameterConfigurations`, to
automatically sample the sets of training and validation parameters.
"""
function train(θ̂, P;
	m,
	ξ = nothing,
	# epochs_per_θ_refresh::Integer = 1, # TODO
	# epochs_per_Z_refresh::Integer = 1, # TODO
	# simulate_just_in_time::Bool = false, # TODO
	loss = Flux.Losses.mae,
	optimiser          = ADAM(1e-4),
    batchsize::Integer = 32,
    epochs::Integer    = 100,
	savepath::String   = "", # "runs/"
	stopping_epochs::Integer = 5,
    use_gpu::Bool      = true,
	verbose::Bool      = true,
	K::Integer         = 10_000
	)

	simulate_just_in_time::Bool = false # TODO
	epochs_per_Z_refresh = 1 # TODO
	epochs_per_θ_refresh = 1 # TODO

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh, simulate_just_in_time)

	# TODO better way to enforce P to be a type and specifically a subtype of ParameterConfigurations?
	@assert P <: Union{AbstractMatrix, ParameterConfigurations}
	@assert K > 0
	@assert epochs_per_θ_refresh % epochs_per_Z_refresh  == 0 "`epochs_per_θ_refresh` must be a multiple of `epochs_per_Z_refresh`"

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
	Z_val = _simulate(θ_val, m)
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
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


"""
	train(θ̂, θ_train::P, θ_val::P; <keyword args>)

Train the neural estimator `θ̂` by providing the training and validation parameter
sets explicitly as `θ_train` and `θ_val`, which are both held fixed during training.
"""
function train(θ̂, θ_train::P, θ_val::P;
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

	_checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh, simulate_just_in_time)

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
	Z_val = _simulate(θ_val, m)
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
				t = @elapsed Z_train = _simulate(θ_train, m)
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
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


"""
	train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T; <keyword args>)

Train the neural estimator `θ̂` by providing the training and validation parameter
sets, `θ_train` and `θ_val`, and the training and validation data sets,
`Z_train` and `Z_val`, all of which are held fixed during training.

The sample size argument `m` is inferred from `Z_val`. The training data `Z_train`
can contain `M` replicates, where `M` is a multiple of `m`; the training data will
then be recycled to imitate on-the-fly simulation. For example, if `M = 50` and
`m = 10`, epoch 1 uses the first 10 replicates, epoch 2 uses the next 10
replicates, and so on, until epoch 6 again uses the first 10 replicates.

Note that the elements of `Z_train` and `Z_val` should each be equally replicated; that
is, the size of the last dimension in each array in `Z_train` should be constant,
and similarly for `Z_val` (although these constants can differ, as discussed above).
"""
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

	m = unique(broadcast(z -> size(z)[end], Z_val))
	M = unique(broadcast(z -> size(z)[end], Z_train))
	@assert length(m) == 1 "The elements of `Z_val` should be equally replicated; that is, the size of the last dimension in each array in `Z_val` should be constant."
	@assert length(M) == 1 "The elements of `Z_train` should be equally replicated; that is, the size of the last dimension in each array in `Z_train` should be constant."
	M = M[1]
	m = m[1]
	@assert M % m == 0 "The number of replicates in the training data, `M`, should be a multiple of the number of replicates in the validation data, `m`."

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
	tmp = _quietDataLoader((indexdata(Z_train, 1:m), _extractθ(θ_train)), batchsize)
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
		Z_train_current = _quietDataLoader((indexdata(Z_train, replicates[epoch]), _extractθ(θ_train)), batchsize)
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
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestweights(savepath)

    return θ̂
end


# ---- Wrapper function for training multiple estimators over a range of sample sizes ----


"""
	train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M; <keyword args>)

Train several neural estimators, each with the architecture given by `θ̂`, using
the sample sizes given by the vector of integers, `M`.

Each neural estimator is pre-trained with the neural estimator trained for the
previous sample size. That is, if `M = [m₁, m₂]`, with `m₂` > `m₁`, the neural
estimator for sample size `m₂` is pre-trainined with the neural
estimator for sample size `m₁`. By pre-training a series of neural estimators with
progressively larger sample sizes, most of the learning is done with small,
computationally cheap sample sizes. Hence, this approach can be beneficial even
if one is only interested in estimation for a single, large sample.

This method is a wrapper for
`train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T)` and, hence, it
inherits its keyword arguments. Further, certain keyword arguments can be given
as vectors. For instance, if we are training two neural estimators, we can use
a different number of epochs by providing `epochs = [e₁, e₂]`. Other arguments
that allow vectors are `batchsize`, `stopping_epochs`, and `optimiser`.
"""
function train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}; args...)  where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}

	@assert all(M .> 0)
	M = sort(M)

	kwargs = (;args...)
	@assert !haskey(kwargs, :m) "`m` should not be provided with this method of `train`"

	# Create one copy of the estimator θ̂ for each sample size in M
	# If we are using the GPU, we first need to move θ̂ there before creating
	# the estimator, otherwise
	if haskey(kwargs, :use_gpu)
		use_gpu = kwargs.use_gpu
	else
		use_gpu = true
	end
	device = _checkgpu(use_gpu, verbose = true)
	θ̂ = θ̂ |> device
	estimators = [deepcopy(θ̂ ) for _ ∈ eachindex(M)]

	for i ∈ eachindex(M)

		mᵢ = M[i]
		@info "training with m=$(mᵢ)"

		# Pre-train if this is not the first estimator
		if i > 1 Flux.loadparams!(estimators[i], Flux.params(estimators[i-1])) end

		# Modify/check the keyword arguments before passing them onto train().
		# If savepath has been provided in the keyword arguments, modify it with
		# information on the training sample size mᵢ. First, we convert the object
		# args to a named tuple. Then, if savepath was included as an argument
		# and it is not an empty string, we use merge() to replace its given
		# value with a modified version that contains mᵢ. We will pass on this
		# modified version of args to train().
		kwargs = (;args...)
		@assert !haskey(kwargs, :m) "`m` should not be provided with this method of `train`"
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, M)

		estimators[i] = train(estimators[i], θ_train, θ_val, Z_train, indexdata(Z_val, 1:mᵢ); kwargs...)
	end

	return estimators
end


# ---- Helper functions ----


function _modifyargs(kwargs, i, M)
	for arg ∈ [:epochs, :batchsize, :stopping_epochs, :optimiser]
		if haskey(kwargs, arg)
			field = getfield(kwargs, arg)
			@assert length(field) ∈ (1, length(M))
			if length(field) > 1
				kwargs = merge(kwargs, NamedTuple{(arg,)}(field[i]))
			end
		end
	end
	kwargs = Dict(pairs(kwargs)) # convert to Dictionary so that kwargs can be passed to train()
	return kwargs
end


function indexdata(Z::V, m) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
	colons  = ntuple(_ -> (:), N - 1)
	broadcast(z -> z[colons..., m], Z)
end

function indexdata(Z::V, m) where {V <: AbstractVector{G}} where {G <: AbstractGraph}
	broadcast(z -> getgraph(z, m), Z)
end

function _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh, simulate_just_in_time)
	@assert batchsize > 0
	@assert epochs > 0
	@assert stopping_epochs > 0
	@assert epochs_per_Z_refresh > 0
	if simulate_just_in_time && epochs_per_Z_refresh != 1 @error "We cannot simulate the data just-in-time if we aren't refreshing it every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`" end
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

# helper functions to reduce code repetition and improve clarity

function _saveweights(θ̂, savepath, epoch)
	if !ispath(savepath) mkpath(savepath) end
	# return to cpu before serialization
	weights = θ̂ |> cpu |> Flux.params
	networkpath = joinpath(savepath, "network_epoch$epoch.bson")
	@save networkpath weights
end

function _saveweights(θ̂, savepath)
	if !ispath(savepath) mkpath(savepath) end
	# return to cpu before serialization
	weights = θ̂ |> cpu |> Flux.params
	networkpath = joinpath(savepath, "network.bson")
	@save networkpath weights
end


function _saveinfo(loss_per_epoch, train_time, savepath::String; verbose::Bool = true)

	verbose && println("Finished training in $(train_time) seconds")

	# Recall that we initialised the training loss to the initial validation
	# loss. Slightly better to just use the training loss from the second epoch:
	loss_per_epoch[1, 1] = loss_per_epoch[2, 1]

	# Save various quantities of interest (in both .bson and .csv format)
    @save joinpath(savepath, "train_time.bson") train_time
	@save joinpath(savepath, "loss_per_epoch.bson") loss_per_epoch
	CSV.write(joinpath(savepath, "train_time.csv"), Tables.table([train_time]), header = false)
	CSV.write(joinpath(savepath, "loss_per_epoch.csv"), Tables.table(loss_per_epoch), header = false)

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

	m = _numberreplicates(Z)
	Z = Flux.batch(Z)
	Z = Z |> device
	θ = θ |> device
	m = m |> device

	# Compute gradients in such a way that the training loss is also saved.
	# This is equivalent to: gradients = gradient(() -> loss(θ̂(Z), θ), γ)
	ls, back = Zygote.pullback(() -> loss(θ̂(Z, m), θ), γ) # NB here we also pass m to θ̂, since Flux.batch() cannot be differentiated
	gradients = back(one(ls))
	update!(optimiser, γ, gradients)

	# Assuming that loss returns an average, convert it to a sum.
	ls = ls * size(θ)[end]
	return ls
end
