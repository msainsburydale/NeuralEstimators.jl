_common_kwd_args = """
- `m`: sample sizes (either an `Integer` or a collection of `Integers`).
- `batchsize::Integer = 32`
- `epochs::Integer = 100`: the maximum number of epochs used during training.
- `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
- `loss = mae`: the loss function, which should return an average loss when applied to multiple replicates.
- `optimiser = ADAM(1e-4)`
- `savepath::String = "runs/"`: path to save the trained `θ̂` and other information.
- `stopping_epochs::Integer = 10`: halt training if the risk doesn't improve in `stopping_epochs` epochs.
- `use_gpu::Bool = true`
"""

"""
	train(θ̂, ξ, P; <keyword args>) where {P <: ParameterConfigurations}

Train the neural estimator `θ̂` by providing the objects `ξ` needed for the
constructor `P` to automatically sample the set of training and validation
parameters, which may be refreshed periodically throughout training via the keyword
argument `epochs_per_θ_refresh`.

# Keyword arguments common to both `train` methods:
$_common_kwd_args

# Simulator keyword arguments only:
- `configs_per_epoch::Integer = 10_000`: how many parameters constitute a single epoch.
- `epochs_per_θ_refresh::Integer = 1`: how often to refresh the training parameters; this must be a multiple of `epochs_per_Z_refresh`.
"""
function train(θ̂, Ω, P;
	m,
	# epochs_per_θ_refresh::Integer = 1, # TODO
	# epochs_per_Z_refresh::Integer = 1, # TODO
	loss = Flux.Losses.mae,
	optimiser          = ADAM(1e-4),
    batchsize::Integer = 32,
    epochs::Integer    = 100,
	stopping_epochs::Integer = 10,
    use_gpu::Bool      = true,
    savepath::String   = "runs/",
	configs_per_epoch::Integer  = 10_000
	)

	epochs_per_Z_refresh = 1 # TODO
	epochs_per_θ_refresh = 1 # TODO

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)

	# TODO better way to enforce P to be a type and specifically a sub-type of ParameterConfigurations?
	@assert P <: ParameterConfigurations
	@assert configs_per_epoch > 0
	@assert epochs_per_θ_refresh % epochs_per_Z_refresh  == 0 "`epochs_per_θ_refresh` must be a multiple of `epochs_per_Z_refresh`"

	loss_path = joinpath(savepath, "loss_per_epoch.bson")
	if isfile(loss_path) rm(loss_path) end
	if !ispath(savepath) mkpath(savepath) end

	device = _checkgpu(use_gpu)
    θ̂ = θ̂ |> device
    γ = Flux.params(θ̂)

	println("Simulating validation parameters and validation data...")
	θ_val = P(Ω, configs_per_epoch ÷ 5 + 1)
	Z_val = _simulate(θ_val, m)
	Z_val = _quietDataLoader(Z_val, batchsize)

	# Initialise the loss per epoch matrix.
	print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	loss_per_epoch   = [initial_val_risk initial_val_risk;]
	println(" Initial validation loss = $initial_val_risk")

	# Save the initial θ̂. This is to prevent bugs in the case that
	# the initial loss does not improve
	_saveweights(θ̂, 0, savepath)

	# Number of batches of θ to use for each epoch
	batches = ceil((configs_per_epoch / batchsize))
	@assert batches > 0

	# For loop creates a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.

	local min_val_risk = initial_val_risk # minimum validation loss; monitor this for early stopping
	local early_stopping_counter = 0
	train_time = @elapsed for epoch ∈ 1:epochs

		# For each batch, update θ̂ and save the training loss
		local train_loss = zero(initial_val_risk)
		epoch_time_train = @elapsed for _ ∈ 1:batches
			parameters = P(Ω, batchsize)
			Z = simulate(parameters, m)
			θ = parameters.θ
			train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
		end
		K = (batchsize * batches)
		train_loss = train_loss / K

		epoch_time_val = @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		println("Epoch: $epoch  Validation loss: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time_train + epoch_time_val, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		@save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			_saveweights(θ̂, epoch, savepath)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save information, the final θ̂, and save the best θ̂ as best_network.bson.
	_saveinfo(loss_per_epoch, train_time, savepath)
	_saveweights(θ̂, epochs, savepath)
	_savebestweights(savepath)

    return θ̂
end


"""
	train(θ̂, θ_train::P, θ_val::P; <keyword args>) where {P <: ParameterConfigurations}

Train the neural estimator `θ̂` by providing the training and validation sets
explicitly as `θ_train` and `θ_val`, which are both held fixed during training.
"""
function train(θ̂, θ_train::P, θ_val::P;
	m,
	loss = Flux.Losses.mae,
	optimiser = ADAM(1e-4),
	batchsize::Integer = 256,
	epochs::Integer = 100,
    use_gpu::Bool = true,
    savepath::String = "runs/",
	stopping_epochs::Integer = 10,
	epochs_per_Z_refresh::Integer = 1
	) where {P <: ParameterConfigurations}

	_checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)

	loss_path = joinpath(savepath, "loss_per_epoch.bson")
	if isfile(loss_path) rm(loss_path) end
	if !ispath(savepath) mkpath(savepath) end

	device = _checkgpu(use_gpu)
    θ̂ = θ̂ |> device
    γ = Flux.params(θ̂)

	println("Simulating validation data...")
	Z_val = _simulate(θ_val, m)
	Z_val = _quietDataLoader(Z_val, batchsize)
	print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	println(" Initial validation loss = $initial_val_risk")

	# Initialise the loss per epoch matrix (NB just using validation for both for now)
	loss_per_epoch   = [initial_val_risk initial_val_risk;]

	# Save the initial θ̂. This is to prevent bugs in the case that the initial
	# risk does not improve
	_saveweights(θ̂, 0, savepath)

	# for loops create a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.
	local Z_train
	local min_val_risk = initial_val_risk # minimum validation loss; monitor this for early stopping
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		# If we are not refreshing Z_train every epoch, we must simulate it in
		# its entirety so that it can be used in subsequent epochs.
		if epochs_per_Z_refresh > 1 && (epoch == 1 || (epoch % epochs_per_Z_refresh) == 0)
			print("Simulating training data...")
			Z_train = nothing
			@sync gc()
			t = @elapsed Z_train = _simulate(θ_train, m)
			Z_train = _quietDataLoader(Z_train, batchsize)
			println(" Finished in $(round(t, digits = 3)) seconds")
		end

		# For each batch, update θ̂ and save the training loss
		local train_loss = zero(initial_val_risk)
		epoch_time_train = @elapsed if epochs_per_Z_refresh > 1
			 for (Z, θ) in Z_train
				train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
		else
			for parameters ∈ _ParameterLoader(θ_train, batchsize = batchsize)
				Z = simulate(parameters, m)
				θ = parameters.θ
				train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
		end
		K = size(θ_train, 2)
		train_loss = train_loss / K

		epoch_time_val = @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		println("Epoch: $epoch  Validation loss: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time_train + epoch_time_val, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		@save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			_saveweights(θ̂, epoch, savepath)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save information, the final θ̂, and the best θ̂ as best_network.bson.
	_saveinfo(loss_per_epoch, train_time, savepath)
	_saveweights(θ̂, epochs, savepath)
	_savebestweights(savepath)

    return θ̂
end



# ---- Helper functions ----

function _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
	@assert batchsize > 0
	@assert epochs > 0
	@assert stopping_epochs > 0
	@assert epochs_per_Z_refresh > 0
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

function _saveweights(θ̂, epoch, savepath)
	# return to cpu before serialization
	weights = θ̂ |> cpu |> Flux.params
	networkpath = joinpath(savepath, "network_epoch$(epoch).bson")
	@save networkpath weights
end


function _saveinfo(loss_per_epoch, train_time, savepath::String)

	println("Finished training in $(train_time) seconds")

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
	ls * size(θ)[end]
	return ls
end




# """
# 	train(θ̂, ξ, P; <keyword args>)
# 	train(θ̂, θ_train::P, θ_val::P; <keyword args>) where {P <: ParameterConfigurations}
#
# Train the neural estimator `θ̂` either by;
# - providing the objects `ξ` needed for the constructor `P` to automatically sample the set of training and validation parameters, or
# - providing these parameter sets explicitly as `θ_train` and `θ_val`.
#
# For both methods, the validation parameters and validation data are
# held fixed so that the validation risk is interpretable. There are a number of
# practical considerations to keep in mind: In particular, see [Balancing time and memory complexity](@ref).
#
# # Common keyword arguments:
# - `m`: sample sizes (either an `Integer` or a collection of `Integers`).
# - `loss = mae`: the loss function, which should return an average loss when applied to multiple replicates.
# - `epochs::Integer = 100`: the maximum number of epochs used during training.
# - `stopping_epochs::Integer = 10`: halt training if the risk doesn't improve in `stopping_epochs` epochs.
# - `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
# - `use_gpu::Bool = true`: whether to use the GPU if it is available.
# - `savepath::String = "runs/"`: path to save the trained `θ̂` and other information.
# # Simulator keywork arguments only:
# - `configs_per_epoch::Integer = 10_000`: how many parameters constitute a single epoch.
# - `epochs_per_θ_refresh::Integer = 1`: how often to refresh the training parameters.
# - `optimiser = ADAM(1e-4)`
# - `batchsize::Integer = 32`
# """
# function train(θ̂, Ω, P;
# 	m,
# 	loss = Flux.Losses.mae,
# 	optimiser          = ADAM(1e-4),
#     batchsize::Integer = 32,
#     epochs::Integer    = 100,  # NB has a different interpretation to the usual meaning. Also, should call this max_epochs.
# 	stopping_epochs::Integer = 10,
#     use_gpu::Bool      = true,
#     savepath::String   = "runs/",
# 	configs_per_epoch::Integer  = 10_000
# 	)
#
#
# 	# TODO epochs_per_θ_refresh
# 	# TODO epochs_per_Z_refresh
# 	# TODO need to @assert epochs_per_Z_refresh a multiple of epochs_per_θ_refresh.
#
# 	epochs_per_Z_refresh = 1 # TODO
#     _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
#
# 	# TODO better way to enforce P to be a type and specifically a sub-type of ParameterConfigurations?
# 	@assert P <: ParameterConfigurations
# 	@assert configs_per_epoch > 0
#
# 	loss_path = joinpath(savepath, "loss_per_epoch.bson")
# 	if isfile(loss_path) rm(loss_path) end
# 	if !ispath(savepath) mkpath(savepath) end
#
# 	device = _checkgpu(use_gpu)
#     θ̂ = θ̂ |> device
#     γ = Flux.params(θ̂)
#
# 	println("Simulating validation parameters and validation data...")
# 	θ_val = P(Ω, configs_per_epoch ÷ 5 + 1)
# 	Z_val = _simulate(θ_val, m)
# 	Z_val = _quietDataLoader(Z_val, batchsize)
#
# 	# Initialise the loss per epoch matrix.
# 	print("Computing the initial validation risk...")
# 	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
# 	loss_per_epoch   = [initial_val_risk initial_val_risk;]
# 	println(" Initial validation loss = $initial_val_risk")
#
# 	# Save the initial θ̂. This is to prevent bugs in the case that
# 	# the initial loss does not improve
# 	_saveweights(θ̂, 0, savepath)
#
# 	# Number of batches of θ to use for each epoch
# 	batches = ceil((configs_per_epoch / batchsize))
# 	@assert batches > 0
#
# 	# For loop creates a new scope for the variables that are not present in the
# 	# enclosing scope, and such variables get a new binding in each iteration of
# 	# the loop; circumvent this by declaring local variables.
# 	local min_val_risk = initial_val_risk # minimum validation loss; monitor this for early stopping
# 	local early_stopping_counter = 0
# 	train_time = @elapsed for epoch ∈ 1:epochs
#
# 		# For each batch, update the θ̂ and save the training loss
# 		local ls = zero(initial_val_risk)
# 		epoch_time_train = @elapsed for _ ∈ 1:batches
#
# 			parameters = P(Ω, batchsize)
# 			Z = simulate(parameters, m)
# 			θ = parameters.θ
# 			ls += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
#
# 		end
# 		ls = ls / configs_per_epoch
#
# 		epoch_time_val = @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
# 		loss_per_epoch = vcat(loss_per_epoch, [ls current_val_risk])
# 		println("Epoch: $epoch  Validation loss: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time_train + epoch_time_val, digits = 3)) seconds")
#
# 		# save the loss every epoch in case training is prematurely halted
# 		@save loss_path loss_per_epoch
#
# 		# If the current loss is better than the previous best, save θ̂ and
# 		# update the minimum validation risk; otherwise, add to the early
# 		# stopping counter
# 		if current_val_risk <= min_val_risk
# 			_saveweights(θ̂, epoch, savepath)
# 			min_val_risk = current_val_risk
# 			early_stopping_counter = 0
# 		else
# 			early_stopping_counter += 1
# 			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
# 		end
#
#     end
#
# 	# save information, the final θ̂, and save the best θ̂ as best_network.bson.
# 	_saveinfo(loss_per_epoch, train_time, savepath)
# 	_saveweights(θ̂, epochs, savepath)
# 	_savebestweights(savepath)
#
#     return θ̂
# end
