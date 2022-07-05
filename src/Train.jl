# TODO Could add argument to train, verbose::Bool = true, and could do the same for estimate().


_common_kwd_args = """
- `m`: sample sizes (either an `Integer` or a collection of `Integers`).
- `batchsize::Integer = 32`
- `epochs::Integer = 100`: the maximum number of epochs used during training.
- `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
- `loss = mae`: the loss function, which should return an average loss when applied to multiple replicates.
- `optimiser = ADAM(1e-4)`
- `savepath::String = "runs/"`: path to save the trained `θ̂` and other information; if savepath is an empty string (i.e., `""`), nothing is saved.
- `simulate_just_in_time::Bool = false`: should we do "just-in-time" data simulation, which improves memory complexity at the cost of time complexity?
- `stopping_epochs::Integer = 10`: cease training if the risk doesn't improve in `stopping_epochs` epochs.
- `use_gpu::Bool = true`
- `verbose::Bool = true`
"""



"""
	train(θ̂, ξ, P; <keyword args>) where {P <: ParameterConfigurations}

Train the neural estimator `θ̂` by providing the invariant model information `ξ`
needed for the constructor `P` to automatically sample the sets of training and
validation parameters.

# Keyword arguments common to both `train` methods:
$_common_kwd_args

# Simulator keyword arguments only:
- `K::Integer = 10_000`: the number of parameters in the training set; the size of the validation set is `K ÷ 5`.
- `epochs_per_θ_refresh::Integer = 1`: how often to refresh the training parameters; this must be a multiple of `epochs_per_Z_refresh`.
"""
function train(θ̂, ξ, P;
	m,
	# epochs_per_θ_refresh::Integer = 1, # TODO
	# epochs_per_Z_refresh::Integer = 1, # TODO
	loss = Flux.Losses.mae,
	optimiser          = ADAM(1e-4),
    batchsize::Integer = 32,
    epochs::Integer    = 100,
	savepath::String   = "runs/",
	# simulate_just_in_time::Bool = false, # TODO
	stopping_epochs::Integer = 10,
    use_gpu::Bool      = true,
	verbose::Bool      = true,
	K::Integer         = 10_000
	)


	simulate_just_in_time::Bool = false # TODO
	epochs_per_Z_refresh = 1 # TODO
	epochs_per_θ_refresh = 1 # TODO

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh, simulate_just_in_time)

	# TODO better way to enforce P to be a type and specifically a sub-type of ParameterConfigurations?
	@assert P <: ParameterConfigurations
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
	θ_val = P(ξ, K ÷ 5 + 1)
	Z_val = _simulate(θ_val, ξ, m)
	Z_val = _quietDataLoader(Z_val, batchsize)

	# Initialise the loss per epoch matrix.
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	loss_per_epoch   = [initial_val_risk initial_val_risk;]
	verbose && println(" Initial validation risk = $initial_val_risk")

	# Save the initial θ̂. This is to prevent bugs in the case that
	# the initial loss does not improve
	savebool && _saveweights(θ̂, 0, savepath)

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
		epoch_time_train = @elapsed for _ ∈ 1:batches
			parameters = P(ξ, batchsize)
			Z = simulate(parameters, ξ, m)
			θ = parameters.θ
			train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
		end
		train_loss = train_loss / (batchsize * batches) # convert to an average

		epoch_time_val = @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time_train + epoch_time_val, digits = 3)) seconds")
		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			savebool && _saveweights(θ̂, epoch, savepath)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save information, the final θ̂, and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _saveweights(θ̂, epochs, savepath)
	savebool && _savebestweights(savepath)

    return θ̂
end


"""
	train(θ̂, ξ, θ_train::P, θ_val::P; <keyword args>) where {P <: ParameterConfigurations}

Train the neural estimator `θ̂` by providing the training and validation sets
explicitly as `θ_train` and `θ_val`, which are both held fixed during training,
as well as the invariant model information `ξ`.
"""
function train(θ̂, ξ, θ_train::P, θ_val::P;
		m,
		batchsize::Integer = 256,
		epochs_per_Z_refresh::Integer = 1,
		epochs::Integer  = 100,
		loss             = Flux.Losses.mae,
		optimiser        = ADAM(1e-4),
		savepath::String = "runs/",
		simulate_just_in_time::Bool = false,
		stopping_epochs::Integer = 10,
		use_gpu::Bool    = true,
		verbose::Bool    = true
		) where {P <: ParameterConfigurations}

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
	Z_val = _simulate(θ_val, ξ, m)
	Z_val = _quietDataLoader(Z_val, batchsize)
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
	verbose && println(" Initial validation risk = $initial_val_risk")

	# Initialise the loss per epoch matrix (NB just using validation for both for now)
	loss_per_epoch = [initial_val_risk initial_val_risk;]

	# Save the initial θ̂. This is to prevent bugs in the case that the initial
	# risk does not improve
	savebool && _saveweights(θ̂, 0, savepath)

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
				t = @elapsed Z_train = _simulate(θ_train, ξ, m)
				Z_train = _quietDataLoader(Z_train, batchsize)
				verbose && println(" Finished in $(round(t, digits = 3)) seconds")
			end

			# For each batch, update θ̂ and compute the training loss
			epoch_time_train = @elapsed for (Z, θ) in Z_train
			   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end

		else

			# For each batch, update θ̂ and compute the training loss
			epoch_time_simulate = 0.0
			epoch_time_train    = 0.0
			for parameters ∈ _ParameterLoader(θ_train, batchsize = batchsize)
				epoch_time_simulate += @elapsed Z = simulate(parameters, ξ, m)
				θ = parameters.θ
				epoch_time_train    += @elapsed train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
			println("Total time spent simulating data: $(round(epoch_time_simulate, digits = 3)) seconds")

		end
		train_loss = train_loss / size(θ_train, 2)


		epoch_time_val = @elapsed current_val_risk = _lossdataloader(loss, Z_val, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time_train + epoch_time_val, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if current_val_risk <= min_val_risk
			savebool && _saveweights(θ̂, epoch, savepath)
			min_val_risk = current_val_risk
			early_stopping_counter = 0
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save information, the final θ̂, and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _saveweights(θ̂, epochs, savepath)
	savebool && _savebestweights(savepath)

    return θ̂
end



# ---- Helper functions ----

function _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh, simulate_just_in_time)
	@assert batchsize > 0
	@assert epochs > 0
	@assert stopping_epochs > 0
	@assert epochs_per_Z_refresh > 0
	if simulate_just_in_time && epochs_per_Z_refresh != 1 @error "We cannot simulate the data just-in-time if we aren't refreshing it every epoch; please either set ` simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`" end
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
