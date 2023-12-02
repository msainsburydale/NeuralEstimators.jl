#NB This behaviour is important for the implementation of trainx() but unnecessary for the user to know.

# If the number of replicates in `Z_train` is a multiple of the
# number of replicates for each element of `Z_val`, the training data will be
# recycled throughout training. For example, if each
# element of `Z_train` consists of 50 replicates, and each
# element of `Z_val` consists of 10 replicates, the first epoch will use the first
# 10 replicates in `Z_train`, the second epoch uses the next 10 replicates, and so
# on, until the sixth epoch again uses the first 10 replicates. Note that this
# requires the data to be subsettable with the function `subsetdata`.

"""
	train(θ̂, sampler::Function, simulator::Function; ...)
	train(θ̂, θ_train::P, θ_val::P, simulator::Function; ...) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

Train a neural estimator `θ̂`.

The methods cater for different variants of "on-the-fly" simulation.
Specifically, a `sampler` can be provided to continuously sample new parameter
vectors from the prior, and a `simulator` can be provided to continuously
simulate new data conditional on the parameters. If
provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data
(`Z_train` and `Z_val`), they will be held fixed during training.

In all methods, the validation parameters and data are held fixed to reduce noise when evaluating the validation risk.

# Keyword arguments common to all methods:
- `loss = mae`
- `epochs::Integer = 100`
- `batchsize::Integer = 32`
- `optimiser = ADAM(1e-4)`
- `savepath::String = ""`: path to save the neural-network weights during training (as `bson` files) and other information, such as the risk vs epoch (the risk function evaluated over the training and validation sets are saved in the first and second columns of `loss_per_epoch.csv`). If `savepath` is an empty string (default), nothing is saved.
- `stopping_epochs::Integer = 5`: cease training if the risk doesn't improve in this number of epochs.
- `use_gpu::Bool = true`
- `verbose::Bool = true`

# Keyword arguments common to `train(θ̂, sampler, simulator)` and `train(θ̂, θ_train, θ_val, simulator)`:
- `m`: sample sizes (either an `Integer` or a collection of `Integers`). The `simulator` is called as `simulator(θ, m)`.
- `epochs_per_Z_refresh::Integer = 1`: how often to refresh the training data.
- `simulate_just_in_time::Bool = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

# Keyword arguments unique to `train(θ̂, sampler, simulator)`:
- `K::Integer = 10000`: number of parameter vectors in the training set; the size of the validation set is `K ÷ 5`.
- `ξ = nothing`: an arbitrary collection of objects that are fixed (e.g., distance matrices). If provided, the parameter sampler is called as `sampler(K, ξ)`; otherwise, the parameter sampler will be called as `sampler(K)`. Can also be provided as `xi`.
- `epochs_per_θ_refresh::Integer = 1`: how often to refresh the training parameters. Must be a multiple of `epochs_per_Z_refresh`. Can also be provided as `epochs_per_theta_refresh`.

# Examples
```
using NeuralEstimators
using Flux
import NeuralEstimators: simulate

# parameter sampler
function sampler(K)
	μ = randn(K) # Gaussian prior
	σ = rand(K)  # Uniform prior
	θ = hcat(μ, σ)'
	return θ
end

# data simulator
simulator(θ_matrix, m) = [θ[1] .+ θ[2] * randn(1, m) for θ ∈ eachcol(θ_matrix)]

# architecture
p = length(Ω)   # number of parameters in the statistical model
w = 32          # width of each layer
ψ = Chain(Dense(1, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p))
θ̂ = DeepSet(ψ, ϕ)

# number of independent replicates to use during training
m = 15

# training: full simulation on-the-fly
θ̂ = train(θ̂, sampler, simulate, m = m, epochs = 5)

# training: simulation on-the-fly with fixed parameters
K = 10000
θ_train = sampler(K)
θ_val   = sampler(K ÷ 5)
θ̂ 		 = train(θ̂, θ_train, θ_val, simulate, m = m, epochs = 5)

# training: fixed parameters and fixed data
Z_train = simulate(θ_train, m)
Z_val   = simulate(θ_val, m)
θ̂ 		 = train(θ̂, θ_train, θ_val, Z_train, Z_val, epochs = 5)
```
"""
function train end

function train(θ̂, sampler, simulator;
	m,
	ξ = nothing, xi = nothing, 
	epochs_per_θ_refresh::Integer = 1, epochs_per_theta_refresh::Integer = 1,
	epochs_per_Z_refresh::Integer = 1,
	simulate_just_in_time::Bool = false,
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

	# Check duplicated arguments that are needed so that the R interface uses ASCII characters only
	@assert isnothing(ξ) || isnothing(xi) "Only one of `ξ` or `xi` should be provided"
	@assert epochs_per_θ_refresh == 1 || epochs_per_theta_refresh == 1 "Only one of `epochs_per_θ_refresh` or `epochs_per_theta_refresh` should be provided"
	if !isnothing(xi) ξ = xi end
	if epochs_per_theta_refresh != 1 epochs_per_θ_refresh = epochs_per_theta_refresh == 1 end

    _checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)

	@assert K > 0
	@assert epochs_per_θ_refresh > 0
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

	verbose && println("Sampling the validation set...")
	θ_val   = isnothing(ξ) ? sampler(K ÷ 5 + 1) : sampler(K ÷ 5 + 1, ξ)
	val_set = _quietDataLoader((simulator(θ_val, m), _extractθ(θ_val)), batchsize)

	# Initialise the loss per epoch matrix.
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, val_set, θ̂, device)
	loss_per_epoch   = [initial_val_risk initial_val_risk;]
	verbose && println(" Initial validation risk = $initial_val_risk")

	# Save initial θ̂ (prevents bugs in the case that the risk does not improve)
	savebool && _saveweights(θ̂, savepath, 0)

	# Number of batches of θ in each epoch
	batches = ceil((K / batchsize))

	store_entire_train_set = epochs_per_Z_refresh > 1 || !simulate_just_in_time

	# For loops create a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.
	local θ_train
	local train_set
	local min_val_risk = initial_val_risk # minimum validation loss, monitored for early stopping
	local early_stopping_counter = 0
	train_time = @elapsed for epoch ∈ 1:epochs

		# Initialise the current training loss to zero
		train_loss = zero(initial_val_risk)

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
				t = @elapsed train_set = _constructset(simulator, θ_train, m, batchsize)
				verbose && println(" Finished in $(round(t, digits = 3)) seconds")
			end

			# For each batch, update θ̂ and compute the training loss
			epoch_time = @elapsed for (Z, θ) in train_set
			   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end

		else

			# Full simulation on the fly and just-in-time sampling:
			epoch_time = @elapsed for _ ∈ 1:batches
				parameters = isnothing(ξ) ? sampler(batchsize) : sampler(batchsize, ξ)
				Z = simulator(parameters, m)
				θ = _extractθ(parameters)
				train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
		end

		# Convert training loss to an average
		train_loss = train_loss / (batchsize * batches)

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, val_set, θ̂, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_loss current_val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_loss, digits = 3))  Validation risk: $(round(current_val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")
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


function train(θ̂, θ_train::P, θ_val::P, simulator;
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
	val_set = _constructset(simulator, θ_val, m, batchsize)
	verbose && print("Computing the initial validation risk...")
	initial_val_risk = _lossdataloader(loss, val_set, θ̂, device)
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
	store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

	local train_set
	local min_val_risk = initial_val_risk
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		train_loss = zero(initial_val_risk)

		if store_entire_train_set

			# Simulate new training data if needed
			if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
				verbose && print("Simulating training data...")
				train_set = nothing
				@sync gc()
				t = @elapsed train_set = _constructset(simulator, θ_train, m, batchsize)
				verbose && println(" Finished in $(round(t, digits = 3)) seconds")
			end

			# For each batch, update θ̂ and compute the training loss
			epoch_time = @elapsed for (Z, θ) in train_set
			   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end

		else

			# For each batch, update θ̂ and compute the training loss
			epoch_time_simulate = 0.0
			epoch_time    = 0.0
			for parameters ∈ _ParameterLoader(θ_train, batchsize = batchsize)
				epoch_time_simulate += @elapsed Z = simulator(parameters, m)
				θ = _extractθ(parameters)
				epoch_time += @elapsed train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
			end
			verbose && println("Total time spent simulating data: $(round(epoch_time_simulate, digits = 3)) seconds")
			epoch_time += epoch_time_simulate

		end
		train_loss = train_loss / size(θ_train, 2)

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, val_set, θ̂, device)
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

		# Training data recycles every x epochs
		if subsetbool
			x = M ÷ m
			replicates = repeat([(1:m) .+ i*m for i ∈ 0:(x - 1)], outer = ceil(Integer, epochs/x))
		end
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
	val_set = _quietDataLoader((Z_val, _extractθ(θ_val)), batchsize)
	initial_val_risk = _lossdataloader(loss, val_set, θ̂, device)
	verbose && println(" Initial validation risk = $initial_val_risk")

	verbose && print("Computing the initial training risk...")
	tmp = subsetbool ? subsetdata(Z_train, 1:m) : Z_train
	tmp = _quietDataLoader((tmp, _extractθ(θ_train)), batchsize)
	initial_train_risk = _lossdataloader(loss, tmp, θ̂, device)
	verbose && println(" Initial training risk = $initial_train_risk")

	# Initialise the loss per epoch matrix and save the initial estimator
	loss_per_epoch = [initial_train_risk initial_val_risk;]
	savebool && _saveweights(θ̂, savepath, 0)

	local min_val_risk = initial_val_risk
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		train_loss = zero(initial_train_risk)

		# For each batch update θ̂ and compute the training loss
		train_set = subsetbool ? subsetdata(Z_train, replicates[epoch]) : Z_train
		train_set = _quietDataLoader((train_set, _extractθ(θ_train)), batchsize)
		epoch_time = @elapsed for (Z, θ) in train_set
		   train_loss += _updatebatch!(θ̂, Z, θ, device, loss, γ, optimiser)
		end
		train_loss = train_loss / size(θ_train, 2)

		epoch_time += @elapsed current_val_risk = _lossdataloader(loss, val_set, θ̂, device)
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

#TODO reduce code repetition

"""
	trainx(θ̂, sampler::Function, simulator::Function, m::Vector{Integer}; ...)
	trainx(θ̂, θ_train, θ_val, simulator::Function, m::Vector{Integer}; ...)
	trainx(θ̂, θ_train, θ_val, Z_train, Z_val, m::Vector{Integer}; ...)
	trainx(θ̂, θ_train, θ_val, Z_train::V, Z_val::V; ...) where {V <: AbstractVector{AbstractVector{Any}}}

A wrapper around `train()` to construct neural estimators for different sample sizes.

The positional argument `m` specifies the desired sample sizes.
Each estimator is pre-trained with the estimator for the previous sample size.
For example, if `m = [m₁, m₂]`, the estimator for sample size `m₂` is
pre-trained with the estimator for sample size `m₁`.

The method for `Z_train` and `Z_val` subsets the data using
`subsetdata(Z, 1:mᵢ)` for each `mᵢ ∈ m`. The method for `Z_train::V` and
`Z_val::V` trains an estimator for each element of `Z_train::V` and `Z_val::V`
and, hence, it does not need to invoke `subsetdata()`, which can be slow or
difficult to define in some cases (e.g., for graphical data). Note that, in this
case, `m` is inferred from the data.

The keyword arguments inherit from `train()`. The keyword arguments `epochs`,
`batchsize`, `stopping_epochs`, and `optimiser` can each be given as vectors.
For example, if we are training two estimators, we can use a different number of
epochs for each estimator by providing `epochs = [epoch₁, epoch₂]`.
"""
function trainx end


function trainx(θ̂, P, simulator, M; args...)

	@assert !(typeof(θ̂) <: Vector) # check that θ̂ is not a vector of estimators, which is common error if one calls trainx() on the output of a previous call to trainx()

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


function trainx(θ̂, θ_train::P, θ_val::P, simulator, M; args...)  where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	@assert !(typeof(θ̂) <: Vector) # check that θ̂ is not a vector of estimators, which is common error if one calls trainx() on the output of a previous call to trainx()

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


# This is for when the data CAN be easily subsetted
function trainx(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}; args...)  where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}

	@assert !(typeof(θ̂) <: Vector) # check that θ̂ is not a vector of estimators, which is common error if one calls trainx() on the output of a previous call to trainx()

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

# This method is for when the data CANNOT be easily subsetted, so another layer of vectors is needed
function trainx(θ̂, θ_train::P, θ_val::P, Z_train::V, Z_val::V; args...) where {V <: AbstractVector{S}} where {S <: Union{V₁, Tuple{V₁, V₂}}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

	@assert !(typeof(θ̂) <: Vector) # check that θ̂ is not a vector of estimators, which is common error if one calls trainx() on the output of a previous call to trainx()

	@assert length(Z_train) == length(Z_val)
	E = length(Z_train) # number of estimators

	kwargs = (;args...)
	verbose = _checkargs_trainx(kwargs)

	# Create a copy of θ̂ for each sample size
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

function _checkargs_trainx(kwargs)
	@assert !haskey(kwargs, :m) "Please provide the number of independent replicates, `m`, as a positional argument (i.e., provide the argument simply as `trainx(..., m)` rather than `trainx(..., m = m)`)."
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

#TODO Surely there is a better way of dispatching here...
function _updatebatch!(θ̂::Union{GNN, PointEstimator{<:GNN}, IntervalEstimator{<:GNN}, IntervalEstimatorCompactPrior{<:GNN}, PointIntervalEstimator{<:GNN}}, Z, θ, device, loss, γ, optimiser)

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
_simulate(simulator, params::P, m) where {P <: Union{AbstractMatrix, ParameterConfigurations}} = (simulator(params, m), _extractθ(params))
_constructset(simulator, params::P, m, batchsize)  where {P <: Union{AbstractMatrix, ParameterConfigurations}} = _quietDataLoader(_simulate(simulator, params, m), batchsize)
