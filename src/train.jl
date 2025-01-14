# - `optimiser`: An Optimisers.jl optimisation rule, using `Adam()` by default. When the training data and/or parameters are held fixed, the default is to use L₂ regularisation with penalty coefficient λ=1e-4, so that `optimiser = Flux.setup(OptimiserChain(WeightDecay(1e-4), Adam()), θ̂)`. Otherwise, when the training data and parameters are simulated "on the fly", by default no regularisation is used, so that `optimiser = Flux.setup(Adam(), θ̂)`.
# TODO savepath::String = "" -> savepath::Union{String,Nothing} = nothing
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
- `loss = mae`: loss function to evaluate performance. For some classes of estimators (e.g., `QuantileEstimator` and `RatioEstimator`), the loss function does not need to be specified.
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
- `optimiser = ADAM()`: optimisation algorithm used to update the neural-network parameters.
- `savepath::String = ""`: path to save the trained estimator and other information; if an empty string (default), nothing is saved. Otherwise, the neural-network parameters (i.e., the weights and biases) will be saved during training as `bson` files; the risk function evaluated over the training and validation sets will also be saved, in the first and second columns of `loss_per_epoch.csv`, respectively; the best parameters (as measured by validation risk) will be saved as `best_network.bson`.
- `stopping_epochs = 5`: cease training if the risk doesn't improve in this number of epochs.
- `use_gpu = true`: flag indicating whether to use a GPU if one is available.
- `verbose = true`: flag indicating whether information, including empirical risk values and timings, should be printed to the console during training.

# Keyword arguments common to `train(θ̂, sampler, simulator)` and `train(θ̂, θ_train, θ_val, simulator)`:
- `m`: sample sizes (either an `Integer` or a collection of `Integers`). The `simulator` is called as `simulator(θ, m)`.
- `epochs_per_Z_refresh = 1`: the number of passes to make through the training set before the training data are refreshed.
- `simulate_just_in_time = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.

# Keyword arguments unique to `train(θ̂, sampler, simulator)`:
- `K = 10000`: number of parameter vectors in the training set; the size of the validation set is `K ÷ 5`.
- `ξ = nothing`: an arbitrary collection of objects that, if provided, will be passed to the parameter sampler as `sampler(K, ξ)`; otherwise, the parameter sampler will be called as `sampler(K)`. Can also be provided as `xi`.
- `epochs_per_θ_refresh = 1`: the number of passes to make through the training set before the training parameters are refreshed. Must be a multiple of `epochs_per_Z_refresh`. Can also be provided as `epochs_per_theta_refresh`.

# Examples
```
using NeuralEstimators, Flux

function sampler(K)
	μ = randn(K) # Gaussian prior
	σ = rand(K)  # Uniform prior
	θ = hcat(μ, σ)'
	return θ
end

function simulator(θ_matrix, m)
	[θ[1] .+ θ[2] * randn(1, m) for θ ∈ eachcol(θ_matrix)]
end

# architecture
d = 1   # dimension of each replicate
p = 2   # number of parameters in the statistical model
ψ = Chain(Dense(1, 32, relu), Dense(32, 32, relu))
ϕ = Chain(Dense(32, 32, relu), Dense(32, p))
θ̂ = DeepSet(ψ, ϕ)

# number of independent replicates to use during training
m = 15

# training: full simulation on-the-fly
θ̂ = train(θ̂, sampler, simulator, m = m, epochs = 5)

# training: simulation on-the-fly with fixed parameters
K = 10000
θ_train = sampler(K)
θ_val   = sampler(K ÷ 5)
θ̂       = train(θ̂, θ_train, θ_val, simulator, m = m, epochs = 5)

# training: fixed parameters and fixed data
Z_train = simulator(θ_train, m)
Z_val   = simulator(θ_val, m)
θ̂       = train(θ̂, θ_train, θ_val, Z_train, Z_val, epochs = 5)
```
"""
function train end

#NB This behaviour is important for the implementation of trainx() but unnecessary for the user to know.
# If the number of replicates in `Z_train` is a multiple of the
# number of replicates for each element of `Z_val`, the training data will be
# recycled throughout training. For example, if each
# element of `Z_train` consists of 50 replicates, and each
# element of `Z_val` consists of 10 replicates, the first epoch will use the first
# 10 replicates in `Z_train`, the second epoch uses the next 10 replicates, and so
# on, until the sixth epoch again uses the first 10 replicates. Note that this
# requires the data to be subsettable with the function `subsetdata`.

function _train(θ̂, sampler, simulator;
	m,
	ξ = nothing, xi = nothing,
	epochs_per_θ_refresh::Integer = 1, epochs_per_theta_refresh::Integer = 1,
	epochs_per_Z_refresh::Integer = 1,
	simulate_just_in_time::Bool = false,
	loss = Flux.Losses.mae,
	optimiser          = Flux.setup(Flux.Adam(), θ̂),
	# optimiser          = Flux.Adam(),
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
	if epochs_per_theta_refresh != 1 epochs_per_θ_refresh = epochs_per_theta_refresh end

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

	verbose && println("Sampling the validation set...")
	θ_val   = isnothing(ξ) ? sampler(K ÷ 5 + 1) : sampler(K ÷ 5 + 1, ξ)
	val_set = _constructset(θ̂, simulator, θ_val, m, batchsize)

	# Initialise the loss per epoch matrix
	verbose && print("Computing the initial validation risk...")
	val_risk = _risk(θ̂, loss, val_set, device)
	loss_per_epoch   = [val_risk val_risk;]
	verbose && println(" Initial validation risk = $val_risk")

	# Save initial θ̂
	savebool && _savestate(θ̂, savepath, 0)

	# Number of batches of θ in each epoch
	batches = ceil((K / batchsize))

	store_entire_train_set = epochs_per_Z_refresh > 1 || !simulate_just_in_time

	# For loops create a new scope for the variables that are not present in the
	# enclosing scope, and such variables get a new binding in each iteration of
	# the loop; circumvent this by declaring local variables.
	local θ̂_best = deepcopy(θ̂)
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
				t = @elapsed train_set = _constructset(θ̂, simulator, θ_train, m, batchsize)
				verbose && println(" Finished in $(round(t, digits = 3)) seconds")
			end

			# For each batch, update θ̂ and compute the training risk
			epoch_time = @elapsed train_risk = _risk(θ̂, loss, train_set, device, optimiser)

		else
			# Full simulation on the fly and just-in-time sampling
			train_risk = []
			epoch_time = @elapsed for _ ∈ 1:batches
				θ = isnothing(ξ) ? sampler(batchsize) : sampler(batchsize, ξ)
				set = _constructset(θ̂, simulator, θ, m, batchsize)
				rsk = _risk(θ̂, loss, set, device, optimiser)
				push!(train_risk, rsk)
			end
			train_risk = mean(train_risk)
		end

		epoch_time += @elapsed val_risk = _risk(θ̂, loss, val_set, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")
		savebool && @save loss_path loss_per_epoch

		# If the current risk is better than the previous best, save θ̂ and
		# update the minimum validation risk; otherwise, add to the early
		# stopping counter
		if val_risk <= min_val_risk
			savebool && _savestate(θ̂, savepath, epoch)
			min_val_risk = val_risk
			early_stopping_counter = 0
			θ̂_best = deepcopy(θ̂)
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestmodel(savepath)

	# TODO if the user has relied on using train() as a mutating function, the optimal estimator will not be returned. Can I set θ̂ = θ̂_best to fix this? This also ties in with the other TODO down below above trainx(), regarding which device the estimator is on at the end of training.

    return θ̂_best
end

function _train(θ̂, θ_train::P, θ_val::P, simulator;
		m,
		batchsize::Integer = 32,
		epochs_per_Z_refresh::Integer = 1,
		epochs::Integer  = 100,
		loss             = Flux.Losses.mae,
		optimiser          = Flux.setup(Flux.Adam(), θ̂),
		# optimiser        = Flux.setup(OptimiserChain(WeightDecay(1e-4), Flux.Adam()), θ̂),
		# optimiser        = Flux.Adam(),
		savepath::String = "",
		simulate_just_in_time::Bool = false,
		stopping_epochs::Integer = 5,
		use_gpu::Bool    = true,
		verbose::Bool    = true
		) where {P <: Union{AbstractMatrix, ParameterConfigurations}}

	_checkargs(batchsize, epochs, stopping_epochs, epochs_per_Z_refresh)
	if simulate_just_in_time && epochs_per_Z_refresh != 1
		@error "We cannot simulate the data just-in-time if we aren't refreshing the data every epoch; please either set `simulate_just_in_time = false` or `epochs_per_Z_refresh = 1`"
	end

	savebool = savepath != "" # turn off saving if savepath is an empty string
	if savebool
		loss_path = joinpath(savepath, "loss_per_epoch.bson")
		if isfile(loss_path) rm(loss_path) end
		if !ispath(savepath) mkpath(savepath) end
	end

	device = _checkgpu(use_gpu, verbose = verbose)
    θ̂ = θ̂ |> device

	verbose && println("Simulating validation data...")
	val_set = _constructset(θ̂, simulator, θ_val, m, batchsize)
	verbose && print("Computing the initial validation risk...")
	val_risk = _risk(θ̂, loss, val_set, device)
	verbose && println(" Initial validation risk = $val_risk")

	# Initialise the loss per epoch matrix (NB just using validation for both for now)
	loss_per_epoch = [val_risk val_risk;]

	# Save initial θ̂
	savebool && _savestate(θ̂, savepath, 0)

	# We may simulate Z_train in its entirety either because (i) we
	# want to avoid the overhead of simulating continuously or (ii) we are
	# not refreshing Z_train every epoch so we need it for subsequent epochs.
	# Either way, store this decision in a variable.
	store_entire_train_set = !simulate_just_in_time || epochs_per_Z_refresh != 1

	local θ̂_best = deepcopy(θ̂)
	local train_set
	local min_val_risk = val_risk
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		sim_time = 0.0
		if store_entire_train_set
			# Simulate new training data if needed
			if epoch == 1 || (epoch % epochs_per_Z_refresh) == 0
				verbose && print("Simulating training data...")
				train_set = nothing
				@sync gc()
				sim_time = @elapsed train_set = _constructset(θ̂, simulator, θ_train, m, batchsize)
				verbose && println(" Finished in $(round(sim_time, digits = 3)) seconds")
			end
			# Update θ̂ and compute the training risk
			epoch_time = @elapsed train_risk = _risk(θ̂, loss, train_set, device, optimiser)
		else
			# Update θ̂ and compute the training risk
			epoch_time = 0.0
			train_risk = []

			for θ ∈ _ParameterLoader(θ_train, batchsize = batchsize)
				sim_time   += @elapsed set = _constructset(θ̂, simulator, θ, m, batchsize)
				epoch_time += @elapsed rsk = _risk(θ̂, loss, set, device, optimiser)
				push!(train_risk, rsk)
			end
			verbose && println("Total time spent simulating data: $(round(sim_time, digits = 3)) seconds")
			train_risk = mean(train_risk)
		end
		epoch_time += sim_time

		# Compute the validation risk and report to the user
		epoch_time += @elapsed val_risk = _risk(θ̂, loss, val_set, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current risk is better than the previous best, save θ̂ and
		# update the minimum validation risk
		if val_risk <= min_val_risk
			savebool && _savestate(θ̂, savepath, epoch)
			min_val_risk = val_risk
			early_stopping_counter = 0
			θ̂_best = deepcopy(θ̂)
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end
    end

	# save key information and save the best θ̂ as best_network.bson.
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestmodel(savepath)

    return θ̂_best
end

function _train(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T;
		batchsize::Integer = 32,
		epochs::Integer  = 100,
		loss             = Flux.Losses.mae,
		optimiser          = Flux.setup(Flux.Adam(), θ̂),
		# optimiser        = Flux.setup(OptimiserChain(WeightDecay(1e-4), Flux.Adam()), θ̂),
		# optimiser        = Flux.Adam(),
		savepath::String = "",
		stopping_epochs::Integer = 5,
		use_gpu::Bool    = true,
		verbose::Bool    = true
		) where {T, P <: Union{Tuple, AbstractMatrix, ParameterConfigurations}}

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

	verbose && print("Computing the initial validation risk...")
	val_set = _constructset(θ̂, Z_val, θ_val, batchsize)
	val_risk = _risk(θ̂, loss, val_set, device)
	verbose && println(" Initial validation risk = $val_risk")

	verbose && print("Computing the initial training risk...")
	Z̃ = subsetbool ? subsetdata(Z_train, 1:m) : Z_train
	Z̃ = _constructset(θ̂, Z̃, θ_train, batchsize)
	initial_train_risk = _risk(θ̂, loss, Z̃, device)
	verbose && println(" Initial training risk = $initial_train_risk")

	# Initialise the loss per epoch matrix and save the initial estimator
	loss_per_epoch = [initial_train_risk val_risk;]
	savebool && _savestate(θ̂, savepath, 0)

	local θ̂_best = deepcopy(θ̂)
	local min_val_risk = val_risk
	local early_stopping_counter = 0
	train_time = @elapsed for epoch in 1:epochs

		# For each batch update θ̂ and compute the training loss
		Z̃_train = subsetbool ? subsetdata(Z_train, replicates[epoch]) : Z_train
		train_set = _constructset(θ̂, Z̃_train, θ_train, batchsize)
		epoch_time = @elapsed train_risk = _risk(θ̂, loss, train_set, device, optimiser)

		epoch_time += @elapsed val_risk = _risk(θ̂, loss, val_set, device)
		loss_per_epoch = vcat(loss_per_epoch, [train_risk val_risk])
		verbose && println("Epoch: $epoch  Training risk: $(round(train_risk, digits = 3))  Validation risk: $(round(val_risk, digits = 3))  Run time of epoch: $(round(epoch_time, digits = 3)) seconds")

		# save the loss every epoch in case training is prematurely halted
		savebool && @save loss_path loss_per_epoch

		# If the current loss is better than the previous best, save θ̂ and update the minimum validation risk
		if val_risk <= min_val_risk
			savebool && _savestate(θ̂, savepath, epoch)
			min_val_risk = val_risk
			early_stopping_counter = 0
			θ̂_best = deepcopy(θ̂)
		else
			early_stopping_counter += 1
			early_stopping_counter > stopping_epochs && verbose && (println("Stopping early since the validation loss has not improved in $stopping_epochs epochs"); break)
		end

    end

	# save key information
	savebool && _saveinfo(loss_per_epoch, train_time, savepath, verbose = verbose)
	savebool && _savebestmodel(savepath)

    return θ̂_best
end

# General fallback
train(args...; kwargs...) = _train(args...; kwargs...)

# Wrapper functions for specific types of neural estimators
function train(θ̂::Union{IntervalEstimator, QuantileEstimatorDiscrete}, args...; kwargs...)

	# Get the keyword arguments
	kwargs = (;kwargs...)

	# Define the loss function based on the given probabiltiy levels
	τ = Float32.(θ̂.probs)
	# Determine if we need to move τ to the GPU
	use_gpu = haskey(kwargs, :use_gpu) ? kwargs.use_gpu : true
	device  = _checkgpu(use_gpu, verbose = false)
	τ = device(τ)
	# Define the loss function
	qloss = (θ̂, θ) -> quantileloss(θ̂, θ, τ)

	# Notify the user if "loss" is in the keyword arguments
	if haskey(kwargs, :loss)
		@info "The keyword argument `loss` is not required when training a $(typeof(θ̂)), since in this case the quantile loss is always used"
	end
	# Add our quantile loss to the list of keyword arguments
	kwargs = merge(kwargs, (loss = qloss,))

	# Train the estimator
	_train(θ̂, args...; kwargs...)
end

function train(θ̂::QuantileEstimatorContinuous, args...; kwargs...)
	# We define the loss function in the method _risk(θ̂::QuantileEstimatorContinuous)
	# Here, just notify the user if they've assigned a loss function
	kwargs = (;kwargs...)
	if haskey(kwargs, :loss)
		@info "The keyword argument `loss` is not required when training a $(typeof(θ̂)), since in this case the quantile loss is always used"
	end
	_train(θ̂, args...; kwargs...)
end

function train(θ̂::RatioEstimator, args...; kwargs...)

	# Get the keyword arguments and assign the loss function
	kwargs = (;kwargs...)
	if haskey(kwargs, :loss)
		@info "The keyword argument `loss` is not required when training a $(typeof(θ̂)), since in this case the binary cross-entropy (log) loss is always used"
	end
	# kwargs = merge(kwargs, (loss = Flux.logitbinarycrossentropy,))
	_train(θ̂, args...; kwargs...)
end

# ---- Lower level functions ----

# Wrapper function that constructs a set of input and outputs (usually simulated data and corresponding true parameters)
function _constructset(θ̂, simulator::Function, θ::P, m, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	Z = simulator(θ, m)
	_constructset(θ̂, Z, θ, batchsize)
end
function _constructset(θ̂, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	Z = ZtoFloat32(Z)
	θ = θtoFloat32(_extractθ(θ))
	_DataLoader((Z, θ), batchsize)
end
function _constructset(θ̂::RatioEstimator, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	Z = ZtoFloat32(Z)
	θ = θtoFloat32(_extractθ(θ))

	# Size of data set
	K = length(Z) # should equal size(θ, 2)

	# Create independent pairs
	θ̃ = subsetparameters(θ, shuffle(1:K))
	Z̃ = Z # NB memory inefficient to replicate the data in this way, would be better to use a view or similar

	# Combine dependent and independent pairs
	Z = vcat(Z, Z̃)
	θ = hcat(θ, θ̃)

	# Create class labels for output
	labels = [:dependent, :independent]
	output = onehotbatch(repeat(labels, inner = K), labels)[1:1, :]

	# Shuffle everything in case batching isn't shuffled properly downstrean
	idx = shuffle(1:2K)
	Z = Z[idx]
	θ = θ[:, idx]
	output = output[1:1, idx]

	# Combine data and parameters into a single tuple
	input = (Z, θ)

	_DataLoader((input, output), batchsize)
end
function _constructset(θ̂::QuantileEstimatorDiscrete, Z, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	Z = ZtoFloat32(Z)
	θ = θtoFloat32(_extractθ(θ))

	i = θ̂.i
	if isnothing(i)
		input  = Z
		output = θ
	else
		@assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (θ̂.i = $(θ̂.i))"
		θᵢ  = θ[i:i, :]
		θ₋ᵢ = θ[Not(i), :]
		input  = (Z, θ₋ᵢ) # "Tupleise" the input
		output = θᵢ
	end

	_DataLoader((input, output), batchsize)
end
function _constructset(θ̂::QuantileEstimatorContinuous, Zτ, θ::P, batchsize) where {P <: Union{AbstractMatrix, ParameterConfigurations}}
	θ = θtoFloat32(_extractθ(θ))
	Z, τ = Zτ
	Z = ZtoFloat32(Z)
	τ = ZtoFloat32.(τ)

	i = θ̂.i
	if isnothing(i)
		input = (Z, τ)
		output = θ
	else
		@assert size(θ, 1) >= i "The number of parameters in the model (size(θ, 1) = $(size(θ, 1))) must be at least as large as the value of i stored in the estimator (θ̂.i = $(θ̂.i))"
		θᵢ  = θ[i:i, :]
		θ₋ᵢ = θ[Not(i), :]
		# Combine each θ₋ᵢ with the corresponding vector of
		# probability levels, which requires repeating θ₋ᵢ appropriately
		θ₋ᵢτ = map(eachindex(τ)) do k
			τₖ = τ[k]
			θ₋ᵢₖ = repeat(θ₋ᵢ[:, k:k], inner = (1, length(τₖ)))
			vcat(θ₋ᵢₖ, τₖ')
		end
		input  = (Z, θ₋ᵢτ)   # "Tupleise" the input
		output = θᵢ
	end

	_DataLoader((input, output), batchsize)
end

# Computes the risk function in a memory-safe manner, optionally updating the
# neural-network parameters using stochastic gradient descent
function _risk(θ̂, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
		k = size(output)[end]
		if !isnothing(optimiser)
			# NB storing the loss in this way is efficient, but it means that
			# the final training risk that we report for each epoch is slightly inaccurate
			# (since the neural-network parameters are updated after each batch). It would be more
			# accurate (but less efficient) if we computed the training risk once again
			# at the end of each epoch, like we do for the validation risk... might add
			# an option for this in the future, but will leave it for now.

			# "Implicit" style used by Flux <= 0.14
			# γ = Flux.params(θ̂)
			# ls, ∇ = Flux.withgradient(() -> loss(θ̂(input), output), γ)
			# update!(optimiser, γ, ∇)

			# "Explicit" style required by Flux >= 0.15
			ls, ∇ = Flux.withgradient(θ̂ -> loss(θ̂(input), output), θ̂)
			update!(optimiser, θ̂, ∇[1])
		else
			ls = loss(θ̂(input), output)
		end
        # Assuming loss returns an average, convert to a sum and add to total
		sum_loss += ls * k
        K +=  k
    end

    return cpu(sum_loss/K)
end

function _risk(θ̂::RatioEstimator, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
        input, output = input |> device, output |> device
		k = size(output)[end]
		if !isnothing(optimiser)
			ls, ∇ = Flux.withgradient(θ̂ -> Flux.logitbinarycrossentropy(θ̂.deepset(input), output), θ̂)
			update!(optimiser, θ̂, ∇[1])
		else
			ls = Flux.logitbinarycrossentropy(θ̂.deepset(input), output)
		end
        # Assuming loss returns an average, convert to a sum and add to total
		sum_loss += ls * k
        K +=  k
    end

    return cpu(sum_loss/K)
end

function _risk(θ̂::QuantileEstimatorContinuous, loss, set::DataLoader, device, optimiser = nothing)
    sum_loss = 0.0f0
    K = 0
    for (input, output) in set
		k = size(output)[end]
		input, output = input |> device, output |> device

		if isnothing(θ̂.i)
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

		# repeat τ and θ to facilitate broadcasting and indexing
		# note that repeat() cannot be differentiated by Zygote
		p = size(output, 1)
		@ignore_derivatives τ = repeat(τ, inner = (p, 1))
		@ignore_derivatives output = repeat(output, inner = (size(τ, 1) ÷ p, 1))

		if !isnothing(optimiser)

			# "Implicit" style used by Flux <= 0.14
			# γ = Flux.params(θ̂)
			# ls, ∇ = Flux.withgradient(() -> quantileloss(θ̂(input), output, τ), γ)
			# update!(optimiser, γ, ∇)

			# "Explicit" style required by Flux >= 0.15
			ls, ∇ = Flux.withgradient(θ̂ -> quantileloss(θ̂(input), output, τ), θ̂)
			update!(optimiser, θ̂, ∇[1])
		else
			ls = quantileloss(θ̂(input), output, τ)
		end
        # Assuming loss returns an average, convert to a sum and add to total
		sum_loss += ls * k
        K +=  k
    end
    return cpu(sum_loss/K)
end

# ---- Wrapper function for training multiple estimators over a range of sample sizes ----

#TODO (not sure what we want do about the following behaviour, need to think about it): If called as est = trainx(est) then est will be on the GPU; if called as trainx(est) then est will not be on the GPU. Note that the same thing occurs for train(). That is, when the function is treated as mutating, then the estimator will be on the same device that was used during training; otherwise, it will be on whichever device it was when input to the function. Need consistency to improve user experience.
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
For example, if training two estimators, one may use a different number of
epochs for each estimator by providing `epochs = [epoch₁, epoch₂]`.
"""
function trainx end

function _trainx(θ̂; sampler = nothing, simulator = nothing, M = nothing, θ_train = nothing, θ_val = nothing, Z_train = nothing, Z_val = nothing, args...)

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
		if i > 1
			Flux.loadmodel!(estimators[i], Flux.state(estimators[i-1]))
		end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# Train the estimator, dispatching based on the given arguments
		if !isnothing(sampler)
			estimators[i] = train(estimators[i], sampler, simulator; m = mᵢ, kwargs...)
		elseif !isnothing(simulator)
			estimators[i] = train(estimators[i], θ_train, θ_val, simulator; m = mᵢ, kwargs...)
		else
			Z_valᵢ = subsetdata(Z_val, 1:mᵢ) # subset the validation data to the current sample size
			estimators[i] = train(estimators[i], θ_train, θ_val, Z_train, Z_valᵢ; kwargs...)
		end
	end
	return estimators
end
trainx(θ̂, sampler, simulator, M; args...) = _trainx(θ̂, sampler = sampler, simulator = simulator, M = M; args...)
trainx(θ̂, θ_train::P, θ_val::P, simulator, M; args...)  where {P <: Union{AbstractMatrix, ParameterConfigurations}} = _trainx(θ̂, θ_train = θ_train, θ_val = θ_val, simulator = simulator, M = M; args...)

# This method is for when the data can be easily subsetted
function trainx(θ̂, θ_train::P, θ_val::P, Z_train::T, Z_val::T, M::Vector{I}; args...) where {T, P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}

	@assert length(unique(numberreplicates(Z_val))) == 1 "The elements of `Z_val` should be equally replicated: check with `numberreplicates(Z_val)`"
	@assert length(unique(numberreplicates(Z_train))) == 1 "The elements of `Z_train` should be equally replicated: check with `numberreplicates(Z_train)`"

	_trainx(θ̂, θ_train = θ_train, θ_val = θ_val, Z_train = Z_train, Z_val = Z_val, M = M; args...)
end

# This method is for when the data CANNOT be easily subsetted, so another layer of vectors is needed
function trainx(θ̂, θ_train::P, θ_val::P, Z_train::V, Z_val::V; args...) where {V <: AbstractVector{S}} where {S <: Union{V₁, Tuple{V₁, V₂}}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T, P <: Union{AbstractMatrix, ParameterConfigurations}}

	@assert length(Z_train) == length(Z_val)

	@assert !(typeof(θ̂) <: Vector) # check that θ̂ is not a vector of estimators, which is common error if one calls trainx() on the output of a previous call to trainx()

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
		if i > 1
			Flux.loadmodel!(estimators[i], Flux.state(estimators[i-1]))
		end

		# Modify/check the keyword arguments before passing them onto train
		kwargs = (;args...)
		if haskey(kwargs, :savepath) && kwargs.savepath != ""
			kwargs = merge(kwargs, (savepath = kwargs.savepath * "_m$(mᵢ)",))
		end
		kwargs = _modifyargs(kwargs, i, E)

		# Train the estimator for the current sample size
		estimators[i] = train(estimators[i], θ_train, θ_val, Z_trainᵢ, Z_valᵢ; kwargs...)
	end

	return estimators
end

# ---- Miscellaneous helper functions ----

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

function _savestate(θ̂, savepath, epoch = "")
	if !ispath(savepath) mkpath(savepath) end
	model_state = Flux.state(cpu(θ̂))
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
	load_path   = joinpath(path, "network_epoch$(best_epoch).bson")
	save_path   = joinpath(path, "best_network.bson")
	cp(load_path, save_path, force = true)
	
	return nothing
end


ZtoFloat32(Z) = try broadcast.(Float32, Z) catch e Z end
θtoFloat32(θ) = try broadcast(Float32, θ) catch e θ end
