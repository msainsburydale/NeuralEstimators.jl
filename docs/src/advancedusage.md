# Advanced usage

## Backends

[Flux.jl](https://fluxml.ai/Flux.jl/stable/) and [Lux.jl](https://lux.csail.mit.edu/stable/) are the primarily supported backends. These frameworks differ in a key way: Flux stores trainable parameters and states inside the network object, while Lux represents them explicitly as separate objects. Flux's stateful, object-oriented style will feel familiar to PyTorch users, while Lux's explicit, functional style will feel familiar to JAX/Flax users.

Despite these differences, the high-level API of NeuralEstimators.jl is largely consistent across backends. A minimal working example illustrating the typical workflows:

::: code-group

```julia [Flux.jl]
using NeuralEstimators, Flux

# Functions to sample from the model
d, n = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(n))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network mapping n inputs into d outputs
network = Flux.Chain(Flux.Dense(n, 64, gelu), Flux.Dense(64, 64, gelu), Flux.Dense(64, d))

# Initialise a neural estimator
estimator = PointEstimator(network, d; num_summaries = d)

# Train the estimator
estimator = train(estimator, sampler, simulator)

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)

# Apply to observed data
θ = sampler(1)                   # ground truth (not known in practice)
Z = simulator(θ);                # stand-in for real observations
infer(estimator, Z)              # point estimate
```

```julia [Lux.jl (implicit)]
using NeuralEstimators, Lux, Zygote

# Functions to sample from the model
d, n = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(n))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network mapping n inputs into d outputs
network = Lux.Chain(Lux.Dense(n, 64, gelu), Lux.Dense(64, 64, gelu), Lux.Dense(64, d))

# Initialise a neural estimator
estimator = PointEstimator(network, d; num_summaries = d)

# Train the estimator
estimator = train(estimator, sampler, simulator)

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test)
bias(assessment)
rmse(assessment)

# Apply to observed data
θ = sampler(1)                   # ground truth (not known in practice)
Z = simulator(θ);                # stand-in for real observations
infer(estimator, Z)              # point estimate
```

```julia [Lux.jl (idiomatic)]
using NeuralEstimators, Lux, Zygote, Random, Optimisers

# Functions to sample from the model
d, n = 2, 100  # dimension of θ and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(n))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Neural network mapping n inputs into d outputs
network = Lux.Chain(Lux.Dense(n, 64, gelu), Lux.Dense(64, 64, gelu), Lux.Dense(64, d))

# Initialise a neural estimator
estimator = PointEstimator(network, d; num_summaries = d)

# Initialise the parameters and states
rng    = Random.default_rng()
ps, st = Lux.setup(rng, estimator)

# Train the estimator
optimiser  = Adam(5e-4)
trainstate = Lux.Training.TrainState(estimator, ps, st, optimiser)
trainstate = train(trainstate, sampler, simulator)
estimator  = trainstate.model      # the estimator, with the standardisers fitted by train()
ps         = trainstate.parameters
st         = trainstate.states

# Assess the estimator
θ_test = sampler(250)
Z_test = simulator(θ_test);
assessment = assess(estimator, θ_test, Z_test, ps, st)
bias(assessment)
rmse(assessment)

# Apply to observed data
θ = sampler(1)                   # ground truth (not known in practice)
Z = simulator(θ);                # stand-in for real observations
infer(estimator, Z, ps, st)      # point estimate
```

:::

## Performance tips with Lux.jl
 
Consider loading the [optional dependencies](https://lux.csail.mit.edu/stable/manual/performance_pitfalls#Optional-Dependencies-for-Performance) for improved performance on CPUs. 
 
If you plan to use the GPU via both CUDA and [XLA/Reactant](https://lux.csail.mit.edu/stable/manual/compiling_lux_models) in the same session, ensure that CUDA.jl/cuDNN.jl are loaded before Reactant.jl:

```julia
using CUDA, cuDNN
using Reactant
Reactant.set_default_backend("gpu")
```

For the most computationally efficient setup, use XLA/Reactant.jl during training by passing `device = reactant_device()` to [`train`](@ref).

## GPU acceleration

To improve computational efficiency, various GPU backends are supported. Once the relevant package is loaded and a compatible GPU is available, it will be used automatically:

::: code-group

```julia [NVIDIA GPUs]
using CUDA, cuDNN
```

```julia [AMD ROCm GPUs]
using AMDGPU
```

```julia [Metal M-Series GPUs]
using Metal
```

```julia [Intel GPUs]
using oneAPI
```

:::

## Saving and loading estimators

The recommended approach is to let [`train`](@ref) save the neural network for you, by providing a `savepath`, and then to load it with [`loadestimator`](@ref). Since the saved files store the parameters (and states) of the neural networks and not their architecture, loading involves constructing an estimator with the same architecture and then loading the saved parameters into it:
```julia
# Training: saves best_estimator.bson (and final_estimator.bson) to the given savepath
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val, savepath = "path/to/folder")

# Loading (possibly in a later session): construct the same architecture, then load
estimator = PointEstimator(...) # same architecture as above
estimator = loadestimator(estimator, "path/to/folder")
```
By default, the parameters corresponding to the best validation risk are loaded, that is, those of the estimator that was returned by `train()`; pass `best = false` to load the parameters from the final epoch instead.

Note that the same deep-learning backend must be loaded as when training. Note also that, when the estimator contains Lux networks, `loadestimator()` returns a [`LuxEstimator`](@ref) (which stores the neural-network parameters and states) irrespective of whether the estimator passed to it was wrapped in one, so the returned value should always be assigned, as above.

An estimator can also be saved at any time with [`saveestimator`](@ref), which writes the same file as [`train`](@ref) (with either Flux or Lux):
```julia
# Save
saveestimator(estimator, "path/to/folder")

# Load (possibly in a later session): construct the same architecture, then load
estimator = PointEstimator(...) # same architecture as above
estimator = loadestimator(estimator, "path/to/folder")
```

Note that saving only the parameters of the neural networks (e.g., the model state given by `Flux.state`, or the parameters and states of Lux networks) does not save the standardisers or parameter bounds of the estimator, which are needed to use it: use [`saveestimator`](@ref) instead. It is also possible to save the entire estimator, including its architecture, as a Julia struct (see [here](https://fluxml.ai/Flux.jl/stable/guide/saving/#Saving-Models-as-Julia-Structs) for Flux), although such a file can usually be loaded only with the versions of the packages that saved it, so [`saveestimator`](@ref) is recommended for long-term storage.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, the training data can be simulated continuously during training, a technique known as "simulation-on-the-fly". This strategy prevents overfitting and facilitates the use of larger networks that are prone to overfitting when the training data are fixed. Further, it allows for data to be simulated "just-in-time", in the sense that data can be simulated in small batches, used to train the neural estimator, and then immediately removed from memory.

One may also regularly refresh the set of parameters (i.e., inferential targets) used during training, and doing so leads to similar benefits. However, fixing the parameters allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models. Hybrid approaches are also possible, whereby the parameters (and possibly the data) are held fixed for several epochs (i.e., several passes through the training set when performing stochastic gradient descent) before being refreshed. 

The above strategies are facilitated with various methods of [`train()`](@ref) and through user-defined subtypes of [`AbstractParameterSet`](@ref).


## [Standardisation](@id advanced-standardisation)

It is important to ensure that the inputs and outputs of the neural networks are on a reasonable numerical scale, since values with very large absolute value can lead to numerical instability during training (e.g., exploding gradients).

Hence, by default, every estimator standardises the data before they enter the neural networks, by subtracting a mean and dividing by a standard deviation, and also standardises the parameters, except the parameters of a [`SpikeAndSlab`](@ref) distribution and, when this would change the estimator, those of a [`PointEstimator`](@ref) (see below). The means and standard deviations are computed by [`train`](@ref) from the training data before training begins, and they are saved with the estimator (see [Saving and loading estimators](@ref)). Since standardisation is a fixed bijective map, it does not change the quantity being estimated, and the data, parameters, estimates, and posterior draws given to and returned by every function of the package remain on the original scale.

Standardisation is controlled by the keyword arguments `standardise_data` and `standardise_parameters` of each estimator, which may be `true`, `false`, or a [`Standardiser`](@ref) with statistics given by the user (and, for the parameters of a [`PointEstimator`](@ref), `:auto`, the default); for data stored as [`DataAndSummaries`](@ref) objects, `standardise_data` may also be a named tuple `(Z = …, S = …)` of these values for the data and the expert summary statistics. For example:
```julia
# No standardisation of the data
estimator = PosteriorEstimator(d, summary_network; num_summaries, standardise_data = false)

# Statistics given by the user: one per channel of images stored as a W × H × C × K array
s = Standardiser(mean(Z; dims = (1, 2, 4)), std(Z; dims = (1, 2, 4)))
estimator = PosteriorEstimator(d, summary_network; num_summaries, standardise_data = s)

# Standardisation of the parameters of a point estimator, whatever its loss function and output layer (by default, they
# are standardised only if this leaves the estimator unchanged; see PointEstimator)
estimator = PointEstimator(d, summary_network; num_summaries, standardise_parameters = true)

# Expert summary statistics standardised, but not the data given to the summary network
estimator = PosteriorEstimator(d, summary_network; num_summaries, standardise_data = (Z = false, S = true))
```

Note the following:
- By default, the statistics are computed per row of data stored as matrices (e.g., the expert summary statistics of [`DataAndSummaries`](@ref) objects, or the data sets of a [`DeepSet`](@ref), pooled over the replicates), and are pooled over all elements of arrays with three or more dimensions (e.g., images); see [`Standardiser`](@ref).
- Standardisation is applied before the summary network. A transformation in the first layer of the network that relies on the original scale or support of the data, such as `z -> log.(1 .+ z)` for positive data, should therefore be applied in the simulator instead, so that standardisation follows it, or used with `standardise_data = false`. Transformations of this kind remain useful for heavy-tailed data, whose mean and standard deviation are dominated by the largest values. 
- When the parameters of a [`PointEstimator`](@ref) are standardised, its loss function is evaluated on the standardised scale, and an output layer that constrains the estimates (e.g., the activation function `softplus` or `exp`) constrains them on the standardised scale. Hence, by default (`standardise_parameters = :auto`), [`train`](@ref) standardises them only if this leaves the estimator unchanged: if the loss function is, for example, `mae` or `mse` (but not [`tanhloss`](@ref)), and the neural network ends in a `Dense` layer without activation function. Parameter bounds (`lower` and `upper`) are likewise imposed by evaluating the loss on a transformed scale, which leaves the estimator unchanged only for `mae` and quantile losses: with any other loss, impose the constraints within the neural network itself instead (see [`PointEstimator`](@ref)).
- When a summary network is reused in another estimator (e.g., for transfer learning; see [`setsummarynetwork`](@ref)), pass the data standardiser of the original estimator, `standardise_data = estimator.data_standardiser`, so that the network sees data on the same scale. For a [`LuxEstimator`](@ref), as returned by `train()` and `loadestimator()` with Lux, this is `estimator.estimator.data_standardiser`, and for a Lux `TrainState` it is `trainstate.model.data_standardiser`.

Data stored in a type defined by the user are not standardised by default. To standardise them, add a method of the call operator of [`Standardiser`](@ref), `s(Z)`, which applies a standardiser `s` to data `Z`, and, so that [`train`](@ref) can compute the statistics (with `standardise_data = true`, the default), a method of [`NeuralEstimators.fitstandardiser`](@ref). For example, for data sets stored together with indicators of missingness that should not be standardised:
```julia
struct MyData{A, B}
    Z::A # values
    W::B # indicators of missingness
end

# Standardise the values only, with statistics computed from the values only
(s::Standardiser)(d::MyData) = MyData(s(d.Z), d.W)
NeuralEstimators.fitstandardiser(d::MyData) = NeuralEstimators.fitstandardiser(d.Z)
```
Note the following:
- The method of the call operator is applied once to all of the data on the host during training (and by [`summarystatistics`](@ref)), and to each mini-batch at inference, possibly on a GPU, and never within automatic differentiation. It should therefore apply `s` to the arrays that the type stores, as above, which converts them to single precision and moves the statistics to their device, and it should not modify its input.
- The methods are needed for the type of the data given to [`train`](@ref) and [`estimate`](@ref). For example, for a vector of `MyData` objects, define `(s::Standardiser)(Z::AbstractVector{<:MyData}) = map(s, Z)` and a method of `fitstandardiser` that pools the statistics, such as `NeuralEstimators.fitstandardiser(Z::AbstractVector{<:MyData}) = NeuralEstimators.fitstandardiser([d.Z for d in Z])`.
- `fitstandardiser` must return a `Standardiser`, or `nothing` if the statistics cannot be computed (the data are then not standardised). With statistics given by the user (e.g., `standardise_data = Standardiser(mean, std)`), only the method of the call operator is needed.
- The methods should dispatch on the type of the data only, not on that of the standardiser.
- A `Standardiser` should be given to an estimator through `standardise_data`, rather than placed in its neural network, where it would be neither saved with the estimator nor skipped during training (when the data have already been standardised).
- As for any type of data, `numobs` and `getobs` must also be defined, so that the data can be split into mini-batches.

See [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) for further discussion and possible approaches. 

## Regularisation

The term *regularisation* refers to a variety of techniques aimed to reduce overfitting when training a neural network, primarily by discouraging complex models. 

!!! note "Simulation on-the-fly"
	When the training data and parameters are simulated dynamically (i.e., ["on the fly"](@ref "On-the-fly and just-in-time simulation")), overfitting is generally not a concern.

One popular regularisation technique is known as dropout, implemented with `Dropout` ([Flux](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout)/[Lux](https://lux.csail.mit.edu/stable/api/Lux/layers#Dropout-Layers)). Dropout involves temporarily dropping ("turning off") a randomly selected set of neurons (along with their connections) at each iteration of the training stage, which results in a computationally-efficient form of model (neural-network) averaging [(Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html).

Another class of regularisation techniques involve modifying the loss function. For instance, L₁ regularisation (sometimes called lasso regression) adds to the loss a penalty based on the absolute value of the neural-network parameters. Similarly, L₂ regularisation (sometimes called ridge regression) adds to the loss a penalty based on the square of the neural-network parameters. Note that these penalty terms are not functions of the data or of the statistical-model parameters that we are trying to infer. These regularisation techniques can be implemented straightforwardly by providing a custom `optimiser` rule to [`train`](@ref) that includes a [`SignDecay`](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Optimisers.SignDecay) object for L₁ regularisation, or a [`WeightDecay`](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Optimisers.WeightDecay) object for L₂ regularisation. See the [Optimisers.jl](https://fluxml.ai/Flux.jl/stable/reference/training/optimisers/#Composing-Optimisers) and [Flux.jl](https://fluxml.ai/Flux.jl/stable/guide/training/training/#Regularisation) documentation for further details. 

For illustration, the following code constructs a neural Bayes estimator using dropout and L₁ regularisation with penalty coefficient $10^{-4}$:

```julia
using NeuralEstimators, Flux

# Functions to simulate data Z|μ,σ ~ N(μ, σ²) with μ ~ N(0, 1) and σ ~ U(0, 1)
d, n = 2, 100  # number of parameters and number of replicates
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector, n) = θ["μ"] .+ θ["σ"] .* sort(randn(n))
simulator(θ::AbstractMatrix, n) = reduce(hcat, simulator.(eachcol(θ), n))

# Fixed training/validation sets
K = 10000
θ_train = sampler(K)
θ_val   = sampler(K)
Z_train = simulator(θ_train, n)
Z_val   = simulator(θ_val, n)

# Neural network with dropout layers
network = Chain(
	Dense(n, 128, relu), 
	Dropout(0.1), 
	Dense(128, 128, gelu), 
	Dropout(0.1),
	Dense(128, d)
	)

# Initialise estimator
estimator = PointEstimator(network)

# Optimiser with L₁ regularisation
optimiser = OptimiserChain(SignDecay(1e-4), Adam(5e-4))

# Train the estimator
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val; optimiser = optimiser)
```