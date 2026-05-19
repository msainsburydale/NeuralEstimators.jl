# Advanced usage

## Backends

[Flux.jl](https://fluxml.ai/Flux.jl/stable/) and [Lux.jl](https://lux.csail.mit.edu/stable/) are the primarily supported backends. These frameworks differ in a key way: Flux stores trainable parameters and states inside the network object, while Lux represents them explicitly as separate objects. Flux's stateful, object-oriented style will feel familiar to PyTorch users, while Lux's explicit, functional style will feel familiar to JAX/Flax users.

[SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl), which is optimised for small networks on the CPU, is also supported via [Lux.jl](https://lux.csail.mit.edu/stable/api/Lux/interop#Lux-Models-to-Simple-Chains).

Despite these differences, the high-level API of NeuralEstimators.jl is largely consistent across backends. The typical workflows are as follows:

::: code-group

```julia [Flux.jl]
using NeuralEstimators, Flux

network   = Flux.Chain(...)
estimator = PointEstimator(network)
estimator = train(estimator, sampler, simulator)
assess(estimator, θ_test, Z_test)
estimate(estimator, Z)
```

```julia [Lux.jl (implicit)]
using NeuralEstimators, Lux

network   = Lux.Chain(...)
estimator = PointEstimator(network)
estimator = train(estimator, sampler, simulator)          
assess(estimator, θ_test, Z_test)
estimate(estimator, Z)
```

```julia [Lux.jl (idiomatic)]
using NeuralEstimators, Lux, Random, Optimisers

network    = Lux.Chain(...)
estimator  = PointEstimator(network)

# Initialize the parameters/states
rng        = Random.default_rng()
ps, st     = Lux.setup(rng, estimator)

# Training
optimiser  = Adam(5e-4)
trainstate = Lux.Training.TrainState(estimator, ps, st, optimiser)
trainstate = train(trainstate, sampler, simulator)
ps         = trainstate.parameters
st         = trainstate.states

assess(estimator, θ_test, Z_test, ps, st)
estimate(estimator, Z, ps, st)
```

```julia [SimpleChains.jl]
using NeuralEstimators, Lux, SimpleChains

# Define Lux network and convert it to SimpleChains
network  = Lux.Chain(...)
adaptor  = ToSimpleChainsAdaptor(...) # declare input size
network  = adaptor(network)

# Then proceed with Lux workflow... 
```

:::

### Performance tips with Lux.jl
 
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

Neural estimators can be saved and loaded in the same way as regular Flux/Lux models (see the [Flux documentation](https://fluxml.ai/Flux.jl/stable/guide/saving/)). For example, to save and load the model state of a Flux-based neural estimator:
```julia
using Flux
using BSON: @save, @load

# Save
model_state = Flux.state(estimator)
@save "estimator.bson" model_state

# Load (initialise an estimator with the same architecture, then load the state)
@load "estimator.bson" model_state
Flux.loadmodel!(estimator, model_state)
```

For **Lux** users, we save the parameters/states directly:
```julia
using Lux
using BSON: @save, @load

# Save
@save "estimator.bson" parameters=estimator.ps states=estimator.st

# Load (initialise an estimator with the same architecture, then load the parameters/states)
@load "estimator.bson" parameters states
estimator = Lux.setparam(estimator, parameters)
```

It is also straightforward to save the entire estimator including its architecture (see [here](https://fluxml.ai/Flux.jl/stable/guide/saving/#Saving-Models-as-Julia-Structs) for Flux), though saving the model state as above is recommended for long-term storage.

For convenience, [`train`](@ref) supports automatic saving of the model state during training via the `savepath` argument.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, the training data can be simulated continuously during training, a technique known as "simulation-on-the-fly". This strategy prevents overfitting and facilitates the use of larger networks that are prone to overfitting when the training data are fixed. Further, it allows for data to be simulated "just-in-time", in the sense that data can be simulated in small batches, used to train the neural estimator, and then immediately removed from memory.

One may also regularly refresh the set of parameters (i.e., inferential targets) used during training, and doing so leads to similar benefits. However, fixing the parameters allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models. Hybrid approaches are also possible, whereby the parameters (and possibly the data) are held fixed for several epochs (i.e., several passes through the training set when performing stochastic gradient descent) before being refreshed. 

The above strategies are facilitated with various methods of [`train()`](@ref) and through user-defined subtypes of [`AbstractParameterSet`](@ref).


## Feature scaling 

It is important to ensure that the data passed through the neural network are on a reasonable numerical scale, since values with very large absolute value can lead to numerical instability during training (e.g., exploding gradients). 

A relatively simply way to achieve this is by including a transformation in the first layer of the neural network. For example, if the data have positive support, one could define the neural network with the first layer applying a log transformation:

```julia
network = Chain(z -> log.(1 + z), ...)
```

If the data are not strictly positive, one may consider the following signed transformation:

```julia
network = Chain(z -> sign.(z) .* log.(1 .+ abs.(z)), ...)
```

A simple preprocessing layer or transformation pipeline such as this can make a significant difference in performance and stability. See [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling) for further discussion and possible approaches. 

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
train(estimator, θ_train, θ_val, Z_train, Z_val; optimiser = optimiser)
```