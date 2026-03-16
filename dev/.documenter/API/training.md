


## Training {#Training}

The function [`train()`](/API/training#NeuralEstimators.train) is used to train a neural estimator.

After training, the risk history and optimiser state can be accessed and inspected using [`loadrisk()`](/API/training#NeuralEstimators.loadrisk), [`plotrisk()`](/API/training#NeuralEstimators.plotrisk), and [`loadoptimiser()`](/API/training#NeuralEstimators.loadoptimiser).
<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.train' href='#NeuralEstimators.train'><span class="jlbinding">NeuralEstimators.train</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
train(estimator, θ_train::P, θ_val::P, Z_train::T, Z_val::T; ...) where {P, T}
train(estimator, θ_train::P, θ_val::P, simulator::Function; ...) where {P, T}
train(estimator, sampler::Function, simulator::Function; ...)
```


Trains a neural `estimator`.

The methods cater for different variants of &quot;on-the-fly&quot; simulation. Specifically, a `sampler` can be provided to continuously sample new parameter vectors from the prior, and a `simulator` can be provided to continuously simulate new data conditional on the parameters. If provided with specific sets of parameters (`θ_train` and `θ_val`) and/or data (`Z_train` and `Z_val`), they will be held fixed during training.

The validation parameters and data are always held fixed.

The trained estimator is always returned on the CPU. 

**Keyword arguments common to all methods:**
- `loss = Flux.mae` (applicable only to [`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator)): loss function used to train the neural network. 
  
- `epochs = 100`: number of epochs to train the neural network. An epoch is one complete pass through the entire training data set when doing stochastic gradient descent.
  
- `stopping_epochs = 5`: cease training if the risk does not improve in this number of epochs.
  
- `batchsize = 32`: the batchsize to use when performing stochastic gradient descent, that is, the number of training samples processed between each update of the neural-network parameters.
  
- `optimiser = Flux.setup(Adam(5e-4), estimator)`: any [Optimisers.jl](https://fluxml.ai/Optimisers.jl/stable/) optimisation rule for updating the neural-network parameters. When the training data and/or parameters are held fixed, one may wish to employ regularisation to prevent overfitting; for example, `optimiser = Flux.setup(OptimiserChain(WeightDecay(1e-4), Adam()), estimator)`, which corresponds to L₂ regularisation with penalty coefficient λ=10⁻⁴. 
  
- `lr_schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule}`: defines the learning-rate schedule for adaptively changing the learning rate during training. Accepts either a [ParameterSchedulers.jl](https://fluxml.ai/ParameterSchedulers.jl/dev/) object or `nothing` for a fixed learning rate. By default, it uses [`CosAnneal`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/cyclic/#ParameterSchedulers.CosAnneal) with a maximum set to the initial learning rate from `optimiser`, a minimum of zero, and a period equal to the number of epochs. The learning rate is updated at the end of each epoch. 
  
- `freeze_summary_network = false`: if `true` and the estimator has a `summary_network` field, freezes the summary network parameters during training (i.e., only the inference network is updated). In this case, the summary statistics for a given instance of simulated data are computed only once, giving a significant speedup. This is useful for transfer learning, where a pretrained summary network is held fixed while a new inference network is trained for a different model or estimator type.
  
- `use_gpu = true`: flag indicating whether to use a GPU if one is available.
  
- `adtype::AbstractADType = AutoZygote()`: the automatic differentiation backend used to compute gradients during training. The default uses [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/). Alternatively, `AutoEnzyme()` can be used to enable [Enzyme.jl](https://enzymead.github.io/Enzyme.jl/stable/), which can be faster and more memory efficient, and supports mutation and scalar indexing (requires `using Enzyme`).
  
- `savepath::Union{Nothing, String} = tempdir()`: path to save information generated during training. If `nothing`, nothing is saved. Otherwise, the following files are always saved to both `savepath` and `tempdir()` (the latter for convenient within-session access via [`loadrisk`](/API/training#NeuralEstimators.loadrisk), [`plotrisk`](/API/training#NeuralEstimators.plotrisk), and [`loadoptimiser`](/API/training#NeuralEstimators.loadoptimiser)):
  - `loss_per_epoch.csv`: training and validation risk at each epoch, in the first and second columns respectively.
    
  - `best_optimiser.bson`: optimiser state corresponding to the best validation risk.
    
  - `final_optimiser.bson`: optimiser state at the final epoch.
    
  If additionally `savepath != tempdir()`, the following files are also saved to `savepath`:
  - `best_network.bson`: neural-network parameters corresponding to the best validation risk.
    
  - `final_network.bson`: neural-network parameters at the final epoch.
    
  - `train_time.csv`: total training time in seconds.
    
  
- `risk_history::Union{Nothing, Matrix} = nothing`: a matrix with two columns containing the training and validation risk from a previous call to `train()`, used to initialise the risk history when continuing training. Can be loaded from a previous call to `train` using [`loadrisk`](/API/training#NeuralEstimators.loadrisk).
  
- `verbose = true`: flag indicating whether information, including empirical risk values and timings, should be printed to the console during training.
  

**Keyword arguments common to `train(estimator, sampler, simulator)` and `train(estimator, θ_train, θ_val, simulator)`:**
- `simulator_args = ()`: positional arguments passed to the simulator as `simulator(θ, simulator_args...)`.
  
- `simulator_kwargs::NamedTuple = (;)`: keyword arguments passed to the simulator as `simulator(...; simulator_kwargs...)`.
  
- `epochs_per_Z_refresh = 1`: the number of passes to make through the training set before the training data are refreshed.
  
- `simulate_just_in_time = false`: flag indicating whether we should simulate just-in-time, in the sense that only a `batchsize` number of parameter vectors and corresponding data are in memory at a given time.
  

**Keyword arguments unique to `train(estimator, sampler, simulator)`:**
- `sampler_args = ()`: positional arguments passed to the parameter sampler as `sampler(K, sampler_args...)`.
  
- `sampler_kwargs::NamedTuple = (;)`: keyword arguments passed to the parameter sampler as `sampler(...; sampler_kwargs...)`.
  
- `K = 10000`: number of parameter vectors in the training set.
  
- `K_val = K ÷ 2` number of parameter vectors in the validation set.
  
- `epochs_per_θ_refresh = 1`: the number of passes to make through the training set before the training parameters are refreshed. Must be a multiple of `epochs_per_Z_refresh`. Can also be provided as `epochs_per_theta_refresh`.
  

**Examples**

```julia
using NeuralEstimators, Flux

# Data Z|μ,σ ~ N(μ, σ²), priors μ ~ U(0, 1) and σ ~ U(0, 1)
m = 50 # number of replicates in each data set
sampler(K) = NamedMatrix(μ = randn(K), σ = rand(K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(m))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

# Summary network
num_summaries = 6
summary_network = Chain(Dense(m, 64, gelu), Dense(64, 64, gelu), Dense(64, num_summaries))

# Initialise the estimator
estimator = PointEstimator(summary_network, d; num_summaries = num_summaries)

# Training: simulation on-the-fly
K = 1000
estimator  = train(estimator, sampler, simulator, K = K)

# Plot the risk history (using any plotting backend)
using Plots
unicodeplots()
plotrisk()

# Training: simulation on-the-fly with fixed parameters 
θ_train = sampler(K)
θ_val   = sampler(K)
estimator = train(estimator, θ_train, θ_val, simulator, optimiser = loadoptimiser(), risk_history = loadrisk(), freeze_summary_network = true)

# Training: fixed parameters and fixed data
Z_train   = simulator(θ_train)
Z_val     = simulator(θ_val)
estimator = train(estimator, θ_train, θ_val, Z_train, Z_val, optimiser = loadoptimiser(), risk_history = loadrisk(), freeze_summary_network = true)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/train.jl#L1-L89" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.loadrisk' href='#NeuralEstimators.loadrisk'><span class="jlbinding">NeuralEstimators.loadrisk</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
loadrisk(savepath::String = tempdir())
```


Loads the training and validation risk history saved during the most recent call to `train()`. By default, loads from the temporary directory used during the current session. If a `savepath` was provided to `train()`, that path can be passed here to reload risk history from a previous session.

Returns a matrix with two columns: training risk (column 1) and validation risk (column 2).

See also [`plotrisk`](/API/training#NeuralEstimators.plotrisk).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/train.jl#L669-L679" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.plotrisk' href='#NeuralEstimators.plotrisk'><span class="jlbinding">NeuralEstimators.plotrisk</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plotrisk(savepath::String = tempdir())
```


Plots the training and validation risk as a function of epoch, loaded from `savepath`. By default, loads from the temporary directory used during the current session. If a `savepath` was provided to `train()`, that path can be passed here to plot risk history from a previous session.

Requires a plotting package (e.g., `using Plots`) to be loaded.

See also [`loadrisk`](/API/training#NeuralEstimators.loadrisk).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/train.jl#L687-L697" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' >
<summary><a id='NeuralEstimators.loadoptimiser' href='#NeuralEstimators.loadoptimiser'><span class="jlbinding">NeuralEstimators.loadoptimiser</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
loadoptimiser(savepath::String = tempdir(); best::Bool = true)
```


Loads the optimiser saved during the most recent call to `train()`.

By default, loads from the temporary directory used during the current session. If a `savepath` was provided to `train()`, that path can be passed here to reload the optimiser from a previous session. 

By default, loads the optimiser corresponding to the best network (as measured by validation loss). Set `best = false` to load the optimiser from the final epoch instead, which can be useful for resuming training from exactly where it left off.

The returned optimizer can be passed directly to `train()` via the `optimiser`  keyword argument to resume training with the correct optimiser state.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/msainsburydale/NeuralEstimators.jl/blob/138d4afc03ecc00c49eae4b1b31e01adb0ff5ec1/src/train.jl#L646-L661" target="_blank" rel="noreferrer">source</a></Badge>

</details>

