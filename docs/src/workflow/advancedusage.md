# Advanced usage

## Saving and loading neural estimators

As training is by far the most computationally demanding part of the workflow, one often trains an estimator and then saves it for later use. As discussed in the [Flux documentation](https://fluxml.ai/Flux.jl/stable/saving/), there are a number of ways to do this. Perhaps the simplest approach is to save the parameters (i.e., weights and biases) of the neural network in a BSON file:

```
using Flux
using BSON: @save, @load
model_state = Flux.state(θ̂)
@save "estimator.bson" model_state
```

Then, in a later session, one may initialise a neural network with the same architecture used previously, and load the saved parameters:

```
@load "estimator.bson" model_state
Flux.loadmodel!(θ̂, model_state)
```

Note that the estimator `θ̂` must be already defined (i.e., only the network parameters are saved, not the architecture). 

For convenience, the function [`train()`](@ref) allows for the automatic saving of the neural-network parameters during the training stage, via the argument `savepath`. Specifically, if `savepath` is specified, neural estimator's parameters will be saved in the folder `savepath` and, to load the optimal parameters post training, one may use the following code, or similar:

```
using NeuralEstimators
Flux.loadparams!(θ̂, loadbestweights(savepath))
```

Above, the function `loadparams!()` loads the parameters of the best (as determined by [`loadbestweights()`](@ref)) neural estimator saved in `savepath`.


## Storing expensive intermediate objects for data simulation

Parameters sampled from the prior distribution may be stored in two ways. Most simply, they can be stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution. Alternatively, they can be stored in a user-defined struct subtyping [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the $p \times K$ matrix of parameters. With this approach, one may store computationally expensive intermediate objects, such as Cholesky factors, for later use when conducting "on-the-fly" simulation, which is discussed below.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, the training data set, $\mathcal{Z}_{\text{train}}$, can be simulated continuously during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\text{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\text{train}}$ is fixed. Further, this technique allows for data be simulated "just-in-time", in the sense that they can be simulated in small batches, used to train the neural estimator, and then removed from memory. This can substantially reduce pressure on memory resources, particularly when working with large data sets. 

One may also regularly refresh the set $\vartheta_{\text{train}}$ of parameter vectors used during training, and doing so leads to similar benefits. However, fixing $\vartheta_{\text{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models. Hybrid approaches are also possible, whereby the parameters (and possibly the data) are held fixed for several epochs (i.e., several passes through the training set when performing stochastic gradient descent) before being refreshed. 

The above strategies are facilitated with various methods of [`train()`](@ref).

## Regularisation

The term *regularisation* refers to a variety of techniques aimed to reduce overfitting when training a neural network, primarily by discouraging complex models.

One common regularisation technique is known as dropout [(Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html), implemented in Flux's [`Dropout`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout) layer. Dropout involves temporarily dropping ("turning off") a randomly selected set of neurons (along with their connections) at each iteration of the training stage, and this results in a computationally-efficient form of model (neural-network) averaging.

Another class of regularisation techniques involve modifying the loss function. For instance, L₁ regularisation (sometimes called lasso regression) adds to the loss a penalty based on the absolute value of the neural-network parameters. Similarly, L₂ regularisation (sometimes called ridge regression) adds to the loss a penalty based on the square of the neural-network parameters. Note that these penalty terms are not functions of the data or of the statistical-model parameters that we are trying to infer, and therefore do not modify the Bayes risk or the associated Bayes estimator. These regularisation techniques can be implemented straightforwardly by providing a custom `optimiser` to [`train`](@ref) that includes a [`SignDecay`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.SignDecay) object for L₁ regularisation, or a [`WeightDecay`](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Flux.Optimise.WeightDecay) object for L₂ regularisation. See the [Flux documentation](https://fluxml.ai/Flux.jl/stable/training/training/#Regularisation) for further details.

For example, the following code constructs a neural Bayes estimator using dropout and L₁ regularisation with penalty coefficient $\lambda = 10^{-4}$:

```
using NeuralEstimators
using Flux

# Generate data from the model Z ~ N(θ, 1) and θ ~ N(0, 1)
p = 1       # number of unknown parameters in the statistical model
m = 5       # number of independent replicates
d = 1       # dimension of each independent replicate
K = 3000    # number of training samples
θ_train = randn(1, K)
θ_val   = randn(1, K)
Z_train = [μ .+ randn(1, m) for μ ∈ eachcol(θ_train)]
Z_val   = [μ .+ randn(1, m) for μ ∈ eachcol(θ_val)]

# Architecture with dropout layers
ψ = Chain(
	Dense(1, 32, relu),
	Dropout(0.1),
	Dense(32, 32, relu),
	Dropout(0.5)
	)     
ϕ = Chain(
	Dense(32, 32, relu),
	Dropout(0.5),
	Dense(32, 1)
	)           
θ̂ = DeepSet(ψ, ϕ)

# Optimiser with L₂ regularisation
optimiser = Flux.setup(OptimiserChain(SignDecay(1e-4), Adam()), θ̂)

# Train the estimator
train(θ̂, θ_train, θ_val, Z_train, Z_val; optimiser = optimiser)
```

Note that when the training data and/or parameters are held fixed during training, L₂ regularisation with penalty coefficient $\lambda = 10^{-4}$ is applied by default.

## Expert summary statistics

Implicitly, neural estimators involve the learning of summary statistics. However, some summary statistics are available in closed form, simple to compute, and highly informative (e.g., sample quantiles, the empirical variogram, etc.). Often, explicitly incorporating these expert summary statistics in a neural estimator can simplify the optimisation problem, and lead to a better estimator. 

The fusion of learned and expert summary statistics is facilitated by our implementation of the [`DeepSet`](@ref) framework. Note that this implementation also allows the user to construct a neural estimator using only expert summary statistics, following, for example, [Gerber and Nychka (2021)](https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.382) and [Rai et al. (2024)](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.2845). Note also that the user may specify arbitrary expert summary statistics, however, for convenience several standard [User-defined summary statistics](@ref) are provided with the package, including a fast approximate version of the empirical variogram. 

## Variable sample sizes

A neural estimator in the Deep Set representation can be applied to data sets of arbitrary size. However, even when the neural Bayes estimator approximates the true Bayes estimator arbitrarily well, it is conditional on the number of replicates, $m$, and is not necessarily a Bayes estimator for $m^* \ne m$. Denote a data set comprising $m$ replicates as $\boldsymbol{Z}^{(m)} \equiv (\boldsymbol{Z}_1', \dots, \boldsymbol{Z}_m')'$. There are at least two (non-mutually exclusive) approaches one could adopt if data sets with varying $m$ are envisaged, which we describe below.

### Piecewise estimators

If data sets with varying $m$ are envisaged, one could train $l$ neural Bayes estimators for different sample sizes, or groups thereof (e.g., a small-sample estimator and a large-sample estimator).
 Specifically, for sample-size changepoints $m_1$, $m_2$, $\dots$, $m_{l-1}$, one could construct a piecewise neural Bayes estimator,
```math
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*)
=
\begin{cases}
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_1}) & m \leq m_1,\\
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_2}) & m_1 < m \leq m_2,\\
\quad \vdots \\
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}^*_{\tilde{m}_l}) & m > m_{l-1},
\end{cases}
```
where, here, $\boldsymbol{\gamma}^* \equiv (\boldsymbol{\gamma}^*_{\tilde{m}_1}, \dots, \boldsymbol{\gamma}^*_{\tilde{m}_{l-1}})$, and where $\boldsymbol{\gamma}^*_{\tilde{m}}$ are the neural-network parameters optimised for sample size $\tilde{m}$ chosen so that $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*_{\tilde{m}})$ is near-optimal over the range of sample sizes in which it is applied.
This approach works well in practice, and it is less computationally burdensome than it first appears when used in conjunction with pre-training.

Piecewise neural estimators are implemented with the struct, [`PiecewiseEstimator`](@ref), and their construction is facilitated with [`trainx()`](@ref).  

### Training with variable sample sizes

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case, for the neural Bayes estimator, the risk function becomes
```math
\sum_{m \in \mathcal{M}}
P(M=m)\left(
\int_\Theta \int_{\mathcal{Z}^m}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{z}^{(m)}))f(\boldsymbol{z}^{(m)} \mid \boldsymbol{\theta}) \rm{d} \boldsymbol{z}^{(m)} \rm{d} \Pi(\boldsymbol{\theta})
\right).
```
This approach does not materially alter the workflow, except that one must also sample the number of replicates before simulating the data during the training phase. 

The following pseudocode illustrates how one may modify a general data simulator to train under a range of sample sizes, with the distribution of $M$ defined by passing any object that can be sampled using `rand(m, K)` (e.g., an integer range like `1:30`, an integer-valued distribution from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/), etc.):

```
function simulate(parameters, m) 

	## Number of parameter vectors stored in parameters
	K = size(parameters, 2)

	## Generate K sample sizes from the prior distribution for M
	m̃ = rand(m, K)

	## Pseudocode for data simulation
	Z = [<simulate m̃[k] realisations from the model> for k ∈ 1:K]

	return Z
end

## Method that allows an integer to be passed for m
simulate(parameters, m::Integer) = simulate(parameters, range(m, m))
```
