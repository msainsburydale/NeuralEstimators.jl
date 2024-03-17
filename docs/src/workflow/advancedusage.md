# Advanced usage

## Saving and loading neural estimators

As training is by far the most computationally demanding part of the workflow, one often trains an estimator and then saves it for later use. As discussed in the [`Flux` documentation](https://fluxml.ai/Flux.jl/stable/saving/), there are a number of ways to do this. Perhaps the simplest approach is to save the parameters of the neural estimator (e.g., the weights and biases of the neural networks) in a BSON file:

```
using Flux
using BSON: @save, @load
model_state = Flux.state(θ̂)
@save "estimator.bson" model_state
```

Then, to load the neural estimator at a later time, one initialises an estimator with the same architecture used during training, and then loads the saved parameters into this estimator:

```
@load "estimator.bson" model_state
Flux.loadmodel!(θ̂, model_state)
```

Note that the estimator `θ̂` must be already defined (i.e., only the network parameters are saved, not the architecture). That is, the saved model state should be loaded into a neural estimator with the same architecture as the estimator that we wish to load.

As a convenience, the function [`train`](@ref) allows for the automatic saving of the neural-network parameters during the training stage, via the argument `savepath`. Specifically, if `savepath` is specified, [`train`](@ref) automatically saves the neural estimator's parameters in the folder `savepath`; to load them, one may use the following code:

```
using NeuralEstimators
Flux.loadparams!(θ̂, loadbestweights(savepath))
```

Above, the function `loadparams!` loads the parameters of the best (as determined by [`loadbestweights`](@ref)) neural estimator saved in `savepath`.


## Storing expensive intermediate objects for data simulation

Parameters sampled from the prior distribution $\Omega(\cdot)$ may be stored in two ways. Most simply, they can be stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution; this is the approach taken in the example using univariate Gaussian data. Alternatively, they can be stored in a user-defined subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the $p \times K$ matrix of parameters. With this approach, one may store computationally expensive intermediate objects, such as Cholesky factors, for later use when conducting "on-the-fly" simulation, which is discussed below.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, $\mathcal{Z}_{\text{train}}$ can be simulated continuously during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\text{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\text{train}}$ is fixed. Refreshing $\mathcal{Z}_{\text{train}}$ also has an additional computational benefit; data can be simulated "just-in-time", in the sense that they can be simulated from a small batch of $\vartheta_{\text{train}}$, used to train the neural estimator, and then removed from memory. This can reduce pressure on memory resources when $|\vartheta_{\text{train}}|$ is very large.

One may also regularly refresh $\vartheta_{\text{train}}$, and doing so leads to similar benefits. However, fixing $\vartheta_{\text{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models.  

The above strategies are facilitated with various methods of [`train`](@ref).

## Regularisation

The term *regularisation* refers to a variety of techniques aimed to reduce overfitting when training a neural network.

One common regularisation technique is known as dropout [(Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html), implemented in Flux's [`Dropout`](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dropout) layer. Dropout involves temporarily dropping ("turn off") a randomly selected set of neurons (along with their connections) at each iteration of the training stage, and this results in a computationally-efficient form of model (neural network) averaging. 

## Combining learned and expert summary statistics

See [`DeepSet`](@ref).

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

Piecewise neural estimators are implemented with the struct, [`PiecewiseEstimator`](@ref), and their construction is facilitated with [`trainx`](@ref).  

### Training with variable sample sizes

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case, for the neural Bayes estimator, the risk function becomes
```math
R(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}))
\equiv
\sum_{m \in \mathcal{M}}
P(M=m)\left(\int_{\mathcal{S}^m}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}^{(m)}; \boldsymbol{\gamma}))p(\boldsymbol{Z}^{(m)} \mid \boldsymbol{\theta}) {\text{d}} \boldsymbol{Z}^{(m)}\right).
```
 This approach does not materially alter the workflow, except that one must also sample the number of replicates before simulating the data.

 Below we define data simulation for a range of sample sizes (i.e., a range of integers) under a discrete uniform prior for ``M``, the random variable corresponding to sample size.

```
function simulate(parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	## Number of parameter vectors stored in parameters
	K = size(parameters, 2)

	## Generate K sample sizes from the prior distribution for M
	m̃ = rand(m, K)

	## Pseudocode for data simulation
	Z = [<simulate m̃[k] iid realisations from the model> for k ∈ 1:K]

	return Z
end
```

Then, setting the argument `m` in [`train`](@ref) to be an integer range (e.g., `1:30`) will train the neural estimator with the given variable sample sizes.
