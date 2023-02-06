# Advanced usage

In this section, we discuss practical considerations on how to construct neural estimators most effectively.

## Storing expensive intermediate objects for data simulation

Parameters sampled from the prior distribution $\Omega(\cdot)$ may be stored in two ways. Most simply, they can be stored as a $p \times K$ matrix, where $p$ is the number of parameters in the model and $K$ is the number of parameter vectors sampled from the prior distribution; this is the approach taken in the example using univariate Gaussian data. Alternatively, they can be stored in a user-defined subtype of the abstract type [`ParameterConfigurations`](@ref), whose only requirement is a field `θ` that stores the $p \times K$ matrix of parameters. With this approach, one may store computationally expensive intermediate objects, such as Cholesky factors, for later use when conducting "on-the-fly" simulation, which is discussed below.

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, $\mathcal{Z}_{\text{train}}$ can be simulated continuously during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\text{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\text{train}}$ is fixed. Refreshing $\mathcal{Z}_{\text{train}}$ also has an additional computational benefit; data can be simulated "just-in-time", in the sense that they can be simulated from a small batch of $\vartheta_{\text{train}}$, used to train the neural estimator, and then removed from memory. This can reduce pressure on memory resources when $|\vartheta_{\text{train}}|$ is very large.

One may also regularly refresh $\vartheta_{\text{train}}$, and doing so leads to similar benefits. However, fixing $\vartheta_{\text{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models.  

The above strategies are facilitated with the various methods of [`train`](@ref).


## Variable sample sizes

A neural estimator in the Deep Set representation can be applied to data sets of arbitrary size. However, even when the neural Bayes estimator approximates the true Bayes estimator arbitrarily well, it is conditional on the number of replicates, $m$, and is not necessarily a Bayes estimator for $m^* \ne m$. Denote a data set comprising $m$ replicates as $\mathbf{Z}^{(m)} \equiv (\mathbf{Z}_1', \dots, \mathbf{Z}_m')'$. There are at least two (non-mutually exclusive) approaches one could adopt if data sets with varying $m$ are envisaged, which we describe below.

## Piecewise estimators

If data sets with varying $m$ are envisaged, one could train $l$ neural Bayes estimators for different sample sizes, or groups thereof (e.g., a small-sample estimator and a large-sample estimator).
 Specifically, for sample-size changepoints $m_1$, $m_2$, $\dots$, $m_{l-1}$, one could construct a piecewise neural Bayes estimator,
```math
\hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}^*)
=
\begin{cases}
\hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}^*_{\tilde{m}_1}) & m \leq m_1,\\
\hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}^*_{\tilde{m}_2}) & m_1 < m \leq m_2,\\
\quad \vdots \\
\hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}^*_{\tilde{m}_l}) & m > m_{l-1},
\end{cases}
```
where, here, $\mathbf{\gamma}^* \equiv (\mathbf{\gamma}^*_{\tilde{m}_1}, \dots, \mathbf{\gamma}^*_{\tilde{m}_{l-1}})$, and where $\mathbf{\gamma}^*_{\tilde{m}}$ are the neural-network parameters optimised for sample size $\tilde{m}$ chosen so that $\hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}^*_{\tilde{m}})$ is near-optimal over the range of sample sizes in which it is applied.
This approach works well in practice, and it is less computationally burdensome than it first appears when used in conjunction with pre-training.

Piecewise neural estimators are implemented with the struct, [`PiecewiseEstimator`](@ref), and their construction is facilitated with the method of [`train`](@ref) that takes five positional arguments. Below, we replicate the example of inferring $\mu$ and $\sigma$ from $N(\mu, \sigma^2)$ data, but this time we train three neural estimators with sample sizes $\tilde{m}_l$ equal to 1, 10, and 30, respectively.   

```
using NeuralEstimators
import NeuralEstimators: simulate
using Flux
using Distributions

function sample(Ω, K)
	μ = rand(Ω.μ, K)
	σ = rand(Ω.σ, K)
	θ = hcat(μ, σ)'
	return θ
end

Ω = (μ = Normal(0, 1), σ = Uniform(0.1, 1))
θ_train = sample(Ω, 10000)
θ_val   = sample(Ω, 2000)

function simulate(θ_set, m)
	Z = [rand(Normal(θ[1], θ[2]), 1, m) for θ ∈ eachcol(θ_set)]
	Z = broadcast.(Float32, Z)
	return Z
end

## Sample sizes used for training each neural estimator
M = [1, 10, 30]

## Simulate data; need the maximum sample size that will be used during training
Z_train = simulate(θ_train, maximum(M))
Z_val   = simulate(θ_val, maximum(M))

## Define the architecture common to each neural estimator
w = 32   # number of neurons in each layer
p = 2    # number of parameters in the statistical model
ψ = Chain(Dense(1, w, relu), Dense(w, w, relu))
ϕ = Chain(Dense(w, w, relu), Dense(w, p))
θ̂ = DeepSet(ψ, ϕ)

## Train the neural estimators for each sample size in M
estimators = train(θ̂ , θ_train, θ_val, Z_train, Z_val, M, epochs = 10)

## Construct a PiecewiseEstimator from the above neural estimators
m_breaks = [5, 20]
piecewise_estimator = PiecewiseEstimator(estimators, m_breaks)
```

## Training with variable sample sizes

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case, for the neural Bayes estimator, the risk function becomes
```math
R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}))
\equiv
\sum_{m \in \mathcal{M}}
P(M=m)\left(\int_{\mathcal{S}^m}  L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}))p(\mathbf{Z}^{(m)} \mid \mathbf{\theta}) {\text{d}} \mathbf{Z}^{(m)}\right).
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

## Loading previously saved neural estimators

As training is by far the most computationally demanding part of the workflow, one typically trains an estimator and then saves it for later use. More specifically, one usually saves the *parameters* of the neural estimator (e.g., the weights and biases of the neural networks); then, to load the neural estimator at a later time, one initialises an estimator with the same architecture used during training, and then loads the saved parameters into this estimator.

If the argument `savepath` is specified, [`train`](@ref) automatically saves the neural estimator's parameters; to load them, one may use the following code, or similar:

```
using Flux: loadparams!

θ̂ = architecture()
loadparams!(θ̂, loadbestweights(savepath))
```

Above, `architecture()` is a user-defined function that returns a neural estimator with the same architecture as the estimator that we wish to load, but with randomly initialised parameters, and the function `loadparams!` loads the parameters of the best (as determined by [`loadbestweights`](@ref)) neural estimator saved in `savepath`.
