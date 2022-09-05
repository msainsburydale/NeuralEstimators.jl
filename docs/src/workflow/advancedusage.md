# Advanced usage

## On-the-fly and just-in-time simulation

When data simulation is (relatively) computationally inexpensive, $\mathcal{Z}_{\rm{train}}$ can be simulated periodically during training, a technique coined "simulation-on-the-fly". Regularly refreshing $\mathcal{Z}_{\rm{train}}$ leads to lower out-of-sample error and to a reduction in overfitting. This strategy therefore facilitates the use of larger, more representationally-powerful networks that are prone to overfitting when $\mathcal{Z}_{\rm{train}}$ is fixed. Refreshing $\mathcal{Z}_{\rm{train}}$ also has an additional computational benefit; data can be simulated "just-in-time", in the sense that they can be simulated from a small batch of $\vartheta_{\rm{train}}$, used to train the neural estimator, and then removed from memory. This can reduce pressure on memory resources when $|\vartheta_{\rm{train}}|$ is very large.

One may also regularly refresh $\vartheta_{\rm{train}}$, and doing so leads to similar benefits. However, fixing $\vartheta_{\rm{train}}$ allows computationally expensive terms, such as Cholesky factors when working with Gaussian process models, to be reused throughout training, which can substantially reduce the training time for some models.  

The above strategies are facilitated with the various methods of [`train`](@ref).


## Variable sample sizes

A neural estimator in the Deep Set representation can be applied to data sets of arbitrary size. However, even when the neural Bayes estimator approximates the true Bayes estimator arbitrarily well, it is conditional on the number of replicates, $m$, and is not necessarily a Bayes estimator for $m^* \ne m$. Denote a data set comprising $m$ replicates as $\mathbf{Z}^{(m)} \equiv (\mathbf{Z}_1', \dots, \mathbf{Z}_m')'$. There are at least two (non-mutually exclusive) approaches one could adopt if data sets with varying $m$ are envisaged, which we describe below.

### Piecewise estimators

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

Piecewise neural estimators are implemented with the struct, [`PiecewiseEstimator`](@ref).

### Training with a variable sample size

Alternatively, one could treat the sample size as a random variable, $M$, with support over a set of positive integers, $\mathcal{M}$, in which case, for the neural Bayes estimator, the risk function becomes
```math
R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}))
\equiv
\sum_{m \in \mathcal{M}}
P(M=m)\left(\int_{\mathcal{S}^m}  L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathbf{Z}^{(m)}; \mathbf{\gamma}))p(\mathbf{Z}^{(m)} \mid \mathbf{\theta}) d \mathbf{Z}^{(m)}\right).
```
 This approach does not materially alter the workflow, except that one must also sample the number of replicates before simulating the data.

 Below we define data simulation for a range of sample sizes (i.e., a range of integers) under a discrete uniform prior for ``M``, the random variable corresponding to sample size.

```
function simulate(parameters, m::R) where {R <: AbstractRange{I}} where I <: Integer

	# Number of parameter vectors stored in parameters
	K = size(parameters, 2)

	# Generate K sample sizes from the prior distribution for M
	m̃ = rand(m, K)

	# Pseudocode for data simulation
	Z = [<simulate m̃[k] iid realisations from the model> for k ∈ 1:K]

	return Z
end
```

Then, setting the argument `m` in [`train`](@ref) to be an integer range will train the neural estimator with the given variable sample sizes.





## Bootstrapping

Bootstrapping is a powerful technique for estimating the distribution of an estimator and, hence, facilitating uncertainty quantification. Bootstrap methods are considered to be accurate but often too computationally expensive for traditional likelihood-based estimators, but are well suited to fast neural estimators. We implement bootstrapping with  [`parametricbootstrap`](@ref) and [`nonparametricbootstrap`](@ref), with the latter also catering for so-called block bootstrapping.


## Loading previously saved neural estimators

As training is by far the most computationally demanding part of the workflow, one typically trains an estimator and then saves it for later use. More specifically, one usually saves the *parameters* of the neural estimator (e.g., the weights and biases of the neural networks); then, to load the neural estimator at a later time, one initialises an estimator with the same architecture used during training, and then loads the saved parameters into this estimator.

[`train`](@ref) automatically saves the neural estimator's parameters; to load them, one may use the following code, or similar:

```
θ̂ = architecture()
Flux.loadparams!(θ̂, loadbestweights(path))
```

Above, `architecture()` is a user-defined function that returns a neural estimator with the same architecture as the estimator that we wish to load, but with randomly initialised parameters, and the function `Flux.loadparams!` loads the parameters of the best (as determined [`loadbestweights`](@ref)) neural estimator saved in `path`.
