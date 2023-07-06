# Theoretical framework

## Neural Bayes estimators

A statistical model is a set of probability distributions $\mathcal{P}$ on a sample space $\mathcal{S}$. A parametric statistical model is one where the probability distributions in $\mathcal{P}$ are parameterised via some $p$-dimensional parameter vector $\boldsymbol{\theta}$, that is, where $\mathcal{P} \equiv \{P_{\boldsymbol{\theta}} : \boldsymbol{\theta} \in \Theta\}$, where $\Theta$ is the parameter space. Suppose that we have $m$ mutually independent realisations from $P_{\boldsymbol{\theta}} \in \mathcal{P}$, which we collect in $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1',\dots,\boldsymbol{Z}_m')'$. Then, the goal of parameter point estimation is to infer the unknown $\boldsymbol{\theta}$ from $\boldsymbol{Z}$ using an estimator,
```math
\hat{\boldsymbol{\theta}} : \mathcal{S}^m \to \Theta,
```
a mapping from $m$ independent realisations from $\mathcal{P}_{\boldsymbol{\theta}}$ to the parameter space.

Estimators can be constructed intuitively within a decision-theoretic framework.
Consider a non-negative loss function, $L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}))$, which assesses an estimator $\hat{\boldsymbol{\theta}}(\cdot)$ for a given $\boldsymbol{\theta}$ and data set $\boldsymbol{Z}$.  
 The estimator's risk function is the loss averaged over all possible data realisations. Assume, without loss of generality, that our sample space is $\mathcal{S} = \mathbb{R}^n$. Then, the risk function is

```math
 R(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\cdot)) \equiv \int_{\mathcal{S}^m}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}))p(\boldsymbol{Z} \mid \boldsymbol{\theta}) {\text{d}} \boldsymbol{Z},
```

where $p(\boldsymbol{Z} \mid \boldsymbol{\theta}) = \prod_{i=1}^mp(\boldsymbol{Z}_i \mid \boldsymbol{\theta})$ is the likelihood function. A ubiquitous approach in estimator design is to minimise a weighted summary of the risk function known as the Bayes risk,

```math
 r_{\Omega}(\hat{\boldsymbol{\theta}}(\cdot))
 \equiv \int_\Theta R(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\cdot)) {\text{d}}\Omega(\boldsymbol{\theta}),  
```

where $\Omega(\cdot)$ is a prior measure which, for ease of exposition, we will assume admits a density $p(\cdot)$ with respect to Lebesgue measure. A minimiser of the Bayes risk is said to be a *Bayes estimator* with respect to $L(\cdot,\cdot)$ and $\Omega(\cdot)$.

 Recently, neural networks have been used to approximate Bayes estimators. Denote such a neural network by $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma})$, where $\boldsymbol{\gamma}$ are the neural-network parameters.  
Then, Bayes estimators may be approximated by $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*)$, where
```math
\boldsymbol{\gamma}^*
\equiv
\underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \; r_{\Omega}(\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma})).
```

The Bayes risk cannot typically be directly evaluated, but it can be approximated using Monte Carlo methods. Specifically, given a set of $K$ parameter vectors sampled from the prior $\Omega(\cdot)$ denoted by $\vartheta$  and, for each $\boldsymbol{\theta} \in \vartheta$, $J$ sets of $m$ mutually independent realisations from $P_{\boldsymbol{\theta}}$ collected in $\mathcal{Z}_{\boldsymbol{\theta}}$,

```math
 r_{\Omega}(\hat{\boldsymbol{\theta}}(\cdot))
 \approx
\frac{1}{K} \sum_{\boldsymbol{\theta} \in \vartheta} \frac{1}{J} \sum_{\boldsymbol{Z} \in \mathcal{Z}_{\boldsymbol{\theta}}} L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z})).  
```

Therefore, the optimisation problem of finding $\boldsymbol{\gamma}^*$, which is typically performed using stochastic gradient descent, can be approximated using simulation from the model, but does not require evaluation or knowledge of the likelihood function. For sufficiently flexible architectures, the point estimator targets a Bayes estimator with respect to $L(\cdot, \cdot)$ and $\Omega(\cdot)$, and will therefore inherit the associated optimality properties, namely, consistency and asymptotic efficiency. We therefore call the fitted neural estimator a *neural Bayes estimator*.


## Neural Bayes estimators for replicated data

Unique Bayes estimators are invariant to permutations of the mutually independent data collected in $\boldsymbol{Z} \equiv (\boldsymbol{Z}_1',\dots,\boldsymbol{Z}_m')'$. Hence, in these cases, we represent our neural estimators in the DeepSets framework, which is a universal representation for permutation-invariant functions. Specifically, we model our neural estimators as

```math
\hat{\boldsymbol{\theta}}(\boldsymbol{Z}) = \boldsymbol{\phi}(\boldsymbol{T}(\boldsymbol{Z})), \quad \boldsymbol{T}(\boldsymbol{Z})  
= \boldsymbol{a}\big(\{\boldsymbol{\psi}(\boldsymbol{Z}_i) : i = 1, \dots, m\}\big),
```
where $\boldsymbol{\phi}: \mathbb{R}^{q} \to \mathbb{R}^p$ and $\boldsymbol{\psi}: \mathbb{R}^{n} \to \mathbb{R}^q$ are neural networks (whose dependence on parameters $\boldsymbol{\gamma}$ is suppressed for notational convenience), and $\boldsymbol{a}: (\mathbb{R}^q)^m \to \mathbb{R}^q$ is a permutation-invariant set function, which is typically elementwise addition, average, or maximum.


## Construction of neural Bayes estimators

The neural Bayes estimators is conceptually simple and can be used in a wide range of problems where other approaches, such as maximum-likelihood estimation, are computationally infeasible. The estimator also has marked practical appeal, as the general workflow for its construction is only loosely connected to the statistical or physical model being considered. The workflow is as follows:

  1. Define the prior, $\Omega(\cdot)$.
  1. Choose a loss function, $L(\cdot, \cdot)$, typically the absolute-error or squared-error loss.
  1. Design a suitable neural-network architecture for the neural point estimator $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma})$.
  1. Sample parameters from $\Omega(\cdot)$ to form training/validation/test parameter sets.
  1. Given the above parameter sets, simulate data from the model, to form training/validation/test data sets.
  1. Train the neural network (i.e., estimate $\boldsymbol{\gamma}$) by minimising the loss function averaged over the training sets. During training, monitor performance and convergence using the validation sets.
  1. Assess the fitted neural Bayes estimator, $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*)$, using the test set.
