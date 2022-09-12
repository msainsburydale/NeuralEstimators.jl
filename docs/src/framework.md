# Framework

## Parameter estimation

A statistical model is a set of probability distributions $\mathcal{P}$ on a sample space $\mathcal{S}$. A parametric statistical model is one where the probability distributions in $\mathcal{P}$ are parameterised via some $p$-dimensional parameter vector $\mathbf{\theta}$, that is, where $\mathcal{P} \equiv \{P_\mathbf{\theta} : \mathbf{\theta} \in \Theta\}$, where $\Theta$ is the parameter space. Suppose that we have $m$ mutually independent realisations from $P_\mathbf{\theta} \in \mathcal{P}$, which we collect in $\mathbf{Z} \equiv (\mathbf{Z}_1',\dots,\mathbf{Z}_m')'$. Then, the goal of parameter estimation is to infer the unknown $\mathbf{\theta}$ from $\mathbf{Z}$ using an estimator,
```math
\hat{\mathbf{\theta}} : \mathcal{S}^m \to \Theta.
```

## Bayes estimators

Estimators can be constructed intuitively within a decision-theoretic framework.
Consider a non-negative loss function, $L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathbf{Z}))$, which quantifies the quality of an estimator $\hat{\mathbf{\theta}}(\cdot)$ for a given $\mathbf{\theta}$ and data set $\mathbf{Z}$.  
 The estimator's risk function is the loss averaged over all possible data realisations. Assume, without loss of generality, that our sample space is $\mathcal{S} = \mathbb{R}^n$. Then, the risk function is

```math
 R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot)) \equiv \int_{\mathcal{S}^m}  L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathbf{Z}))p(\mathbf{Z} \mid \mathbf{\theta}) d \mathbf{Z},
```

where $p(\mathbf{Z} \mid \mathbf{\theta}) = \prod_{i=1}^mp(\mathbf{Z}_i \mid \mathbf{\theta})$ is the likelihood function. Now, a ubiquitous approach in estimator design is to minimise a weighted summary of the risk function known as the Bayes risk,

```math
 r_{\Omega}(\hat{\mathbf{\theta}}(\cdot))
 \equiv \int_\Theta R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot)) d\Omega(\mathbf{\theta}),  
```

where $\Omega(\cdot)$ is a prior measure which, for ease of exposition, we will assume admits a density $p(\cdot)$ with respect to Lebesgue measure. The Bayes risk cannot typically be directly evaluated, but it can be approximated using Monte Carlo methods. Specifically, given a set of $K$ parameters sampled from the prior $\Omega(\cdot)$ denoted by $\vartheta$  and, for each $\mathbf{\theta} \in \vartheta$, $J$ sets of $m$ mutually independent realisations from $P_{\mathbf{\theta}}$ collected in $\mathcal{Z}_{\mathbf{\theta}}$, then

```math
 r_{\Omega}(\hat{\mathbf{\theta}}(\cdot))
 \approx
\frac{1}{K} \sum_{\mathbf{\theta} \in \vartheta} \bigg(\frac{1}{J} \sum_{\mathbf{Z} \in \mathcal{Z}_{\mathbf{\theta}}} L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathbf{Z}))\bigg).  
```

A minimiser of the Bayes risk is said to be a *Bayes estimator* with respect to $L(\cdot,\cdot)$ and $\Omega(\cdot)$.


## Neural Bayes estimators


Unique Bayes estimators are invariant to permutations of the conditionally independent data $\mathbf{Z}$. Hence, we represent our neural estimators in the Deep Set framework, which is a universal representation for permutation-invariant functions. Specifically, we model our neural estimators as

```math
\hat{\mathbf{\theta}}(\mathbf{Z}; \mathbf{\gamma}) = \mathbf{\phi}(\mathbf{T}(\mathbf{Z}; \mathbf{\gamma}); \mathbf{\gamma}), \quad \mathbf{T}(\mathbf{Z}; \mathbf{\gamma})  
= \mathbf{a}\big(\{\mathbf{\psi}(\mathbf{Z}_i; \mathbf{\gamma}) : i = 1, \dots, m\}\big).
```
where $\mathbf{\phi}: \mathbb{R}^{q} \to \mathbb{R}^p$ and $\mathbf{\psi}: \mathbb{R}^{n} \to \mathbb{R}^q$ are neural networks whose parameters are collected in $\mathbf{\gamma}$, and $\mathbf{a}: (\mathbb{R}^q)^m \to \mathbb{R}^q$ is a permutation-invariant set function (typically elementwise addition, average, or maximum). Then, our neural estimator is $\hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}^*)$, where
```math
\mathbf{\gamma}^*
\equiv
\underset{\mathbf{\gamma}}{\mathrm{arg\,min}} \; r_{\Omega}(\hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma})),
```
with the Bayes risk approximated using Monte Carlo methods.
Since the resulting neural estimator minimises (a Monte Carlo approximation of) the Bayes risk, we call it a *neural Bayes estimator*.


## Construction of neural Bayes estimators

The neural Bayes estimator is conceptually simple and can be used in a wide range of problems where other approaches, such as maximum-likelihood estimation, are computationally infeasible. The estimator also has marked practical appeal, as the general workflow for its construction is only loosely connected to the statistical or physical model being considered. The workflow is as follows:
  1. Define $\Omega(\cdot)$, the prior distribution for $\mathbf{\theta}$.
  1. Sample parameters from $\Omega(\cdot)$ to form sets of parameters $\vartheta_{\rm{train}}$, $\vartheta_{\rm{val}}$, and $\vartheta_{\rm{test}}$.
  1.  Simulate data from the model, $\mathcal{P}$, using these sets of parameters, yielding the data sets $\mathcal{Z}_{\rm{train}}$, $\mathcal{Z}_{\rm{val}}$, and $\mathcal{Z}_{\rm{test}}$, respectively.
  1. Choose a loss function $L(\cdot, \cdot)$.
  1. Design neural network architectures for $\mathbf{\phi}(\cdot; \mathbf{\gamma})$ and $\mathbf{\psi}(\cdot; \mathbf{\gamma})$.
  1. Using the training sets $\mathcal{Z}_{\textrm{train}}$ and $\vartheta_{\rm{train}}$, train the neural network under $L(\cdot,\cdot)$ to obtain the neural Bayes estimator, $\hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}^*)$. During training, continuously monitor progress based on $\mathcal{Z}_{\textrm{val}}$ and $\vartheta_{\rm{val}}$.
  1. Assess $\hat{\mathbf{\theta}}(\cdot; \mathbf{\gamma}^*)$ using $\mathcal{Z}_\textrm{test}$ and $\vartheta_{\rm{test}}$.
