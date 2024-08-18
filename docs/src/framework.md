# Framework

In this section, we provide an overview of point estimation using neural Bayes estimators. For a more detailed discussion on the framework and its implementation, see the paper [Likelihood-Free Parameter Estimation with Neural Bayes Estimators](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522). For an accessible introduction to amortised neural inferential methods more broadly, see the review paper [Neural Methods for Amortised Inference](https://arxiv.org/abs/2404.12484).

### Neural Bayes estimators

A parametric statistical model is a set of probability distributions on a sample space $\mathcal{Z} \subseteq \mathbb{R}^n$, where the probability distributions are parameterised via some parameter vector $\boldsymbol{\theta}$ on a parameter space $\Theta \subseteq \mathbb{R}^p$. Suppose that we have data from one such distribution, which we denote as $\boldsymbol{Z}$. Then, the goal of parameter point estimation is to come up with an estimate of the unknown $\boldsymbol{\theta}$ from $\boldsymbol{Z}$ using an estimator,

```math
 \hat{\boldsymbol{\theta}} : \mathcal{Z} \to \Theta,
```
which is a mapping from the sample space to the parameter space.

Estimators can be constructed within a decision-theoretic framework. Consider a nonnegative loss function, $L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}))$, which assesses an estimator $\hat{\boldsymbol{\theta}}(\cdot)$ for a given $\boldsymbol{\theta}$ and data set $\boldsymbol{Z} \sim f(\boldsymbol{z} \mid \boldsymbol{\theta})$, where $f(\boldsymbol{z} \mid \boldsymbol{\theta})$ is the probability density function of the data conditional on $\boldsymbol{\theta}$. An estimator's *Bayes risk* is its loss averaged over all possible parameter values and data realisations,

```math
\int_\Theta \int_{\mathcal{Z}}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{z}))f(\boldsymbol{z} \mid \boldsymbol{\theta}) \rm{d} \boldsymbol{z} \rm{d} \Pi(\boldsymbol{\theta}),  
```
where $\Pi(\cdot)$ is a prior measure for $\boldsymbol{\theta}$. Any minimiser of the Bayes risk is said to be a *Bayes estimator* with respect to $L(\cdot, \cdot)$ and $\Pi(\cdot)$.

Bayes estimators are theoretically attractive: for example, unique Bayes estimators are admissible and, under suitable regularity conditions and the squared-error loss, are consistent and asymptotically efficient. Further, for a large class of prior distributions, every set of conditions that imply consistency of the maximum likelihood (ML) estimator also imply consistency of Bayes estimators. Importantly, Bayes estimators are not motivated purely by asymptotics: by construction, they are Bayes irrespective of the sample size and model class. Unfortunately, however, Bayes estimators are typically unavailable in closed form for the complex models often encountered in practice. A way forward is to assume a flexible parametric model for $\hat{\boldsymbol{\theta}}(\cdot)$, and to optimise the parameters within that model in order to approximate the Bayes estimator. Neural networks are ideal candidates, since they are universal function approximators, and because they are also fast to evaluate, usually involving only simple matrix-vector operations.

Let $\hat{\boldsymbol{\theta}}(\boldsymbol{Z}; \boldsymbol{\gamma})$ denote a neural network that returns a point estimate from data $\boldsymbol{Z}$, where $\boldsymbol{\gamma}$ contains the neural-network parameters. Bayes estimators may be approximated with $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*)$ by solving the optimisation problem,  

```math
\boldsymbol{\gamma}^*
\equiv
\underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \;
\frac{1}{K} \sum_{k = 1}^K L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{z}; \boldsymbol{\gamma})),
```
whose objective function is a Monte Carlo approximation of the Bayes risk made using a set $\{\boldsymbol{\theta}^{(k)} : k = 1, \dots, K\}$ of parameter vectors sampled from the prior $\Pi(\cdot)$ and, for each $k$, data $\boldsymbol{Z}^{(k)}$ simulated from $f(\boldsymbol{z} \mid  \boldsymbol{\theta})$. Note that this Monte Carlo approximation does not involve evaluation, or knowledge, of the likelihood function.

 The Monte Carlo approximation of the Bayes risk can be straightforwardly minimised with respect to $\boldsymbol{\gamma}$ using back-propagation and stochastic gradient descent. For sufficiently flexible architectures, the point estimator targets a Bayes estimator with respect to $L(\cdot, \cdot)$ and $\Pi(\cdot)$. We therefore call the fitted neural point estimator a  *neural Bayes estimator*. Like Bayes estimators, neural Bayes estimators target a specific point summary of the posterior distribution. For instance, the absolute-error and squared-error loss functions lead to neural Bayes estimators that approximate the posterior median and mean, respectively.

### Construction of neural Bayes estimators

The neural Bayes estimator is conceptually simple and can be used in a wide range of problems where other approaches, such as maximum-likelihood estimation, are computationally infeasible. The estimator also has marked practical appeal, as the general workflow for its construction is only loosely connected to the statistical or physical model being considered. The workflow is as follows:

  1. Define the prior, $\Pi(\cdot)$.
  1. Choose a loss function, $L(\cdot, \cdot)$, typically the mean-absolute-error or mean-squared-error loss.
  1. Design a suitable neural-network architecture for the neural point estimator $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma})$.
  1. Sample parameters from $\Pi(\cdot)$ to form training/validation/test parameter sets.
  1. Given the above parameter sets, simulate data from the model, to form training/validation/test data sets.
  1. Train the neural network (i.e., estimate $\boldsymbol{\gamma}$) by minimising the loss function averaged over the training sets. During training, monitor performance and convergence using the validation sets.
  1. Assess the fitted neural Bayes estimator, $\hat{\boldsymbol{\theta}}(\cdot; \boldsymbol{\gamma}^*)$, using the test set.
