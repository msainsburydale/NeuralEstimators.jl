# Methodology

Here, we provide an overview of the amortised neural inferential methods supported by the package; for further details, see the review paper by [Zammit-Mangion et al. (2025)](https://arxiv.org/abs/2404.12484) and the references therein.


**Notation:** We denote model parameters of interest by $\boldsymbol{\theta} \equiv (\theta_1, \dots, \theta_d)' \in \Theta$, where $\Theta \subseteq \mathbb{R}^d$ is the parameter space. We denote data by $\boldsymbol{Z} \equiv (Z_1, \dots, Z_n)' \in \mathcal{Z}$, where $\mathcal{Z} \subseteq \mathbb{R}^n$ is the sample space. We denote neural-network parameters by $\boldsymbol{\gamma}$. For simplicity, we assume that all measures admit densities with respect to the Lebesgue measure. We use $\pi(\cdot)$ to denote the prior density function of the parameters. The input argument to a generic density function $p(\cdot)$ serves to specify both the random variable associated with the density and its evaluation point.

## Neural Bayes estimators

The goal of parametric point estimation is to estimate $\boldsymbol{\theta}$ from data $\boldsymbol{Z}$ using an estimator, $\hat{\boldsymbol{\theta}} : \mathcal{Z}\to\Theta$. Estimators can be constructed intuitively within a decision-theoretic framework based on average-risk optimality. Specifically, consider a loss function $L: \Theta \times \Theta \to [0, \infty)$. Then the Bayes risk of the estimator $\hat{\boldsymbol{\theta}}(\cdot)$ is  
```math
\int_\Theta \int_{\mathcal{Z}}  L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}(\boldsymbol{Z}))p(\boldsymbol{Z} \mid \boldsymbol{\theta}) \pi(\boldsymbol{\theta}) \rm{d} \boldsymbol{Z} \rm{d}\boldsymbol{\theta}.  
```
Any minimiser of the Bayes risk is said to be a *Bayes estimator* with respect to $L(\cdot, \cdot)$ and $\pi(\cdot)$. 

Bayes estimators are functionals of the posterior distribution (e.g., the Bayes estimator under quadratic loss is the posterior mean), and are therefore often unavailable in closed form. A way forward is to assume a flexible parametric function for $\hat{\boldsymbol{\theta}}(\cdot)$, and to optimise the parameters within that function in order to approximate the Bayes estimator. Neural networks are ideal candidates, since they are universal function approximators, and because they are fast to evaluate. Let $\hat{\boldsymbol{\theta}}_{\boldsymbol{\gamma}} : \mathcal{Z}\to\Theta$ denote a neural network parameterised by $\boldsymbol{\gamma}$. Then a Bayes estimator may be approximated by $\hat{\boldsymbol{\theta}}_{\boldsymbol{\gamma^*}}(\cdot)$, where 
```math
\boldsymbol{\gamma}^* \equiv \underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \; \frac{1}{K} \sum_{k = 1}^K L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}_{\boldsymbol{\gamma}}(\boldsymbol{Z}^{(k)})),
```
with $\boldsymbol{\theta}^{(k)} \sim \pi(\boldsymbol{\theta})$ and, independently for each $k$, $\boldsymbol{Z}^{(k)} \sim p(\boldsymbol{Z} \mid  \boldsymbol{\theta}^{(k)})$. The process of obtaining $\boldsymbol{\gamma}^*$ is referred to as "training the network", and this can be performed efficiently using back-propagation and stochastic gradient descent. The trained neural network $\hat{\boldsymbol{\theta}}_{\boldsymbol{\gamma^*}}(\cdot)$ approximately minimises the Bayes risk, and therefore it is called a *neural Bayes estimator* [(Sainsbury-Dale at al., 2024)](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522). 

Once trained, a neural Bayes estimator can be applied repeatedly to real data sets at a fraction of the computational cost of conventional inferential methods. It is therefore ideal to use a neural Bayes estimator in settings where inference needs to be made repeatedly; in this case, the initial training cost is said to be amortised over time. 

### Uncertainty quantification with neural Bayes estimators

Uncertainty quantification with neural Bayes estimators often proceeds through the bootstrap distribution (e.g., [Lenzi et al., 2023](https://doi.org/10.1016/j.csda.2023.107762); [Sainsbury-Dale et al., 2024](https://doi.org/10.1080/00031305.2023.2249522); [Richards et al., 2025](https://arxiv.org/abs/2306.15642)). Bootstrap-based approaches are particularly attractive when nonparametric bootstrap is possible (e.g., when the data are independent replicates), or when simulation from the fitted model is fast, in which case parametric bootstrap is also computationally efficient. However, these conditions are not always met and, although bootstrap-based approaches are often considered to be fairly accurate and favourable to methods based on asymptotic normality, there are situations where bootstrap procedures are not reliable (see, e.g., [Canty et al., 2006](https://doi.org/10.1002/cjs.5550340103), pg. 6). 

Alternatively, by leveraging ideas from (Bayesian) quantile regression, one may construct a neural Bayes estimator that approximates a set of marginal posterior quantiles ([Fisher et al., 2023](https://doi.org/10.5705/ss.202020.0348); [Sainsbury-Dale et al., 2025](https://doi.org/10.1080/10618600.2024.2433671)), which can then be used to construct credible intervals for each parameter. Inference then remains fully amortised since, once the estimators are trained, both point estimates and credible intervals can be obtained with virtually zero computational cost. Specifically, posterior quantiles can be targeted by training a neural Bayes estimator under the loss function
```math
L(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}; \tau) \equiv \sum_{j=1}^d (\hat{\theta} - \theta)\{\mathbb{I}(\hat{\theta}_j - \theta_j) - \tau\}, \quad 0 < \tau < 1,
```
where $\mathbb{I}(\cdot)$ denotes the indicator function, since the Bayes estimator under this loss function is the vector of marginal posterior $\tau$-quantiles ([Sainsbury-Dale et al., 2025](https://doi.org/10.1080/10618600.2024.2433671), Sec. 2.2.4). 


## Neural posterior estimators

We now describe amortised approximate posterior inference through the minimisation of an expected Kullback-Leibler (KL) divergence. Throughout, we let $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$ denote a parametric approximation to the posterior distribution $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$, where the approximate-distribution parameters $\boldsymbol{\kappa}$ belong to a space $\mathcal{K}$. 

We first consider the non-amortised case, where the optimal parameters $\boldsymbol{\kappa}^*$ for a single data set $\boldsymbol{Z}$ are found by minimising the KL divergence between $p(\boldsymbol{\theta} \mid \boldsymbol{Z})$ and $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$: 
```math
  \boldsymbol{\kappa}^* \equiv \argmin_{\boldsymbol{\kappa}} \rm{KL}\{p(\boldsymbol{\theta} \mid \boldsymbol{Z}) \, \| \, q(\boldsymbol{\theta} ;\boldsymbol{\kappa})\}. 
```
The resulting approximate posterior $q(\boldsymbol{\theta}; \boldsymbol{\kappa}^*)$ targets the true posterior in the sense that the KL divergence is zero if and only if $q(\boldsymbol{\theta}; \boldsymbol{\kappa}^*) = p(\boldsymbol{\theta} \mid \boldsymbol{Z})$ for all $\boldsymbol{\theta} \in \Theta$. However, solving this optimisation problem is often computationally demanding even for a single data set $\boldsymbol{Z}$, and solving it for many different data sets can be computationally prohibitive. The optimisation problem can be amortised by treating the parameters $\boldsymbol{\kappa}$ as a function $\boldsymbol{\kappa} : \mathcal{Z} \to \mathcal{K}$, and then choosing the function $\boldsymbol{\kappa}^*(\cdot)$ that minimises an expected KL divergence: 
```math
\boldsymbol{\kappa}^*(\cdot) \equiv \argmin_{\boldsymbol{\kappa}(\cdot)} \mathbb{E}_{\boldsymbol{Z}}[\rm{KL}\{p(\boldsymbol{\theta} \mid \boldsymbol{Z}) \, \| \, q(\boldsymbol{\theta} ;\boldsymbol{\kappa}(\boldsymbol{Z}))\}].
```
In practice, we approximate $\boldsymbol{\kappa}^*(\cdot)$ using a neural network $\boldsymbol{\kappa}_{\boldsymbol{\gamma}} : \mathcal{Z} \to \mathcal{K}$ parameterised by $\boldsymbol{\gamma}$, fit by minimising a Monte Carlo approximation of the expected KL divergence above: 
```math
\boldsymbol{\gamma}^* \equiv \argmin_{\boldsymbol{\gamma}} -\sum_{k=1}^K \log q(\boldsymbol{\theta}^{(k)}; \boldsymbol{\kappa}_{\boldsymbol{\gamma}}(\boldsymbol{Z}^{(k)})).
```
Once trained, the neural network $\boldsymbol{\kappa}_{\boldsymbol{\gamma}^*}(\cdot)$ may be used to estimate the optimal approximate-distribution parameters $\boldsymbol{\kappa}^*$ given data $\boldsymbol{Z}$ at almost no computational cost. The neural network $\boldsymbol{\kappa}_{\boldsymbol{\gamma}^*}(\cdot)$, together with the corresponding approximate distribution $q(\cdot; \boldsymbol{\kappa})$, is collectively referred to as a *neural posterior estimator*. 

There are numerous options for the approximate distribution $q(\cdot; \boldsymbol{\kappa})$. For instance, $q(\boldsymbol{\theta};\boldsymbol{\kappa})$ can be modelled as a Gaussian distribution (e.g., [Chan et al., 2018](https://pubmed.ncbi.nlm.nih.gov/33244210/)), where the parameters $\boldsymbol{\kappa} = (\boldsymbol{\mu}', \rm{vech}(\boldsymbol{L})')'$ consist of a $d$-dimensional mean parameter $\boldsymbol{\mu}$ and the $d(d+1)/2$ non-zero elements of the lower Cholesky factor $\boldsymbol{L}$ of a covariance matrix, and the half-vectorization operator $\rm{vech}(\cdot)$ vectorises the lower triangle of its matrix argument. One may also consider Gaussian mixtures (e.g., [Papamakarios \& Murray, 2016](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html)) or trans-Gaussian distributions (e.g., [Maceda et al., 2024](https://arxiv.org/abs/2404.10899)). However, the most widely adopted approach is to model $q(\cdot\,; \boldsymbol{\kappa})$ using a normalising flow (e.g., [Ardizzone et al., 2019](https://openreview.net/forum?id=rJed6j0cKX); [Radev et al., 2022](https://ieeexplore.ieee.org/document/9298920)), excellent reviews for which are given by [Kobyzev et al. (2020)](https://ieeexplore.ieee.org/document/9089305) and [Papamakarios (2021)](https://dl.acm.org/doi/abs/10.5555/3546258.3546315). A particularly popular class of normalising flow is the affine coupling flow (e.g., [Dinh et al., 2016](https://arxiv.org/abs/1605.08803); [Kingma \& Dhariwal, 2018](https://papers.nips.cc/paper_files/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html); [Ardizzone et al., 2019](https://arxiv.org/abs/1907.02392)): since it is a universal density approximator ([Teshima et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/2290a7385ed77cc5592dc2153229f082-Paper.pdf)), it serves as the default and recommended choice for the approximate distribution in this package. 


## Neural ratio estimators

Finally, we describe amortised inference by approximation of the likelihood-to-evidence ratio, 
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv p(\boldsymbol{Z} \mid \boldsymbol{\theta})/p(\boldsymbol{Z}),
```
where $p(\boldsymbol{Z} \mid \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{Z})$
is the marginal likelihood (also known as the model evidence). 

The likelihood-to-evidence ratio is ubiquitous in statistical inference. For example, likelihood ratios of the form $p(\boldsymbol{Z}\mid \boldsymbol{\theta}_0)/p(\boldsymbol{Z}\mid \boldsymbol{\theta}_1)=r(\boldsymbol{Z}, \boldsymbol{\theta}_0)/r(\boldsymbol{Z}, \boldsymbol{\theta}_1)$ are central to hypothesis testing and model comparison, and naturally appear in the transition probabilities of most standard MCMC algorithms used for Bayesian inference. Further, since the likelihood-to-evidence ratio is a prior-free quantity, its approximation facilitates Bayesian inference that require multiple fits of the same model under different prior distributions. 

Unlike the methods discussed earlier, the likelihood-to-evidence ratio might not immediately seem like a quantity well-suited for approximation by neural networks, which are typically trained by minimising empirical risk functions. However, this ratio emerges naturally as a simple transformation of the optimal solution to a standard binary classification problem, derived through the minimisation of an average risk. Specifically, consider a binary classifier $c(\boldsymbol{Z}, \boldsymbol{\theta})$ that distinguishes dependent data-parameter pairs ${(\boldsymbol{Z}', \boldsymbol{\theta}')' \sim p(\boldsymbol{Z}, \boldsymbol{\theta})}$ with class labels $Y=1$ from independent data-parameter pairs ${(\tilde{\boldsymbol{Z}}', \tilde{\boldsymbol{\theta}}')' \sim p(\boldsymbol{Z})p(\boldsymbol{\theta})}$ with class labels $Y=0$, and where the classes are balanced. Then, the Bayes classifier under binary cross-entropy loss is defined as 
```math
c^*(\cdot, \cdot) 
\equiv
\argmin_{c(\cdot, \cdot)} \sum_{y\in\{0, 1\}} \textrm{Pr}(Y = y)  \int_\Theta\int_\mathcal{Z}p(\boldsymbol{Z}, \boldsymbol{\theta} \mid Y = y)L_{\textrm{BCE}}\{y, c(\boldsymbol{Z}, \boldsymbol{\theta})\}\rm{d} \boldsymbol{Z} \rm{d} \boldsymbol{\theta},
```
where $L_{\textrm{BCE}}(y, c) \equiv -y\log(c) - (1 - y) \log(1 - c)$. It can be shown (e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html), App. B)  that the Bayes classifier is given by 
```math
c^*(\boldsymbol{Z}, \boldsymbol{\theta}) = \frac{p(\boldsymbol{Z}, \boldsymbol{\theta})}{p(\boldsymbol{Z}, \boldsymbol{\theta}) + p(\boldsymbol{\theta})p(\boldsymbol{Z})}, \quad \boldsymbol{Z} \in \mathcal{Z}, \boldsymbol{\theta} \in \Theta,
```
and, hence,
```math
r(\boldsymbol{Z}, \boldsymbol{\theta}) = \frac{c^*(\boldsymbol{Z}, \boldsymbol{\theta})}{1 - c^*(\boldsymbol{Z}, \boldsymbol{\theta})}, \quad \boldsymbol{Z} \in \mathcal{Z}, \boldsymbol{\theta} \in \Theta.
```
This connection links the likelihood-to-evidence ratio to the average-risk-optimal solution of a standard binary classification problem, and consequently provides a foundation for approximating the ratio using neural networks. Specifically, let $c_{\boldsymbol{\gamma}}: \mathcal{Z} \times \Theta \to (0, 1)$ denote a neural network parametrised by $\boldsymbol{\gamma}$. Then the Bayes classifier may be approximated by $c_{\boldsymbol{\gamma}^*}(\cdot, \cdot)$, where 
```math
 \boldsymbol{\gamma}^* \equiv \argmin_{\boldsymbol{\gamma}} -\sum_{k=1}^K \Big[\log\{c_{\boldsymbol{\gamma}}(\boldsymbol{Z}^{(k)}, \boldsymbol{\theta}^{(k)})\} +  \log\{1 - c_{\boldsymbol{\gamma}}(\boldsymbol{Z}^{(\sigma(k))}, \boldsymbol{\theta}^{(k)})\} \Big],
```
with $\boldsymbol{\theta}^{(k)} \sim p(\boldsymbol{\theta})$ simulated from a proposal distribution $p(\boldsymbol{\theta})$ that does not necessarily coincide with the prior distribution, $\boldsymbol{Z}^{(k)} \sim p(\boldsymbol{Z} \mid \boldsymbol{\theta}^{(k)})$, and $\sigma(\cdot)$ a random permutation of $\{1, \dots, K\}$. Once the neural network is trained, $r_{\boldsymbol{\gamma}^*}(\boldsymbol{Z}, \boldsymbol{\theta}) \equiv c_{\boldsymbol{\gamma}^*}(\boldsymbol{Z}, \boldsymbol{\theta})\{1 - c_{\boldsymbol{\gamma}^*}(\boldsymbol{Z}, \boldsymbol{\theta})\}^{-1}$, $\boldsymbol{Z} \in \mathcal{Z}, \boldsymbol{\theta} \in \Theta$, may be used to quickly approximate the likelihood-to-evidence ratio, and therefore it is called a *neural ratio estimator*. 

Inference based on a neural ratio estimator may proceed in a frequentist setting via maximum likelihood and likelihood ratios (e.g., [Walchessen et al., 2024](https://doi.org/10.1016/j.spasta.2024.100848)), and in a Bayesian setting by facilitating the computation of transition probabilities in Hamiltonian Monte Carlo and MCMC algorithms (e.g., [Hermans et al., 2020](https://proceedings.mlr.press/v119/hermans20a.html)). Further, an approximate posterior distribution can be obtained via the identity ${p(\boldsymbol{\theta} \mid \boldsymbol{Z})} = \pi(\boldsymbol{\theta}) r(\boldsymbol{\theta}, \boldsymbol{Z})$, and sampled from using standard sampling techniques (e.g., [Thomas et al., 2022](https://doi.org/10.1214/20-BA1238)).