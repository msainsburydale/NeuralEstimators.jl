# Motivation


Definition of an estimator:

```math
\hat{\mathbf{\theta}} : \mathcal{S}^m \to \Theta
```

Permutation invariance:

```math
\hat{\mathbf{\theta}}(\mathbf{Z}_1, \dots, \mathbf{Z}_m) = \hat{\mathbf{\theta}}(\mathbf{Z}_{\pi(1)}, \dots, \mathbf{Z}_{\pi(m)})
```

Under some arbitrary loss function ``L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathcal{Z}))``, the risk function:

```math
R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot)) \equiv \int_{\mathcal{S}^m}  L(\mathbf{\theta}, \hat{\mathbf{\theta}}(\mathcal{Z}))p(\mathcal{Z} \mid \mathbf{\theta}) d \mathcal{Z}
```

Weighted average risk function:


```math
r_{\Omega}(\hat{\mathbf{\theta}}(\cdot))
\equiv \int_\Theta R(\mathbf{\theta}, \hat{\mathbf{\theta}}(\cdot)) d\Omega(\mathbf{\theta}),  
```


Deep Set (Zaheer et al., 2017) representation of an estimator:

```math
\begin{aligned}
\hat{\mathbf{\theta}}(\mathcal{Z}) &= \mathbf{\phi}(\mathbf{T}(\mathcal{Z})) \\
\mathbf{T}(\mathcal{Z})  &= \sum_{\mathbf{Z} \in \mathcal{Z}} \mathbf{\psi}(\mathbf{Z})
\end{aligned}
```

Optimisation task:

```math
\hat{\mathbf{\theta}}_{\mathbf{\gamma}^*}(\cdot):
\mathbf{\gamma}^*
\equiv
\underset{\mathbf{\gamma}}{\mathrm{arg\,min}} \; r_{\Omega}(\hat{\mathbf{\theta}}_{\mathbf{\gamma}}(\cdot)).
```

Monte Carlo approximation of the weighted average risk:

```math
r_{\Omega}(\hat{\mathbf{\theta}}(\cdot))
\approx
\frac{1}{K} \sum_{k = 1}^K \frac{1}{J} \sum_{j = 1}^J L(\mathbf{\theta}_k, \hat{\mathbf{\theta}}(\mathcal{Z}_{kj})).  
```
