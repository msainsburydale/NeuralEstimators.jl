# Model-specific functions


## Data simulators

The philosophy of `NeuralEstimators` is to cater for arbitrary statistical models by having the user define their statistical model implicitly through simulated data. However, the following functions have been included as they may be helpful to others, and their source code illustrates how a user could formulate code for their own model.

See also [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) for a large range of distributions implemented in Julia, and the package [RCall](https://juliainterop.github.io/RCall.jl/stable/) for calling R functions within Julia. 

```@docs
simulategaussianprocess

simulateschlather
```

## Spatial point processes

```@docs
maternclusterprocess
```

## Covariance functions

These covariance functions may be of use for various models.

```@docs
matern

maternchols
```


## Density functions

Density functions are not needed in the workflow of `NeuralEstimators`. However, as part of a series of comparison studies between neural estimators and likelihood-based estimators given in various paper, we have developed the following functions for evaluating the density function for several popular distributions. We include these in `NeuralEstimators` to cater for the possibility that they may be of use in future comparison studies.

```@docs
gaussiandensity

schlatherbivariatedensity
```
