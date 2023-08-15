# Model-specific functions


## Data simulators

The philosophy of `NeuralEstimators` is to cater for arbitrary statistical models by having the user define their statistical model implicitly through simulated data. However, the following functions have been included as they may be helpful to others, and their source code provide an example for how a user could formulate code for their own model. If you've developed similar functions that you think may be helpful to others, please get in touch or make a pull request.

```@docs
simulategaussianprocess

simulateschlather
```

## Spatial point processes

```@docs
maternclusterprocess
```

## Low-level functions

These low-level functions may be of use for various models.

```@docs
matern

maternchols
```


## Density functions

Density functions are not needed in the workflow of `NeuralEstimators`. However, as part of a series of comparison studies between neural estimators and likelihood-based estimators given in the manuscript, we have developed the following density functions, and we include them in `NeuralEstimators` to cater for the possibility that they may be of use in future comparison studies.

```@docs
gaussiandensity

schlatherbivariatedensity
```
