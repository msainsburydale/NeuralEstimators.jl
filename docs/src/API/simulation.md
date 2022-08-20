# Data simulation

The philosophy of `NeuralEstimators` is to cater for arbitrary statistical models by relying on the user to define the statistical model implicitly, either by providing data simulated from the model or by defining a function for data simulation. The following functions (in particular, their source code) serve as examples for how a user may formulate data simulation code for their own statistical model.

## Model simulators

```@docs
simulategaussianprocess

simulateschlather

simulateconditionalextremes
```

## Density functions

```@docs
gaussiandensity

schlatherbivariatedensity

Subbotin
```

## Miscellaneous functions
```@docs
matern

maternchols

incgamma
```
