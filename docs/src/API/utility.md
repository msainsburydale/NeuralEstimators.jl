# Miscellaneous


## Core

These functions can appear during the core workflow, and may need to be
overloaded in some applications.

```@docs
subsetparameters

numberreplicates

subsetdata
```

## Architecture layers

These layers can be used at the end of an architecture to ensure that the
neural estimator provides valid parameters.

```@docs
Compress

CholeskyParameters

CovarianceMatrixParameters

CorrelationMatrixParameters

SplitApply
```

## Utility functions

```@docs
vectotril

containertype

loadbestweights

stackarrays

expandgrid
```
