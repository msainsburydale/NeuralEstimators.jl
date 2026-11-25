```@meta
CollapsedDocStrings = true
```

# Miscellaneous

## Core

These functions can appear during the core workflow, and may need to be
overloaded in some applications.

```@docs
numberreplicates

subsetreplicates
```

## Standardisation

By default, the data and the parameters are standardised before they enter the neural networks (the parameters of a [`PointEstimator`](@ref) only if this leaves the estimator unchanged), with statistics computed from the training data and saved with the estimator (see the keyword arguments `standardise_data` and `standardise_parameters` of each estimator).

```@docs
Standardiser

NeuralEstimators.fitstandardiser
```

## Downstream-inference algorithms

```@docs 
EM
```

## Utility functions

```@docs
adjacencymatrix

containertype

encodedata

expandgrid

maternclusterprocess

removedata

rowwisenorm

scaledlogistic

scaledlogit

spatialgraph

stackarrays

vectotril
```