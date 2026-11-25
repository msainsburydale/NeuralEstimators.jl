```@meta
CollapsedDocStrings = true
```

## Training

The function [`train()`](@ref) is used to train a neural estimator.

After training, the risk history, the trained neural network, and the optimiser state can be loaded using [`loadrisk()`](@ref), [`plotrisk()`](@ref), [`loadestimator()`](@ref), and [`loadoptimiser()`](@ref). An estimator can also be saved at any time with [`saveestimator()`](@ref).

```@docs
train

loadrisk

plotrisk

loadestimator

saveestimator

loadoptimiser
```