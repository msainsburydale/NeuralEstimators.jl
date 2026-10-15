```@meta
CollapsedDocStrings = true
```

## Inference

The following functions facilitate the use of a trained neural estimator with observed data.

```@docs
infer
```

### Posterior inference

```@docs
sampleposterior

logratio

posteriormean 

posteriormedian

posteriorquantile

spikeprobability
```

### Point estimation

```@docs
estimate

bootstrap

interval

quantiles
```

### Learned prior distributions

The following functions are used with a trained [`PriorEstimator`](@ref).

```@docs
sample(::PriorEstimator, ::Integer)

logdensity

forward

inverse
```