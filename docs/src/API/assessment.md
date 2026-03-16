```@meta
CollapsedDocStrings = true
```

## Post-training assessment

The function [`assess`](@ref) can be used to assess a trained estimator. The resulting [`Assessment`](@ref) object contains ground-truth parameters, estimates, and other quantities that can be used to compute quantitative and qualitative diagnostics.

```@docs
assess

Assessment

plot(assessment::Assessment)

risk

bias

rmse

coverage
```