# Loss functions

When training an estimator of type [`PointEstimator`](@ref), a loss function must be specified that determines the Bayes estimator that will be approximated. In addition to the standard loss functions provided by `Flux` (e.g., `mae`, `mse`, which allow for the approximation of posterior medians and means, respectively), the following loss functions are provided with the package. 

```@docs
tanhloss

kpowerloss

quantileloss

intervalscore
```
