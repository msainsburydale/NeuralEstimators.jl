```@meta
CollapsedDocStrings = true
```

# Loss functions

When training an estimator of type [`PointEstimator`](@ref), a loss function must be specified that determines the Bayes estimator that will be approximated. In addition to the standard loss functions (e.g., `mae`, `mse`, which allow for the approximation of posterior medians and means, respectively), the following loss functions are provided with the package. 

When the parameters are standardised (keyword argument `standardise_parameters` of [`PointEstimator`](@ref)), the loss is evaluated on the standardised scale. This leaves the Bayes estimator unchanged for a loss that is a sum over the parameters of terms that each depend on the error of an estimate only, through a positively homogeneous function (e.g., `mae`, `mse`, and [`quantileloss`](@ref)), but not otherwise: for example, the constant κ of [`tanhloss`](@ref), and the constant ϵ of [`kpowerloss`](@ref) (with `safeorigin = true`), then act on the standardised scale, and the joint versions of these losses (`joint = true`) also weight each parameter by the reciprocal of its standard deviation. By default, [`train`](@ref) therefore standardises the parameters of a [`PointEstimator`](@ref) only for losses of the former kind, which it checks numerically (see [`PointEstimator`](@ref)).

Parameter bounds (keyword arguments `lower` and `upper` of [`PointEstimator`](@ref)) are imposed by evaluating the loss on an unbounded scale, which leaves the Bayes estimator unchanged only for `mae` and the quantile loss, whose Bayes estimators are posterior medians and quantiles. With any other loss (e.g., `mse` or [`tanhloss`](@ref)), impose the constraints within the neural network itself, with an output layer such as [`Compress`](@ref) or an activation function such as `softplus`, rather than with `lower` and `upper`.

```@docs
tanhloss

kpowerloss

quantileloss

intervalscore
```
