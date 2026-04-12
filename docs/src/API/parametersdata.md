```@meta
CollapsedDocStrings = true
```

# Parameters

Sampled parameters (e.g., from the prior distribution) are often stored as $d \times K$ matrices, where $d$ is the dimension of the parameter vector of interest and $K$ is the number of sampled parameter vectors. However, any batchable object (compatible with [`numobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.numobs)/[`getobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.getobs)) is supported.

It can sometimes be helpful to wrap the parameters in a user-defined type that also stores expensive intermediate objects needed for simulating data (e.g., Cholesky factors). The user-defined type should be a subtype of [`AbstractParameterSet`](@ref), whose only requirement is a field `θ` that stores parameters.

For convenience, parameters can be stored with named dimensions; see, for example, [`NamedMatrix`](@ref).

```@docs
AbstractParameterSet

NamedMatrix
```

# Data

Simulated data sets are stored as mini-batches in a format amenable to the chosen neural-network architecture; the only requirement is compatibility with [`numobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.numobs)/[`getobs`](https://juliaml.github.io/MLUtils.jl/dev/api/#MLCore.getobs). For example, when constructing an estimator from data collected over a two-dimensional grid, one may use a CNN, with each data set stored in the final dimension of a four-dimensional array.

Precomputed (expert) summary statistics can be incorporated by wrapping the simulated data in a [`DataSet`](@ref) object, which couples the raw data with a matrix of summary statistics.

```@docs
DataSet
```