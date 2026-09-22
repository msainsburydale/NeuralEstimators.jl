# Workflow overview

The typical workflow when using the package is as follows:


1. **Sample parameters** $\boldsymbol{\theta}$ (from the prior or proposal distribution) to form training/validation/test parameter sets. 
    - Parameters are typically stored as $d \times K$ matrices, where $d$ is the dimension of $\boldsymbol{\theta}$ and $K$ is the number of parameter vectors in the given parameter set, though any batchable object is supported.
2. **Simulate data** from the model conditional on these parameters, to form training/validation/test data sets. 
    - Simulated data sets are stored as batches in a format amenable to the chosen neural-network architecture (see Step 3).
3. **Construct a neural network** that maps $K$ data sets to a $d^* \times K$ matrix of summary statistics for $\boldsymbol{\theta}$, where $d^*$ is user-specified. 
    - The architecture class (e.g., MLP, CNN, GNN, DeepSet) should reflect the structure of the data (e.g., unstructured, grid, graph, exchangeable). 
    - Any [Flux.jl](https://fluxml.ai/Flux.jl/stable/) or [Lux.jl](https://lux.csail.mit.edu/stable/) model can be used. 
    - User-defined summary statistics can also be incorporated, either alongside the learned summaries or as the sole input to the estimator (see [here](@ref "Expert summary statistics")).
4. **Initialise a neural estimator** by wrapping the neural network in the type corresponding to the intended inferential method ([`PointEstimator`](@ref), [`PosteriorEstimator`](@ref), [`RatioEstimator`](@ref)). These constructors also initialise any additional neural networks required to map the summary statistics from Step 3 to the appropriate output space.
5. **Train** the estimator using [`train`](@ref) and the training set, monitoring performance and convergence using the validation set. 
6. **Assess** the estimator using [`assess`](@ref) and the test set.
7. Use the estimator to make inference from observed data using [`infer`](@ref).

For a minimal working example, see [Quick start](@ref).