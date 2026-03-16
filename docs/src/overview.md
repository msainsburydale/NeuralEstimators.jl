# Workflow overview

The typical workflow when using the package is as follows:

1. Sample parameters $\boldsymbol{\theta}$ (from the prior or proposal distribution) to form training/validation/test parameter sets. Parameters are typically stored as $d \times K$ matrices, where $d$ is the dimension of $\boldsymbol{\theta}$ and $K$ is the number of parameter vectors in the given parameter set, though any batchable object is supported.
1. Simulate data from the model conditional on the above parameter sets, to form training/validation/test data sets. Simulated data sets are stored as batches in a format amenable to the chosen neural-network architecture (see Step 3).
1. Construct a neural network that maps $K$ data sets to a $d^* \times K$ matrix of summary statistics for $\boldsymbol{\theta}$, where $d^*$ is user-specified. The architecture class (e.g., MLP, CNN, GNN, DeepSet) should reflect the structure of the data (e.g., unstructured, grid, graph, replicated). Any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used.
1. Construct a [`NeuralEstimator`](@ref "Estimators") by wrapping the neural network in the subtype corresponding to the intended inferential method ([`PointEstimator`](@ref), [`PosteriorEstimator`](@ref), [`RatioEstimator`](@ref)).
1. Train the `NeuralEstimator` using [`train`](@ref) and the training set, monitoring performance and convergence using the validation set. 
1. Assess the `NeuralEstimator` using [`assess`](@ref) and the test set.
1. Apply the `NeuralEstimator` to observed data (see [Inference with observed data](@ref "Inference with observed data")).

For a minimal working example, see [Quick start](@ref).