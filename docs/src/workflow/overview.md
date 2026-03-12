
# Overview

Neural inferential methods have marked practical appeal, as their implementation is only loosely connected to the statistical or physical model being considered. The workflow when using the package `NeuralEstimators` is as follows:

1. Sample parameters from the prior, $\pi(\boldsymbol{\theta})$, to form training/validation/test parameter sets. Alternatively, define a function to [sample parameters dynamically during training](@ref "On-the-fly and just-in-time simulation"). Parameters are typically stored as $d \times K$ matrices, with $d$ the dimensionality of the parameter vector and $K$ the number of parameter vectors in the given parameter set, though any batchable object is supported.

1. Simulate data from the model conditional on the above parameter sets, to form training/validation/test data sets. Alternatively, define a function to [simulate data dynamically during training](@ref "On-the-fly and just-in-time simulation"). Simulated data sets are stored as mini-batches in a format amenable to the chosen neural-network architecture (see Step 3). 
1. Design and initialise a suitable neural network that maps data to $\mathbb{R}^{d^*}$ (i.e., learned summary statistics of user-specified dimension $d^*$). The architecture class (e.g., MLP, CNN, GNN, DeepSet) should reflect the structure of the data (e.g., unstructured, grid, graph, replicated). Given $K$ data sets stored appropriately (see Step 2), the neural network should output a $d^* \times K$ matrix. Any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used to construct the neural network.
1. Construct a [`NeuralEstimator`](@ref "Estimators") by wrapping the neural network in the subtype corresponding to the intended inferential method:
    * [`PointEstimator`](@ref): neural Bayes estimators under user-defined loss functions;
    * [`PosteriorEstimator`](@ref): neural posterior estimators (also choose an 
      [approximate distribution](@ref "Approximate distributions"));
    * [`RatioEstimator`](@ref): neural ratio estimators.
1. Train the `NeuralEstimator` using [`train()`](@ref) and the training set, monitoring performance and convergence using the validation set. For generic neural Bayes estimators, specify a [loss function](@ref "Loss functions"). 
1. Assess the `NeuralEstimator` using [`assess()`](@ref) and the test set. 

Once the `NeuralEstimator` has passed our assessments and is deemed to be well calibrated, it may be used to make [inference with observed data](@ref "Inference with observed data"). 

Next, see the [Examples](@ref) and, once familiar with the basic workflow, see [Advanced usage](@ref) for further practical considerations on how to most effectively construct neural estimators.