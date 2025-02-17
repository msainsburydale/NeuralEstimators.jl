
# Overview

Neural inferential methods have marked practical appeal, as their implementation is only loosely connected to the statistical or physical model being considered. The workflow when using the package `NeuralEstimators` is as follows:

1. Sample parameters from the prior, $\pi(\boldsymbol{\theta})$, to form training/validation/test parameter sets. Alternatively, define a function to [sample parameters dynamically during training](@ref "On-the-fly and just-in-time simulation"). Parameters are stored as $d \times K$ matrices, with $d$ the dimensionality of the parameter vector and $K$ the number of parameter vectors in the given parameter set. 
1. Simulate data from the model conditional on the above parameter sets, to form training/validation/test data sets. Alternatively, define a function to simulate data dynamically during training. Data are stored as objects of type `Vector{A}`, where each element of the vector is associated with one parameter vector, and the subtype `A` depends on the multivariate structure of the data (e.g., a `Matrix` for unstructured multivariate data, a multidimensional `Array` for gridded data, or a `GNNGraph` for graphical or irregular spatial data).
1. If constructing a neural posterior estimator, choose an [approximate posterior distribution](@ref "Approximate distributions") $q(\boldsymbol{\theta}; \boldsymbol{\kappa})$. 
1. Design and initialise a suitable neural network. The architecture class (e.g., MLP, CNN, GNN) should align with the multivariate structure of the data (e.g., unstructured, grid, graph). The specific input and output spaces depend on the chosen inferential method: 
    * For neural Bayes estimators, the neural network is a mapping $\mathcal{Z}\to\Theta$, where $\mathcal{Z}$ denotes the sample space and $\Theta$ denotes the parameter space.
    * For neural posterior estimators, the neural network is a mapping $\mathcal{Z}\to\mathcal{K}$, where $\mathcal{K}$ denotes the space of the approximate-distribution parameters $\boldsymbol{\kappa}$. 
    * For neural ratio estimators, the neural network is a mapping $\mathcal{Z}\times\Theta\to\mathbb{R}$. 
    
    Any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used to construct the neural network. To integrate it into the workflow, one need only define a method that transforms $K$-dimensional vectors of data (see Step 2 above) into matrices with $K$ columns, where the number of rows corresponds to the dimensionality of the output spaces listed above (see the [Gridded data](@ref) example). The type [`DeepSet`](@ref) serves as a convenient wrapper for embedding standard neural networks (e.g., MLPs, CNNs, GNNs) in a framework for making inference with an arbitrary number of independent replicates, and it comes with pre-defined methods for handling the transformations from a $K$-dimensional vector of data to a matrix output. 
1. Wrap the neural network (and possibly the approximate distribution) in a  [subtype of `NeuralEstimator`](@ref "Estimators") corresponding to the intended inferential method:
    * For neural Bayes estimators under general, user-defined loss functions, use [`PointEstimator`](@ref); 
    * For neural posterior estimators, use [`PosteriorEstimator`](@ref);
    * For neural ratio estimators, use [`RatioEstimator`](@ref). 
1. Train the `NeuralEstimator` using [`train()`](@ref) and the training set, monitoring performance and convergence using the validation set. For generic neural Bayes estimators, specify a [loss function](@ref "Loss functions"). 
1. Assess the `NeuralEstimator` using [`assess()`](@ref) and the test set. 

Once the `NeuralEstimator` has passed our assessments and is deemed to be well calibrated, it may be used to make inference with observed data. 

Next, see the [Examples](@ref) and, once familiar with the basic workflow, see [Advanced usage](@ref) for further practical considerations on how to most effectively construct neural estimators.