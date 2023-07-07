# Overview

To develop a neural estimator with `NeuralEstimators.jl`,

- Sample parameters from the prior distribution: the parameters are stored as $p \times K$ matrices, with $p$ the number of parameters in the model and $K$ the number of samples (i.e., parameter configurations) in the given parameter set (i.e., training, validation, or test set).
- Simulate data from the assumed model over the parameter sets generated above. These data are stored as a `Vector{A}`, with each element of the vector associated with one parameter configuration, and where `A` depends on the representation of the neural estimator (e.g., an `Array` for CNN-based estimators, or a `GNNGraph` for GNN-based estimators).
- Initialise a neural network, `θ̂`, that will be trained into a neural Bayes estimator.  
- Train `θ̂` under the chosen loss function using [`train`](@ref).
- Assess `θ̂` using [`assess`](@ref). The resulting object of class [`Assessment`](@ref) can be used to assess the estimator with respect to the entire parameter space by estimating the risk function with [`risk`](@ref), or used to plot the empirical sampling distribution of the estimator.
- Apply `θ̂` to observed data (once its performance has been checked in the above step). Bootstrap-based uncertainty quantification is facilitated with [`bootstrap`](@ref) and [`interval`](@ref). 

See the [Examples](@ref) and, once familiar with the basic workflow, see [Advanced usage](@ref) for practical considerations on how to most effectively construct neural estimators.
