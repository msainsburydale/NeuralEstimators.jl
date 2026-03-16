
# Workflow overview {#Workflow-overview}

The typical workflow when using the package is as follows:
1. Sample parameters $\boldsymbol{\theta}$ (from the prior or proposal distribution) to form training/validation/test parameter sets. Parameters are typically stored as $d \times K$ matrices, where $d$ is the dimension of $\boldsymbol{\theta}$ and $K$ is the number of parameter vectors in the given parameter set, though any batchable object is supported.
  
2. Simulate data from the model conditional on the above parameter sets, to form training/validation/test data sets. Simulated data sets are stored as batches in a format amenable to the chosen neural-network architecture (see Step 3).
  
3. Construct a neural network that maps $K$ data sets to a $d^* \times K$ matrix of summary statistics for $\boldsymbol{\theta}$, where $d^*$ is user-specified. The architecture class (e.g., MLP, CNN, GNN, DeepSet) should reflect the structure of the data (e.g., unstructured, grid, graph, replicated). Any [Flux](https://fluxml.ai/Flux.jl/stable/) model can be used.
  
4. Construct a [`NeuralEstimator`](/API/estimators#Estimators) by wrapping the neural network in the subtype corresponding to the intended inferential method ([`PointEstimator`](/API/estimators#NeuralEstimators.PointEstimator), [`PosteriorEstimator`](/API/estimators#NeuralEstimators.PosteriorEstimator), [`RatioEstimator`](/API/estimators#NeuralEstimators.RatioEstimator)).
  
5. Train the `NeuralEstimator` using [`train`](/API/training#NeuralEstimators.train) and the training set, monitoring performance and convergence using the validation set. 
  
6. Assess the `NeuralEstimator` using [`assess`](/API/assessment#NeuralEstimators.assess) and the test set.
  
7. Apply the `NeuralEstimator` to observed data (see [Inference with observed data](/API/inference#Inference-with-observed-data)).
  

For a minimal working example, see [Quick start](/index#Quick-start).
