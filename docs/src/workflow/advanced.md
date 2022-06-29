# Advanced usage

## Reusing intermediate objects (e.g., Cholesky factors) for multiple parameter configurations


## Balancing time and memory complexity

"On-the-fly" simulation refers to simulating new values for the parameters, θ, and/or the data, Z, continuously during training. "Just-in-time" simulation refers to simulating small batches of parameters and data, training the neural estimator with this small batch, and then removing the batch from memory.   

There are three variants of on-the-fly and just-in-time simulation, each with advantages and disadvantages.

- Resampling θ and Z every epoch. This approach is the most theoretically justified and has the best memory complexity, since both θ and Z can be simulated just-in-time, but it has the worst time complexity.
- Resampling θ every x epochs, resampling Z every epoch. This approach can reduce time complexity if generating θ (or intermediate objects thereof) dominates the computational cost. Further, memory complexity may be kept low since Z can still be simulated just-in-time.
- Resampling θ every x epochs, resampling Z every y epochs, where x is a multiple of y. This approach minimises time complexity but has the largest memory complexity, since both θ and Z must be stored in full. Note that fixing θ and Z (i.e., setting y = ∞) often leads to worse out-of-sample performance and, hence, is generally discouraged.

The keyword arguments `epochs_per_θ_refresh` and `epochs_per_Z_refresh` in `train()` are intended to cater for these simulation variants.

## Loading previously saved estimators

## Piece-wise estimators conditional on the sample size
