module NeuralEstimators

# Write your package code here.

export func

"""
# General work flow

The first step is to define Ω, the prior distribution for the p-dimensional
parameter vector, θ. This can be an object of any type. We also need a method
`sample(Ω, K)` that returns K samples of θ as a p × K matrix.
```
Ω = Normal(0, 0.5)
sample(Ω, K) = rand(Ω, K)'
```

Next, we define a `struct`, whose name is arbitrary, which must have the field
θ but can store other information. Storing information in a `struct` is
somewhat redundant for the current simple model but is useful for more
complicated models, as will be shown later, and it is needed for the different
variants of on-the-fly and just-in-time simulation, also discussed later.
So, for consistency, we require it across the board. We also define a constructor.
```
struct Parameters <: ParameterConfigurations
    θ
end
```

Next, we implicitly define the statistical model by providing a simulate() method
for data simulation conditional on θ. The function takes two arguments, θ and `m`, with the
type of `m` depending on the values the estimator will be used for (e.g.,
a single sample size, in which case m will be an Integer). Irrespective of the
type of m, simulate() should return a a vector of (multi-dimensional) arrays.
```
simulate(params::Parameters, m::Integer) = [rand(Normal(t, 1), θ) for t ∈ params.θ]
```

We then choose an architecture for modelling ψ(⋅) and ϕ(⋅) in the Deep Set framework,
and initialise the neural estimator as a DeepSet object.
```
p = 1
w = 32
q = 16
ψ = Chain(Dense(n, w, relu), Dense(w, q, relu))
ϕ = Chain(Dense(q, w, relu), Dense(w, p), flatten)
θ̂ = DeepSet(ψ, ϕ)
```

Train the neural estimator using `train()`. This optimisation is performed with
respect to an arbitrary loss function L (default absolute error loss).
```
θ̂ = train(θ̂, Ω, m = 1, ...)
```

θ̂ now approximates the Bayes estimator for θ. The performance of θ̂ can be
tested using the function estimate(), which tests θ̂ on a set of testing
parameters sampled from Ω. The estimates are saved as a .csv file for convenience.
```
estimate(θ̂, Ω, ...)
```

## Variants of on-the-fly and just-in-time simulation

The above approach assumes that θ and Z are continuously refreshed every epoch.
This approach is the simplest, most theoretically justified, and has the best memory
complexity, since both θ and Z can be simulated just-in-time; however, it is
also the most time expensive. There are two alternatives to this approach:

- Refresh θ every x epochs, refresh Z every epoch. This can reduce time complexity if generating parameters involves computationally expensive terms, such as Cholesky factors, and memory complexity may be kept low since Z can still be simulated just-in-time.
- refresh θ every x epochs, refresh Z every y>x epochs. This minimises time complexity but has the large memory complexity, since both θ and Z cannot be simulated just-in-time and must be stored in full.

To cater for these variants, another `train()` method is available, which
takes the additional argument `params` (discussed below), and the keyword
arguments `epochs_per_θ_refresh` and `epochs_per_Z_refresh`.

The argument `params` should be a struct whose only requirement is a field
named θ, which stores the parameters. In addition to being needed for the
simulation variants discussed above, this struct is also useful in more
complicated models when re-using expensive terms, like Cholesky factors, can
significantly reduce the computational burden. The name of the struct is
arbitrary, but it is usually convenient to name it in reflection of the statistic model.
```
struct NormalParameterConfigurations
    θ
end
```

Then, the neural estimator is trained as before.
```
θ̂ = train(θ̂, Ω, NormalParameterConfigurations, epochs_per_θ_refresh = 5, epochs_per_Z_refresh = 10)
```
"""
func(x) = x^2

end


# TODO Would be much neater if we only had a single method! Perhaps it's worth
# having a slightly more complicated workflow for very simple models? I'm not sure!


# General idea:
# - Ω contains everything needed for sampling parameters, including
# objects like distance matrices.
# - Parameters are used to store the sampled parameters, as well as
# any objects associated with the parameters that may be useful for data
# simulation (e.g., Cholesky factors).

# # Very simple example:
# Ω = Normal(0, 0.5)
# struct Parameters θ end
# function sample(Ω, K::Integer)
#     θ = rand(Ω, K)'
#     Parameters(θ)
# end
#
# # More complicated (e.g., ConditionalExtremesParameterConfigurations):
# Ω = (
# 	# parameters associated with a(.) and b(.)
# 	κ = (p = rlunif, min = 1.0, max = 2.0),
# 	λ = (p = rlunif, min = 2.0, max = 5.0),
# 	β = (p = rlunif, min = 0.05, max = 2.0),
# 	# Covariance parameters associated with the Gaussian process
#     ρ  = (p = rlunif, min = 2.0, max = 7.0),
# 	ν  = (p = rlunif, min = 0.4, max = 2.5),
# 	# Parameters of the Subbotin distribution
# 	μ  = (p = runif, min = -1.0, max = 1.0),
# 	τ  = (p = rlunif, min = 0.3, max = 0.9),
# 	δ₁ = (p = rlunif, min = 1.3, max = 7.0)
# )
#
#
# # Distance matrix would be a data simulation object, no?
# Ω = Normal(0, 0.5)
# function sample(Ω, K::Integer)
#     θ = rand(Ω, K)'
#     Parameters(θ)
# end
#
#
# #FIXME How would I get real information into these objects? E.g., how would I
# # get the spatial locations of a real data set into this workflow? That's very
# # important; we aren't always in a simulation setting! Maybe a struct should be
# # provided that contains all information needed for data simulation, including
# # the prior. Or, should this just be a tuple or something?
# # Yes, I think a tuple of objects makes sense... Whatever stays constant during
# # training (e.g., the prior, distance matrices, etc.). Sample will then act on
# # this tuple to produce ParameterConfigurations object.
# Ω = Normal(0, 0.5)
# constant_objects = (Ω = Ω)
# struct ParameterConfigurations θ end
# sample(t, K) = ParameterConfigurations(rand(t.Ω, K)')
#
#
# Ω = Normal(0, 0.5)
# struct ParameterConfigurations θ end
# sample(Ω, K::Integer) = ParameterConfigurations(rand(Ω, K)')
#
# # No, I think the prior should be treated specially.
# constant_objects = (Ω = Ω)
