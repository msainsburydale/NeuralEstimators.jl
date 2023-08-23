using Functors: @functor

"""
	NeuralEstimator

An abstract supertype for neural estimators.
"""
abstract type NeuralEstimator end

# ---- PointEstimator  ----

"""
    PointEstimator(arch)

A simple point estimator, that is, a mapping from the sample space to the
parameter space, defined by the given architecture `arch`.
"""
struct PointEstimator{F} <: NeuralEstimator
	arch::F
end
@functor PointEstimator (arch,)
(pe::PointEstimator)(Z) = pe.arch(Z)


# ---- IntervalEstimator: credible intervals  ----

"""
	IntervalEstimator(arch_lower, arch_upper)
	IntervalEstimator(arch)
A neural estimator that produces credible intervals constructed as,

```math
[l(Z), l(Z) + \\mathrm{exp}(u(Z))],
```

where ``l(⋅)`` and ``u(⋅)`` are the neural networks `arch_lower` and
`arch_upper`, both of which should transform data into ``p``-dimensional vectors,
where ``p`` is the number of parameters in the model. If only a single neural
network architecture `arch` is provided, it will be used for both `arch_lower`
and `arch_upper`.

Internally, the output from `arch_lower` and `arch_upper` are concatenated, so
that `IntervalEstimator` objects transform data into ``2p``-dimensional vectors.

# Examples
```
using NeuralEstimators
using Flux

# Generate some toy data
n = 2  # bivariate data
m = 10 # number of independent replicates
Z = rand(n, m)

# Create an architecture
p = 3  # parameters in the model
w = 8  # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
architecture = DeepSet(ψ, ϕ)

# Initialise the interval estimator
estimator = IntervalEstimator(architecture)

# Apply the interval estimator
estimator(Z)
interval(estimator, Z, parameter_names = ["ρ", "σ", "τ"])
```
"""
struct IntervalEstimator{F, G} <: NeuralEstimator
	l::F
	u::G
end
IntervalEstimator(l) = IntervalEstimator(l, deepcopy(l))
@functor IntervalEstimator
(c::IntervalEstimator)(Z) = vcat(c.l(Z), c.l(Z) .+ exp.(c.u(Z)))
# Ensure that IntervalEstimator objects are not constructed with PointEstimator:
IntervalEstimator(l::PointEstimator, u::PointEstimator) = IntervalEstimator(l.arch, u.arch)




# ---- QuantileEstimator: estimating arbitrary quantiles of the posterior distribution ----

# Should Follow up with this point from Gnieting's paper:
# 9.2 Quantile Estimation
# Koenker and Bassett (1978) proposed quantile regression using an optimum score estimator based on the proper scoring rule (41).


#TODO this is a topic of ongoing research with Jordan
"""
    QuantileEstimator()

Coming soon: this structure will allow for the simultaneous estimation of an
arbitrary number of marginal quantiles of the posterior distribution.
"""
struct QuantileEstimator{F, G} <: NeuralEstimator
	l::F
	u::G
end
# @functor QuantileEstimator
# (c::QuantileEstimator)(Z) = vcat(c.l(Z), c.l(Z) .+ exp.(c.u(Z)))



# ---- PiecewiseEstimator ----

"""
	PiecewiseEstimator(estimators, breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of changepoints, `breaks`, which should contain one element fewer
than the number of `estimators`.

Any estimator can be included in `estimators`, including any of the subtypes of
`NeuralEstimator` exported with the package `NeuralEstimators` (e.g., `PointEstimator`,
`IntervalEstimator`, etc.).

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). We construct a piecewise
# estimator with a sample-size changepoint of 30, which dispatches θ̂₁ if m ≤ 30
# and θ̂₂ if m > 30.

using NeuralEstimators
using Flux

n = 2  # bivariate data
p = 3  # number of parameters in the model
w = 8  # width of each layer

ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₁ = DeepSet(ψ₁, ϕ₁)

ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂₂ = DeepSet(ψ₂, ϕ₂)

θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], [30])
Z = [rand(n, 1, m) for m ∈ (10, 50)]
θ̂(Z)
```
"""
struct PiecewiseEstimator <: NeuralEstimator
	estimators
	breaks
	function PiecewiseEstimator(estimators, breaks)
		if length(breaks) != length(estimators) - 1
			error("The length of `breaks` should be one fewer than the number of `estimators`")
		elseif !issorted(breaks)
			error("`breaks` should be in ascending order")
		else
			new(estimators, breaks)
		end
	end
end
@functor PiecewiseEstimator (estimators,)

function (pe::PiecewiseEstimator)(Z)
	# Note that this is an inefficient implementation, analogous to the inefficient
	# DeepSet implementation. A more efficient approach would be to subset Z based
	# on breaks, apply the estimators to each block of Z, then combine the estimates.
	breaks = [pe.breaks..., Inf]
	m = numberreplicates(Z)
	θ̂ = map(eachindex(Z)) do i
		# find which estimator to use, and then apply it
		mᵢ = m[i]
		j = findfirst(mᵢ .<= breaks)
		pe.estimators[j](Z[[i]])
	end
	return stackarrays(θ̂)
end

# Clean printing:
Base.show(io::IO, pe::PiecewiseEstimator) = print(io, "\nPiecewise estimator with $(length(pe.estimators)) estimators and sample size change-points: $(pe.breaks)")
Base.show(io::IO, m::MIME"text/plain", pe::PiecewiseEstimator) = print(io, pe)
