using Functors: @functor

abstract type NeuralEstimator end

# ---- PointEstimator  ----

struct PointEstimator{F} <: NeuralEstimator
	arch::F
end
@functor PointEstimator (arch,)
(pe::PointEstimator)(Z) = pe.arch(Z)



# ---- CIEstimator: credible intervals  ----

"""
    CIEstimator(lower, upper)

# Examples
```
using NeuralEstimators
using Flux

n = 2  # bivariate data
p = 3  # number of parameters in the model
w = 8  # width of each layer

ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₁ = Chain(Dense(w, w, relu), Dense(w, p));
l = DeepSet(ψ₁, ϕ₁)

ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p));
u = DeepSet(ψ₂, ϕ₂)

ciestimator = CIEstimator(l, u)

m = 10
Z = [rand(n, m) for _ in 1:4]
ciestimator(Z)
confidenceinterval(ciestimator, Z, parameter_names = ["ρ", "σ", "τ"])
```
"""
struct CIEstimator{F, G} <: NeuralEstimator
	l::F
	u::G
end
@functor CIEstimator
(c::CIEstimator)(Z) = vcat(c.l(Z), c.l(Z) .+ exp.(c.u(Z)))



# ---- QuantileEstimator: estimating arbitrary quantiles of the posterior distribution ----

# struct QuantileEstimator <: NeuralEstimator
# 	l::F
# 	u::G
# end
# @functor QuantileEstimator
# (c::CIEstimator)(Z) = vcat(c.l(Z), c.l(Z) .+ exp.(c.u(Z)))


# ---- PiecewiseEstimator: variable sample sizes ----


"""
	PiecewiseEstimator(estimators, breaks)
Creates a piecewise estimator from a collection of `estimators`, based on the
collection of sample-size changepoints, `breaks`, which should contain one
element fewer than the number of `estimators`.

# Examples
```
# Suppose that we've trained two neural estimators. The first, θ̂₁, is trained
# for small sample sizes (e.g., m ≤ 30), and the second, `θ̂₂`, is trained for
# moderate-to-large sample sizes (e.g., m > 30). Then we construct a piecewise
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
