using Functors: @functor
using RecursiveArrayTools: VectorOfArray, convert

# ---- Aggregation (pooling) and misc functions ----

elementwise_mean(X::A) where {A <: AbstractArray{T, N}} where {T, N} = mean(X, dims = N)
elementwise_sum(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = sum(X, dims = N)
elementwise_logsumexp(X::A)  where {A <: AbstractArray{T, N}} where {T, N} = logsumexp(X, dims = N)

function _agg(a::String)
	@assert a ∈ ["mean", "sum", "logsumexp"]
	if a == "mean"
		elementwise_mean
	elseif a == "sum"
		elementwise_sum
	elseif a == "logsumexp"
		elementwise_logsumexp
	end
end

# ---- DeepSet ----

"""
    DeepSet(ψ, ϕ, a)
	DeepSet(ψ, ϕ; a::String = "mean")

The Deep Set representation,

```math
θ̂(𝐙) = ϕ(𝐓(𝐙)),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the model, `ψ` and `ϕ`
are neural networks, and `a` is a permutation-invariant aggregation function.

To make the architecture agnostic to the sample size ``m``, the aggregation
function `a` must aggregate over the replicates. It can be specified as a
positional argument of type `Function`, or as a keyword argument with
permissible values `"mean"`, `"sum"`, and `"logsumexp"`.

`DeepSet` objects act on data stored as `Vector{A}`, where each
element of the vector is associated with one parameter vector (i.e., one set of
independent replicates), and where `A` depends on the form of the data and the
chosen architecture for `ψ`. As a rule of thumb, when the data are stored as an
array, the replicates are stored in the final dimension of the array. (This is
usually the 'batch' dimension, but batching with `DeepSet`s is done at the set
level, i.e., sets of replicates are batched together.) For example, with
gridded spatial data and `ψ` a CNN, `A` should be
a 4-dimensional array, with the replicates stored in the 4ᵗʰ dimension.

Note that, internally, data stored as `Vector{Arrays}` are first
concatenated along the replicates dimension before being passed into the inner
neural network `ψ`; this means that `ψ` is applied to a single large array
rather than many small arrays, which can substantially improve computational
efficiency, particularly on the GPU.

Set-level information, ``𝐱``, that is not a function of the data can be passed
directly into the outer network `ϕ` in the following manner,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐱')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

This is done by providing a `Tuple{Vector{A}, Vector{B}}`, where
the first element of the tuple contains the vector of data sets and the second
element contains the vector of set-level information.

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
w = 32 # width of each layer
ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
ϕ = Chain(Dense(w, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ)

# Apply the estimator
Z₁ = rand(n, 3);                  # single set of 3 realisations
Z₂ = [rand(n, m) for m ∈ (3, 3)]; # two sets each containing 3 realisations
Z₃ = [rand(n, m) for m ∈ (3, 4)]; # two sets containing 3 and 4 realisations
θ̂(Z₁)
θ̂(Z₂)
θ̂(Z₃)

# Repeat the above but with set-level information:
qₓ = 2
ϕ  = Chain(Dense(w + qₓ, w, relu), Dense(w, p));
θ̂  = DeepSet(ψ, ϕ)
x₁ = rand(qₓ)
x₂ = [rand(qₓ) for _ ∈ eachindex(Z₂)]
θ̂((Z₁, x₁))
θ̂((Z₂, x₂))
θ̂((Z₃, x₂))
```
"""
struct DeepSet{T, F, G}
	ψ::T
	ϕ::G
	a::F
end
DeepSet(ψ, ϕ; a::String = "mean") = DeepSet(ψ, ϕ, _agg(a))
@functor DeepSet
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSet) = print(io, D)


# Single data set
function (d::DeepSet)(Z::A) where A
	d.ϕ(d.a(d.ψ(Z)))
end

# Single data set with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	d.ϕ(vcat(t, x))
end

# Multiple data sets: simple fallback method using broadcasting
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where A
  	stackarrays(d.(Z))
end

# Multiple data sets: optimised version for array data.
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa. Note that constructing
	# colons in this manner makes the function agnostic to ndims(ψa).
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Aggregate each set of transformed features: The resulting vector from the
	# list comprehension is a vector of arrays, where the last dimension of each
	# array is of size 1. Then, stack this vector of arrays into one large array,
	# where the last dimension of this large array has size equal to length(v).
	# Note that we cannot pre-allocate and fill an array, since array mutation
	# is not supported by Zygote (which is needed during training).
	t = stackarrays([d.a(ψa[colons..., idx]) for idx ∈ indices])

	# Apply the outer network
	θ̂ = d.ϕ(t)

	return θ̂
end

# Multiple data sets with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z = tup[1]
	x = tup[2]
	t = d.a.(d.ψ.(Z))
	u = vcat.(t, x)
	stackarrays(d.ϕ.(u))
end

# Multiple data sets: optimised version for array data + vector set-level covariates.
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}

	Z = tup[1]
	X = tup[2]

	# Almost exactly the same code as the method defined above, but here we also
	# concatenate the covariates X before passing them into the outer network
	z = stackarrays(Z)
	ψa = d.ψ(z)
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		x = X[i]
		u = vcat(t, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	θ̂ = d.ϕ(u)

	return θ̂
end


# ---- DeepSetExpert: DeepSet with expert summary statistics ----

# Note that this struct is necessary because the Vector{Array} method of
# `DeepSet` concatenates the arrays into a single large array before passing
# the data into ψ.
"""
	DeepSetExpert(ψ, ϕ, S, a)
	DeepSetExpert(ψ, ϕ, S; a::String = "mean")
	DeepSetExpert(deepset::DeepSet, ϕ, S)

Identical to `DeepSet`, but with additional expert summary statistics,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where `S` is a function that returns a vector of expert summary statistics.

The constructor `DeepSetExpert(deepset::DeepSet, ϕ, S)` inherits `ψ` and `a`
from `deepset`.

Similarly to `DeepSet`, set-level information can be incorporated by passing a
`Tuple`, in which case we have

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)', 𝐱')'),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}).
```

# Examples
```
using NeuralEstimators
using Flux

n = 10 # number of observations in each realisation
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
S = samplesize
qₛ = 1
qₜ = 32
w = 16
ψ = Chain(Dense(n, w, relu), Dense(w, qₜ, relu));
ϕ = Chain(Dense(qₜ + qₛ, w), Dense(w, p));
θ̂ = DeepSetExpert(ψ, ϕ, S)

# Apply the estimator
Z₁ = rand(n, 3);                  # single set
Z₂ = [rand(n, m) for m ∈ (3, 4)]; # two sets
θ̂(Z₁)
θ̂(Z₂)

# Repeat the above but with set-level information:
qₓ = 2
ϕ  = Chain(Dense(qₜ + qₛ + qₓ, w, relu), Dense(w, p));
θ̂  = DeepSetExpert(ψ, ϕ, S)
x₁ = rand(qₓ)
x₂ = [rand(qₓ) for _ ∈ eachindex(Z₂)]
θ̂((Z₁, x₁))
θ̂((Z₂, x₂))
```
"""
struct DeepSetExpert{F, G, H, K}
	ψ::G
	ϕ::F
	S::H
	a::K
end
Flux.@functor DeepSetExpert
Flux.trainable(d::DeepSetExpert) = (d.ψ, d.ϕ)
DeepSetExpert(ψ, ϕ, S; a::String = "mean") = DeepSetExpert(ψ, ϕ, S, _agg(a))
DeepSetExpert(deepset::DeepSet, ϕ, S) = DeepSetExpert(deepset.ψ, ϕ, S, deepset.a)
Base.show(io::IO, D::DeepSetExpert) = print(io, "\nDeepSetExpert object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")
Base.show(io::IO, m::MIME"text/plain", D::DeepSetExpert) = print(io, D)

# Single data set
function (d::DeepSetExpert)(Z::A) where {A <: AbstractArray{T, N}} where {T, N}
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s)
	d.ϕ(u)
end

# Single data set with set-level covariates
function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z = tup[1]
	x = tup[2]
	t = d.a(d.ψ(Z))
	s = d.S(Z)
	u = vcat(t, s, x)
	d.ϕ(u)
end

# Multiple data sets: simple fallback method using broadcasting
function (d::DeepSetExpert)(Z::V) where {V <: AbstractVector{A}} where A
  	stackarrays(d.(Z))
end


# Multiple data sets: optimised version for array data.
function (d::DeepSetExpert)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Construct the combined neural and expert summary statistics
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		s = d.S(Z[i])
		u = vcat(t, s)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end

# Multiple data sets with set-level covariates
function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z = tup[1]
	x = tup[2]
	t = d.a.(d.ψ.(Z))
	s = d.S.(Z)
	u = vcat.(t, s, x)
	stackarrays(d.ϕ.(u))
end

# Multiple data sets with set-level covariates: optimised version for array data + vector set-level covariates.
function (d::DeepSetExpert)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}

	Z = tup[1]
	X = tup[2]

	# Convert to a single large Array
	z = stackarrays(Z)

	# Apply the inner neural network to obtain the neural summary statistics
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# concatenate the neural summary statistics with X
	u = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		s = d.S(Z[i])
		x = X[i]
		u = vcat(t, s, x)
		u
	end
	u = stackarrays(u)

	# Apply the outer network
	d.ϕ(u)
end

@doc raw"""
    Compress(a, b, k = 1)
Layer that compresses its input to be within the range `a` and `b`, where each
element of `a` is less than the corresponding element of `b`.

The layer uses a logistic function,

```math
l(θ) = a + \frac{b - a}{1 + e^{-kθ}},
```

where the arguments `a` and `b` together combine to shift and scale the logistic
function to the range (`a`, `b`), and the growth rate `k` controls the steepness
of the curve.

The logistic function given [here](https://en.wikipedia.org/wiki/Logistic_function)
contains an additional parameter, θ₀, which is the input value corresponding to
the functions midpoint. In `Compress`, we fix θ₀ = 0, since the output of a
randomly initialised neural network is typically around zero.

# Examples
```
using NeuralEstimators
using Flux

a = [25, 0.5, -pi/2]
b = [500, 2.5, 0]
p = length(a)
K = 100
θ = randn(p, K)
l = Compress(a, b)
l(θ)

n = 20
θ̂ = Chain(Dense(n, p), l)
Z = randn(n, K)
θ̂(Z)
```
"""
struct Compress{T}
  a::T
  b::T
  k::T
end
Compress(a, b) = Compress(a, b, ones(eltype(a), length(a)))

(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-l.k .* θ))

Flux.@functor Compress
Flux.trainable(l::Compress) =  ()

# ---- Layers to construct Covariance and Correlation matrices ----

# Based on the approach described [here](https://stats.stackexchange.com/a/404453).
# The mappping of `v` to the Cholesky factor is a well behaved diffeomorphism,
# so it can be inverted; however, this inversion is not implemented.
"""
	vectocholesky(v::Vector; correlation::Bool = false)
Transforms a vector `v` ∈ ℝᵈ, where d is a triangular number, into a valid
Cholesky factor for either a correlation matrix or a covariance matrix.
"""
function vectocholesky(v; correlation::Bool = false)
	if correlation
		L = vectotril(v; strict = true)
		L = unitdiagonal(L)
	else
		L = vectotril(v)
	end
	x = rowwisenorm(L)
	L = L ./ x
	LowerTriangular(L)
end
rowwisenorm(A,dims=2) = sqrt.(sum(abs2,A; dims = dims))
unitdiagonal(A) = A - Diagonal(A[diagind(A)]) + I

@doc raw"""
	CorrelationMatrix(d)
Layer for constructing the parameters of an unconstrained `d`×`d` correlation matrix.

The layer transforms a `Matrix` with (`d`-1)`d`÷2 rows into a `Matrix` with
the same dimension.

Internally, the layer constructs a valid Cholesky factor 𝐋 and then extracts
the strict lower triangle from the positive-definite correlation matrix 𝐑 = 𝐋𝐋'.
The lower triangle is extracted and vectorised in line with Julia's column-major
ordering. For example, when modelling the correlation matrix

```math
\begin{bmatrix}
1   & R₁₂ &  R₁₃ \\
R₂₁ & 1   &  R₂₃\\
R₃₁ & R₃₂ & 1\\
\end{bmatrix},
```

the rows of the matrix returned by a `CorrelationMatrix` layer are ordered as

```math
R₂₁, R₃₁, R₃₂,
```

which means that the output can easily be transformed into the implied
correlation matrices using [`vectotril`](@ref) and `Symmetric`.

# Examples
```
using NeuralEstimators
using LinearAlgebra

d = 4
p = d*(d-1)÷2
l = CorrelationMatrix(d)
θ = randn(p, 50)

# returns a matrix of parameters
θ = l(θ)

# convert matrix of parameters to implied correlation matrices
R = map(eachcol(θ)) do y
	R = Symmetric(cpu(vectotril(y, strict = true)), :L)
	R[diagind(R)] .= 1
	R
end
```
"""
struct CorrelationMatrix{T <: Integer, Q}
  d::T
  idx::Q
end
function CorrelationMatrix(d::Integer)
	idx = tril(trues(d, d), -1)
	idx = findall(vec(idx)) # convert to scalar indices
	return CorrelationMatrix(d, idx)
end
function (l::CorrelationMatrix)(x)
	ArrayType = containertype(x)
	p, K = size(x)
	L = [vectocholesky(x[:, k], correlation = true) for k ∈ 1:K]
	R = broadcast(x -> x*permutedims(x), L) # permutedims(x) rather than x' because Transpose/Adjoints don't work well with Zygote
	θ = broadcast(x -> x[l.idx], R)
	θ = hcat(θ...)
	θ = convert(ArrayType, θ)
	return θ
end
(l::CorrelationMatrix)(x::AbstractVector) = l(reshape(x, :, 1))

@doc raw"""
    CovarianceMatrix(d)
Layer for constructing the parameters of an unconstrained `d`×`d` covariance matrix.

The layer transforms a `Matrix` with `d`(`d`+1)÷2 rows into a `Matrix` of the
same dimension.

Internally, the layer constructs a valid Cholesky factor 𝐋 and then extracts
the lower triangle from the positive-definite covariance matrix 𝚺 = 𝐋𝐋'. The
lower triangle is extracted and vectorised in line with Julia's column-major
ordering. For example, when modelling the covariance matrix

```math
\begin{bmatrix}
Σ₁₁ & Σ₁₂ & Σ₁₃ \\
Σ₂₁ & Σ₂₂ & Σ₂₃ \\
Σ₃₁ & Σ₃₂ & Σ₃₃ \\
\end{bmatrix},
```

the rows of the matrix returned by a `CovarianceMatrix` layer will
be ordered as

```math
Σ₁₁, Σ₂₁, Σ₃₁, Σ₂₂, Σ₃₂, Σ₃₃,
```

which means that the output can easily be transformed into the implied
covariance matrices using [`vectotril`](@ref) and `Symmetric`.

# Examples
```
using NeuralEstimators
using LinearAlgebra

d = 4
p = d*(d+1)÷2
θ = randn(p, 50)

l = CovarianceMatrix(d)
θ = l(θ)
Σ = [Symmetric(cpu(vectotril(y)), :L) for y ∈ eachcol(θ)]
```
"""
struct CovarianceMatrix{T <: Integer, G}
  d::T
  idx::G
end
function CovarianceMatrix(d::Integer)
	idx = tril(trues(d, d))
	idx = findall(vec(idx)) # convert to scalar indices
	return CovarianceMatrix(d, idx)
end
function (l::CovarianceMatrix)(x)
	ArrayType = containertype(x)
	p, K = size(x)
	L = [vectocholesky(x[:, k], correlation = false) for k ∈ 1:K]
	Σ = broadcast(x -> x*permutedims(x), L) # permutedims(x) rather than x' because Transpose/Adjoints don't work well with Zygote
	θ = broadcast(x -> x[l.idx], Σ)
	θ = hcat(θ...)
	θ = convert(ArrayType, θ)
	return θ
end
(l::CovarianceMatrix)(x::AbstractVector) = l(reshape(x, :, 1))
