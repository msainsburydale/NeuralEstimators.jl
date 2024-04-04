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

#TODO Optimised version with expert summary statistics that can be applied to replicates independently (i.e., they can be used like ψ). It might need some special functionality, where there is a second aggregation function associated with the expert summary statistics.
#TODO In the constructor for DeepSet, check if Parallel appears in any layers of ψ. If it does, provide the information in the comment below:

# Note that, internally, data stored as `Vector{Arrays}` are first
# concatenated along the replicates dimension before being passed into the inner
# neural network `ψ`; this means that `ψ` is applied to a single large array
# rather than many small arrays, which can substantially improve computational
# efficiency, particularly on the GPU.

"""
	(S::Vector{Function})(z)
Method allows a vector of vector-valued functions to be applied to a single
input `z` and then concatenated, which allows users to provide a vector of
functions as a user-defined summary statistic in [`DeepSet`](@ref) objects.

Examples
```
f(z) = rand32(2)
g(z) = rand32(3) .+ z
S = [f, g]
S(1)
```
"""
(S::Vector{Function})(z) = vcat([s(z) for s ∈ S]...)
# (S::Vector)(z) = vcat([s(z) for s ∈ S]...) # can use a more general construction like this to allow for vectors of NeuralEstimators to be called in this way

"""
    DeepSet(ψ, ϕ, a; S = nothing)
	DeepSet(ψ, ϕ; a::String = "mean", S = nothing)
The DeepSets representation,

```math
θ̂(𝐙) = ϕ(𝐓(𝐙)),	 	 𝐓(𝐙) = 𝐚(\\{ψ(𝐙ᵢ) : i = 1, …, m\\}),
```

where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates from the statistical model,
`ψ` and `ϕ` are neural networks, and `a` is a permutation-invariant function.
Expert summary statistics can be incorporated as,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)')'),
```

where `S` is a function that returns a vector of user-defined summary statistics.
These user-defined summary statistics are typically provided either as a
`Function` that returns a `Vector`, or a vector of such functions.

To ensure that the architecture is agnostic to the sample size ``m``, the
aggregation function `a` must aggregate over the replicates. It can be specified
as a positional argument of type `Function`, or as a keyword argument with
permissible values `"mean"`, `"sum"`, and `"logsumexp"`.

`DeepSet` objects act on data of type `Vector{A}`, where each
element of the vector is associated with one data set (i.e., one set of
independent replicates from the statistical model), and where the type `A`
depends on the form of the data and the chosen architecture for `ψ`.
As a rule of thumb, when `A` is an array, the replicates are stored in the final
dimension. The final dimension is usually the 'batch' dimension, but batching with `DeepSet`
objects is done at the data set level (i.e., sets of replicates are batched
together). For example, with gridded spatial data and `ψ` a CNN, `A` should be a
4-dimensional array, with the replicates stored in the 4ᵗʰ dimension.

Set-level information, ``𝐱``, that is not a function of the data can be passed
directly into the outer network `ϕ` in the following manner,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐱')'),	 	 
```

or, in the case that expert summary statistics are also used,

```math
θ̂(𝐙) = ϕ((𝐓(𝐙)', 𝐒(𝐙)', 𝐱')').	 
```

This is done by providing a `Tuple{Vector{A}, Vector{Vector}}`, where
the first element of the tuple contains a vector of data sets and the second
element contains a vector of set-level information (i.e., one vector for each
data set).

# Examples
```
using NeuralEstimators, Flux

n = 10 # dimension of each replicate
p = 4  # number of parameters in the statistical model

# Construct the neural estimator
S = samplesize
qₛ = 1  # dimension of expert summary statistic
qₜ = 16 # dimension of neural summary statistic
w = 16 # width of each hidden layer
ψ = Chain(Dense(n, w, relu), Dense(w, qₜ, relu));
ϕ = Chain(Dense(qₜ + qₛ, w, relu), Dense(w, p));
θ̂ = DeepSet(ψ, ϕ, S = S)

# Toy data
Z = [rand32(n, m) for m ∈ (3, 4)]; # two data sets containing 3 and 4 replicates

# Apply the data
θ̂(Z)

# Inference with set-level information
qₓ = 2 # dimension of set-level vector
ϕ  = Chain(Dense(qₜ + qₛ + qₓ, w, relu), Dense(w, p));
θ̂  = DeepSet(ψ, ϕ; S = S)
x  = [rand32(qₓ) for _ ∈ eachindex(Z)]
θ̂((Z, x))
```
"""
struct DeepSet{T, F, G, K}
	ψ::T
	ϕ::G
	a::F
	S::K
end
@layer DeepSet
DeepSet(ψ, ϕ, a; S = nothing) = DeepSet(ψ, ϕ, a, S)
DeepSet(ψ, ϕ; a::String = "mean", S = nothing) = DeepSet(ψ, ϕ, _agg(a), S)
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")

# Single data set
function (d::DeepSet)(Z::A) where A
	d.ϕ(summary(d, Z))
end
function summary(d::DeepSet, Z::A) where A
	t = d.a(d.ψ(Z))
	if !isnothing(d.S)
		s = d.S(Z)
		t = vcat(t, s)
	end
	return t
end

# Single data set with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z, x = tup
	t = summary(d, Z)
	u = vcat(t, x)
	d.ϕ(u)
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractMatrix{T}} where T
	Z, x = tup
	if size(x, 2) == 1
		# Catches the simple case that the user accidentally passed an Nx1 matrix
		# rather than an N-dimensional vector. Also used by RatioEstimator.
		d((Z, vec(x)))
	else
		# Designed for situations where we have a fixed data set and want to
		# evaluate the deepset object for many different set-level information
		t = summary(d, Z) # summary statistics only need to be computed once
		tx = vcat(repeat(t, 1, size(x, 2)), x) # NB ideally we'd avoid copying t so many times here, using @view
		d.ϕ(tx) # Sanity check: stackarrays([d((Z, vec(x̃))) for x̃ in eachcol(x)])
	end
end

# Multiple data sets: simple fallback method using broadcasting
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where A
  	stackarrays(d.(Z))
end

# Multiple data sets: optimised version for array data.
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	# Convert to a single large array
	z = stackarrays(Z)

	# Apply the inner neural network
	ψa = d.ψ(z)

	# Compute the indices needed for aggregation and construct a tuple of colons
	# used to subset all but the last dimension of ψa.
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)

	# Construct the summary statistics
	# NB for some reason, with the new "explicit" gradient() required by
	# Flux/Zygote, an error is caused if one uses the same variable name outside
	# and inside a broadcast like this. For instance, if for the object "stats"
	# below we had called it "t", then error would be thrown by gradient().
	stats = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		if !isnothing(d.S)
			s = d.S(Z[i])
			t = vcat(t, s)
		end
		t
	end

	# Stack into a single array and apply the outer network
	return d.ϕ(stackarrays(stats))
end

# Multiple data sets with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z, x = tup
	t = d.a.(d.ψ.(Z))
	if !isnothing(d.S)
		s = d.S.(Z)
		t = vcat.(t, s)
	end
	# TODO can the above be replaced by?: t = summary.(Ref(d), Z)
	t = vcat.(t, x)
	stackarrays(d.ϕ.(t))
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractMatrix{T}} where {A, T}
	Z, x = tup
	if size(x, 2) == length(Z)
		# Catches the simple case that the user accidentally passed an NxM matrix
		# rather than an M-dimensional vector of N-vector.
		# Also used by RatioEstimator.
		d((Z, eachcol(x)))
	else
		# Designed for situations where we have a several data sets and we want
		# to evaluate the deepset object for many different set-level information
		[d((z, x)) for z in Z]
	end
end

# Multiple data sets: optimised version for array data + vector set-level covariates.
# (basically the same code as array method without covariates)
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A <: AbstractArray{T, N}, B <: AbstractVector{T}} where {T, N}
	Z, X = tup
	z = stackarrays(Z)
	ψa = d.ψ(z)
	indices = _getindices(Z)
	colons  = ntuple(_ -> (:), ndims(ψa) - 1)
	stats = map(eachindex(Z)) do i
		idx = indices[i]
		t = d.a(ψa[colons..., idx])
		if !isnothing(d.S)
			s = d.S(Z[i])
			t = vcat(t, s)
		end
		u = vcat(t, X[i])
		u
	end
	d.ϕ(stackarrays(stats))
end




# ---- Activation functions -----

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
using NeuralEstimators, Flux

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
  # TODO should check that b > a
end
Compress(a, b) = Compress(float.(a), float.(b), ones(eltype(float.(a)), length(a)))
Compress(a::Number, b::Number) = Compress([float(a)], [float(b)])
(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-l.k .* θ))
@layer Compress
Flux.trainable(l::Compress) =  ()


#TODO documentation and unit testing
export TruncateSupport
struct TruncateSupport
	a
	b
	p::Integer
end
function (l::TruncateSupport)(θ::AbstractMatrix)
	p = l.p
	m = size(θ, 1)
	@assert m ÷ p == m/p "Number of rows in the input must be a multiple of the number of parameters in the statistical model"
	r = m ÷ p
	idx = repeat(1:p, inner = r)
	y = [tuncatesupport.(θ[i:i, :], Ref(l.a[idx[i]]), Ref(l.b[idx[i]])) for i in eachindex(idx)]
	reduce(vcat, y)
end
TruncateSupport(a, b) = TruncateSupport(float.(a), float.(b), length(a))
TruncateSupport(a::Number, b::Number) = TruncateSupport([float(a)], [float(b)], 1)
Flux.@functor TruncateSupport
Flux.trainable(l::TruncateSupport) = ()
tuncatesupport(θ, a, b) = min(max(θ, a), b)

# ---- Layers to construct Covariance and Correlation matrices ----

triangularnumber(d) = d*(d+1)÷2

@doc raw"""
    CovarianceMatrix(d)
	(object::CovarianceMatrix)(x::Matrix, cholesky::Bool = false)
Transforms a vector 𝐯 ∈ ℝᵈ to the parameters of an unconstrained `d`×`d`
covariance matrix or, if `cholesky = true`, the lower Cholesky factor of an
unconstrained `d`×`d` covariance matrix.

The expected input is a `Matrix` with T(`d`) = `d`(`d`+1)÷2 rows, where T(`d`)
is the `d`th triangular number (the number of free parameters in an
unconstrained `d`×`d` covariance matrix), and the output is a `Matrix` of the
same dimension. The columns of the input and output matrices correspond to
independent parameter configurations (i.e., different covariance matrices).

Internally, the layer constructs a valid Cholesky factor 𝐋 and then extracts
the lower triangle from the positive-definite covariance matrix 𝚺 = 𝐋𝐋'. The
lower triangle is extracted and vectorised in line with Julia's column-major
ordering: for example, when modelling the covariance matrix

```math
\begin{bmatrix}
Σ₁₁ & Σ₁₂ & Σ₁₃ \\
Σ₂₁ & Σ₂₂ & Σ₂₃ \\
Σ₃₁ & Σ₃₂ & Σ₃₃ \\
\end{bmatrix},
```

the rows of the matrix returned by a `CovarianceMatrix` are ordered as

```math
\begin{bmatrix}
Σ₁₁ \\
Σ₂₁ \\
Σ₃₁ \\
Σ₂₂ \\
Σ₃₂ \\
Σ₃₃ \\
\end{bmatrix},
```

which means that the output can easily be transformed into the implied
covariance matrices using [`vectotril`](@ref) and `Symmetric`.

See also [`CorrelationMatrix`](@ref).

# Examples
```
using NeuralEstimators
using Flux
using LinearAlgebra

d = 4
l = CovarianceMatrix(d)
p = d*(d+1)÷2
θ = randn(p, 50)

# Returns a matrix of parameters, which can be converted to covariance matrices
Σ = l(θ)
Σ = [Symmetric(cpu(vectotril(x)), :L) for x ∈ eachcol(Σ)]

# Obtain the Cholesky factor directly
L = l(θ, true)
L = [LowerTriangular(cpu(vectotril(x))) for x ∈ eachcol(L)]
L[1] * L[1]'
```
"""
struct CovarianceMatrix{T <: Integer, G, H}
  d::T          # dimension of the matrix
  p::T          # number of free parameters in the covariance matrix, the triangular number T(d) = `d`(`d`+1)÷2
  tril_idx::G   # cartesian indices of lower triangle
  diag_idx::H   # which of the T(d) rows correspond to the diagonal elements of the `d`×`d` covariance matrix (linear indices)
end
function CovarianceMatrix(d::Integer)
	p = triangularnumber(d)
	tril_idx = tril(trues(d, d))
	diag_idx = [1]
	for i ∈ 1:(d-1)
		push!(diag_idx, diag_idx[i] + d-i+1)
	end
	return CovarianceMatrix(d, p, tril_idx, diag_idx)
end
function (l::CovarianceMatrix)(v, cholesky_only::Bool = false)

	d = l.d
	p, K = size(v)
	@assert p == l.p "the number of rows must be the triangular number T(d) = d(d+1)÷2 = $(l.p)"

	# Ensure that diagonal elements are positive
	#TODO the solution might be to replace the comprehension with map(): see https://github.com/FluxML/Flux.jl/issues/2187
	L = vcat([i ∈ l.diag_idx ? softplus.(v[i:i, :]) : v[i:i, :] for i ∈ 1:p]...)
	cholesky_only && return L

	# Insert zeros so that the input v can be transformed into Cholesky factors
	zero_mat = zero(L[1:d, :]) # NB Zygote does not like repeat()
	x = d:-1:1      # number of rows to extract from v
	j = cumsum(x)   # end points of the row-groups of v
	k = j .- x .+ 1 # start point of the row-groups of v
	L = vcat(L[k[1]:j[1], :], [vcat(zero_mat[1:i.-1, :], L[k[i]:j[i], :]) for i ∈ 2:d]...)

	# Reshape to a three-dimensional array of Cholesky factors
	L = reshape(L, d, d, K)

	# Batched multiplication and transpose to compute covariance matrices
	Σ = L ⊠ batched_transpose(L) # alternatively: PermutedDimsArray(L, (2,1,3)) or permutedims(L, (2, 1, 3))

	# Extract the lower triangle of each matrix
	Σ = Σ[l.tril_idx, :]

	return Σ
end
(l::CovarianceMatrix)(v::AbstractVector) = l(reshape(v, :, 1))

@doc raw"""
    CorrelationMatrix(d)
	(object::CorrelationMatrix)(x::Matrix, cholesky::Bool = false)
Transforms a vector 𝐯 ∈ ℝᵈ to the parameters of an unconstrained `d`×`d`
correlation matrix or, if `cholesky = true`, the lower Cholesky factor of an
unconstrained `d`×`d` correlation matrix.

The expected input is a `Matrix` with T(`d`-1) = (`d`-1)`d`÷2 rows, where T(`d`-1)
is the (`d`-1)th triangular number (the number of free parameters in an
unconstrained `d`×`d` correlation matrix), and the output is a `Matrix` of the
same dimension. The columns of the input and output matrices correspond to
independent parameter configurations (i.e., different correlation matrices).

Internally, the layer constructs a valid Cholesky factor 𝐋 for a correlation
matrix, and then extracts the strict lower triangle from the correlation matrix
𝐑 = 𝐋𝐋'. The lower triangle is extracted and vectorised in line with Julia's
column-major ordering: for example, when modelling the correlation matrix

```math
\begin{bmatrix}
1   & R₁₂ &  R₁₃ \\
R₂₁ & 1   &  R₂₃\\
R₃₁ & R₃₂ & 1\\
\end{bmatrix},
```

the rows of the matrix returned by a `CorrelationMatrix` layer are ordered as

```math
\begin{bmatrix}
R₂₁ \\
R₃₁ \\
R₃₂ \\
\end{bmatrix},
```

which means that the output can easily be transformed into the implied
correlation matrices using [`vectotril`](@ref) and `Symmetric`.

See also [`CovarianceMatrix`](@ref).

# Examples
```
using NeuralEstimators
using LinearAlgebra
using Flux

d  = 4
l  = CorrelationMatrix(d)
p  = (d-1)*d÷2
θ  = randn(p, 100)

# Returns a matrix of parameters, which can be converted to correlation matrices
R = l(θ)
R = map(eachcol(R)) do r
	R = Symmetric(cpu(vectotril(r, strict = true)), :L)
	R[diagind(R)] .= 1
	R
end

# Obtain the Cholesky factor directly
L = l(θ, true)
L = map(eachcol(L)) do x
	# Only the strict lower diagonal elements are returned
	L = LowerTriangular(cpu(vectotril(x, strict = true)))

	# Diagonal elements are determined under the constraint diag(L*L') = 𝟏
	L[diagind(L)] .= sqrt.(1 .- rowwisenorm(L).^2)
	L
end
L[1] * L[1]'
```
"""
struct CorrelationMatrix{T <: Integer, G}
  d::T                # dimension of the matrix
  p::T                # number of free parameters in the correlation matrix, the triangular number T(d-1) = (`d`-1)`d`÷2
  tril_idx_strict::G  # cartesian indices of strict lower triangle
end
function CorrelationMatrix(d::Integer)
	tril_idx_strict = tril(trues(d, d), -1)
	p = triangularnumber(d-1)
	return CorrelationMatrix(d, p, tril_idx_strict)
end
function (l::CorrelationMatrix)(v, cholesky_only::Bool = false)

	d = l.d
	p, K = size(v)
	@assert p == l.p "the number of rows must be the triangular number T(d-1) = (d-1)d÷2 = $(l.p)"

	# Insert zeros so that the input v can be transformed into Cholesky factors
	zero_mat = zero(v[1:d, :]) # NB Zygote does not like repeat()
	x = (d-1):-1:0           # number of rows to extract from v
	j = cumsum(x[1:end-1])   # end points of the row-groups of v
	k = j .- x[1:end-1] .+ 1 # start points of the row-groups of v
	L = vcat([vcat(zero_mat[1:i, :], v[k[i]:j[i], :]) for i ∈ 1:d-1]...)
	L = vcat(L, zero_mat)

	# Reshape to a three-dimensional array of Cholesky factors
	L = reshape(L, d, d, K)

	# Unit diagonal
	one_matrix = one(L[:, :, 1])
	L = L .+ one_matrix

	# Normalise the rows
	L = L ./ rowwisenorm(L)

	cholesky_only && return L[l.tril_idx_strict, :]

	# Transpose and batched multiplication to compute correlation matrices
	R = L ⊠ batched_transpose(L) # alternatively: PermutedDimsArray(L, (2,1,3)) or permutedims(L, (2, 1, 3))

	# Extract the lower triangle of each matrix
	R = R[l.tril_idx_strict, :]

  return R
end
(l::CorrelationMatrix)(v::AbstractVector) = l(reshape(v, :, 1))


# # Example input data helpful for prototyping:
# d = 4
# K = 100
# triangularnumber(d) = d*(d+1)÷2
#
# p = triangularnumber(d-1)
# v = collect(range(1, p*K))
# v = reshape(v, p, K)
# l = CorrelationMatrix(d)
# l(v) - l(v, true) # note that the first columns of a correlation matrix and its Cholesky factor will always be identical
#
# using LinearAlgebra
# R = rand(d, d); R = R * R'
# D = Diagonal(1 ./ sqrt.(R[diagind(R)]))
# R = Symmetric(D * R *D)
# L = cholesky(R).L
# LowerTriangular(R) - L
#
# p = triangularnumber(d)
# v = collect(range(1, p*K))
# v = reshape(v, p, K)
# l = CovarianceMatrix(d)
# l(v) - l(v, true)


# ---- Layers ----

"""
	DensePositive(layer::Dense, g::Function)
	DensePositive(layer::Dense; g::Function = Flux.relu)
Wrapper around the standard
[Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) layer that
ensures positive weights (biases are left unconstrained).

This layer can be useful for constucting (partially) monotonic neural networks (see, e.g., [`QuantileEstimatorContinuous`])(@ref).

# Examples
```
using NeuralEstimators, Flux

layer = DensePositive(Dense(5 => 2))
x = rand32(5, 64)
layer(x)
```
"""
struct DensePositive
	layer::Dense
	g::Function
	last_only::Bool
end
DensePositive(layer::Dense; g::Function = Flux.relu, last_only::Bool = false) = DensePositive(layer, g, last_only)
@layer DensePositive
# Simple version of forward pass:
# (d::DensePositive)(x) = d.layer.σ.(Flux.softplus(d.layer.weight) * x .+ d.layer.bias)
# Complex version of forward pass based on Flux's Dense code:
function (d::DensePositive)(x::AbstractVecOrMat)
  a = d.layer # extract the underlying fully-connected layer
  _size_check(a, x, 1 => size(a.weight, 2))
  σ = NNlib.fast_act(a.σ, x) # replaces tanh => tanh_fast, etc
  xT = _match_eltype(a, x)   # fixes Float64 input, etc.
  if d.last_only
	  weight = d.g.(hcat(a.weight[:, 1:end-1], a.weight[:, end:end]))
  else
	  weight = d.g.(a.weight)
  end
  σ.(weight * xT .+ a.bias)
end
function (a::DensePositive)(x::AbstractArray)
  a = d.layer # extract the underlying fully-connected layer
  _size_check(a, x, 1 => size(a.weight, 2))
  reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end
