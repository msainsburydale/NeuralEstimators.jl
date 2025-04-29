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

# NB ideally wouldn't use this, but for backwards compatability I can't remove it now
struct ElementwiseAggregator
	a::Function
end
(e::ElementwiseAggregator)(x::A) where {A <: AbstractArray{T, N}} where {T, N} = e.a(x, dims = N)


@doc raw"""
    DeepSet(ψ, ϕ, a = mean; S = nothing)
	(ds::DeepSet)(Z::Vector{A}) where A <: Any
	(ds::DeepSet)(tuple::Tuple{Vector{A}, Vector{Vector}}) where A <: Any
The DeepSets representation ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114); [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)),
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}(\mathbf{T}(\mathbf{Z})), \quad
\mathbf{T}(\mathbf{Z}) = \mathbf{a}(\{\boldsymbol{\psi}(\mathbf{Z}_i) : i = 1, \dots, m\}),

```
where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are independent replicates of data, 
`ψ` and `ϕ` are neural networks, and `a` is a permutation-invariant aggregation
function. 

The function `a` must operate on arrays and have a keyword argument `dims` for 
specifying the dimension of aggregation (e.g., `sum`, `mean`, `maximum`, `minimum`, `logsumexp`).

`DeepSet` objects act on data of type `Vector{A}`, where each
element of the vector is associated with one data set (i.e., one set of
independent replicates), and where `A` depends on the chosen architecture for `ψ`. 
Independent replicates within each data set are stored in the batch dimension. 
For example, with gridded spatial data and `ψ` a CNN, `A` should be a 4-dimensional array, 
with replicates stored in the 4ᵗʰ dimension. 

For computational efficiency, 
array data are first concatenated along their final dimension 
(i.e., the replicates dimension) before being passed into the inner network `ψ`, 
thereby ensuring that `ψ` is applied to a single large array, rather than multiple small ones. 
	
Expert summary statistics can be incorporated as

```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})')'),
```
where `S` is a function that returns a vector of user-defined summary statistics.
These user-defined summary statistics are provided either as a
`Function` that returns a `Vector`, or as a vector of functions. In the case that
`ψ` is set to `nothing`, only expert summary statistics will be used. See [Expert summary statistics](@ref) for further discussion on their use. 

Set-level inputs (e.g., covariates) ``𝐗`` can be passed
directly into the outer network `ϕ` in the following manner: 
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{X}')'),
```
or, when expert summary statistics are also used,
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})', \mathbf{X}')').
```
This is done by calling the `DeepSet` object on a
`Tuple{Vector{A}, Vector{Vector}}`, where the first element of the tuple
contains a vector of data sets and the second element contains a vector of
set-level inputs (i.e., one vector for each data set).

# Examples
```
using NeuralEstimators, Flux

# Two data sets containing 3 and 4 replicates
d = 5  # number of parameters in the model
n = 10 # dimension of each replicate
Z = [rand32(n, m) for m ∈ (3, 4)]

# Construct DeepSet object
S = samplesize
dₛ = 1   # dimension of expert summary statistic
dₜ = 16  # dimension of neural summary statistic
w  = 32  # width of hidden layers
ψ  = Chain(Dense(n, w, relu), Dense(w, dₜ, relu))
ϕ  = Chain(Dense(dₜ + dₛ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ; S = S)

# Apply DeepSet object to data
ds(Z)

# With set-level inputs 
dₓ = 2 # dimension of set-level inputs 
ϕ  = Chain(Dense(dₜ + dₛ + dₓ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ; S = S)
X  = [rand32(dₓ) for _ ∈ eachindex(Z)]
ds((Z, X))
```
"""
struct DeepSet{T, G, K, A}
	ψ::T
	ϕ::G
	a::A
	S::K
end
function DeepSet(ψ, ϕ, a::Function = mean; S = nothing)
	@assert !isnothing(ψ) | !isnothing(S) "At least one of `ψ` or `S` must be given"
	DeepSet(ψ, ϕ, ElementwiseAggregator(a), S)
end
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")


# Single data set
function (d::DeepSet)(Z::A) where A
	d.ϕ(summarystatistics(d, Z))
end
# Single data set with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where T
	Z, x = tup
	t = summarystatistics(d, Z)
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
		# evaluate the object for many different set-level covariates
		t = summarystatistics(d, Z) # only needs to be computed once
		tx = vcat(repeat(t, 1, size(x, 2)), x) # NB ideally we'd avoid copying t so many times here, using @view
		d.ϕ(tx) # Sanity check: stackarrays([d((Z, vec(x̃))) for x̃ in eachcol(x)])
	end
end
# Multiple data sets
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where A
	# Stack into a single array before applying the outer network
	d.ϕ(stackarrays(summarystatistics(d, Z)))
end
# Multiple data sets with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
	Z, x = tup
	t = summarystatistics(d, Z)
	tx = vcat.(t, x)
	d.ϕ(stackarrays(tx))
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V, M}} where {V <: AbstractVector{A}, M <: AbstractMatrix{T}} where {A, T}
	Z, x = tup
	if size(x, 2) == length(Z)
		# Catches the simple case that the user accidentally passed an NxM matrix
		# rather than an M-dimensional vector of N-vector.
		# Also used by RatioEstimator.
		d((Z, eachcol(x)))
	else
		# Designed for situations where we have a several data sets and we want
		# to evaluate the object for many different set-level covariates
		[d((z, x)) for z in Z]
	end
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{M}} where {M <: AbstractMatrix{T}} where {A, T}
	# Multiple data sets Z, each applied over multiple set-level covariates
	# (NB similar to above method, but the set-level covariates are allowed to be different for each data set)
	# (This is used during training by QuantileEstimatorContinuous, where each data set is allowed multiple and different probability levels)
	Z, X = tup
	@assert length(Z) == length(X)
	result = [d((Z[k], X[k])) for k ∈ eachindex(Z)]
	reduce(hcat, vec.(permutedims.(result)))
end

# Single data set
function summarystatistics(d::DeepSet, Z::A) where A
	if !isnothing(d.ψ)
		t = d.a(d.ψ(Z))
	end
	if !isnothing(d.S)
		s = @ignore_derivatives d.S(Z)
		if !isnothing(d.ψ)
			t = vcat(t, s)
		else
			t = s
		end
	end
	return t
end
# Multiple data sets: general fallback using broadcasting
function summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where A
  	summarystatistics.(Ref(d), Z)
end

# Multiple data sets: optimised version for array data
function summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
	if !isnothing(d.ψ)
		if _first_N_minus_1_dims_identical(Z)
			# Stack Z = [A₁, A₂, ...] into a single large N-dimensional array and then apply the inner network
			ψa = d.ψ(stackarrays(Z))

			# Compute the indices needed for aggregation (i.e., the indicies associated with each Aᵢ in the stacked array)
			mᵢ  = size.(Z, N) # number of replicates for every element in Z
			cs  = cumsum(mᵢ)
			indices = [(cs[i] - mᵢ[i] + 1):cs[i] for i ∈ eachindex(Z)]
			
			# Construct the summary statistics
			t = map(indices) do idx
				d.a(getobs(ψa, idx))
			end

			if !isnothing(d.S)
				s = @ignore_derivatives d.S.(Z) # NB any expert summary statistics S are applied to the original data sets directly (so, if Z[i] is a supergraph, all subgraphs are independent replicates from the same data set)
				if !isnothing(d.ψ)
					t = vcat.(t, s)
				else
					t = s
				end
			end
		
			return t
		else 
			# Array sizes differ, so therefore cannot stack together; use simple (and slower) broadcasting method (identical to general fallback method defined above)
			return summarystatistics.(Ref(d), Z)
		end
	end
end

# Multiple data sets: optimised version for graph data
function summarystatistics(d::DeepSet, Z::V) where {V <: AbstractVector{G}} where {G <: GNNGraph}

	@assert isnothing(d.ψ) || typeof(d.ψ) <: GNNSummary "For graph input data, the summary network ψ should be a `GNNSummary` object"

	if !isnothing(d.ψ)

		if @ignore_derivatives _first_N_minus_1_dims_identical(Z)

			# For efficiency, convert Z from a vector of (super)graphs into a single
			# supergraph before applying the neural network. Since each element of Z
			# may itself be a supergraph (where each subgraph corresponds to an
			# independent replicate), record the grouping of independent replicates
			# so that they can be combined again later in the function
			m = numberreplicates.(Z)

			# Propagation and readout
			g = @ignore_derivatives Flux.batch(Z) # NB batch() causes array mutation, so do not attempt to compute derivatives through this call
			R = d.ψ(g)

			# Split R based on the original vector of data sets Z
			if ndims(R) == 2
				
				# R is a matrix, with column dimension M = sum(m), and we split R
				# based on the original grouping specified by m
				# NB since this only works for identical m, there is some code redundancy here I believe
				ng = length(m)
				cs = cumsum(m)
				indices = [(cs[i] - m[i] + 1):cs[i] for i ∈ 1:ng]
				R̃ = [R[:, idx] for idx ∈ indices]
			elseif ndims(R) == 3
				R̃ = [R[:, :, i] for i ∈ 1:size(R, 3)]
			end
		else
			# Array sizes differ, so therefore cannot stack together; use simple (and slower) broadcasting method 
			R̃ = d.ψ.(Z)
		end

		# Now we have a vector of matrices, where each matrix corresponds to the
		# readout vectors R₁, …, Rₘ for a given data set. Now, aggregate these
		# readout vectors into a single summary statistic for each data set:
		t = d.a.(R̃)
	end

	if !isnothing(d.S)
		s = @ignore_derivatives d.S.(Z) # NB any expert summary statistics S are applied to the original data sets directly (so, if Z[i] is a supergraph, all subgraphs are independent replicates from the same data set)
		if !isnothing(d.ψ)
			t = vcat.(t, s)
		else
			t = s
		end
	end

	return t
end

function _first_N_minus_1_dims_identical(arrays::Vector{<:AbstractArray})
    # Get the size of the first array up to N-1 dimensions
    first_size = size(arrays[1])[1:end-1]
    
    # Loop over the remaining arrays and compare their first N-1 dimensions
    for i in 2:length(arrays)
        if size(arrays[i])[1:end-1] != first_size
            return false  # Dimensions do not match
        end
    end
    
    return true  # All arrays have the same first N-1 dimensions
end

function _first_N_minus_1_dims_identical(v::AbstractVector{<:GNNGraph})
    # For each graph, extract the node features as a vector
    vecs = [[x.ndata[k] for k in keys(x.ndata)] for x in v]

    # Assume all graphs have the same keys (same number of features)
    k = length(vecs[1])
    @assert all(length(vec) == k for vec in vecs)

    # Split vecs into k vectors, each collecting one feature across graphs
    arrays = [ [vec[i] for vec in vecs] for i in 1:k ]

    # Check that the dimensions match for each group of feature arrays
    return all(_first_N_minus_1_dims_identical.(arrays))
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
Flux.trainable(l::Compress) =  NamedTuple()


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
Flux.trainable(l::TruncateSupport) = NamedTuple()
tuncatesupport(θ, a, b) = min(max(θ, a), b)

# ---- Layers to construct Covariance and Correlation matrices ----

#TODO update notation: d -> n, p -> d
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
using NeuralEstimators, Flux, LinearAlgebra

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
struct CovarianceMatrix{T <: Integer}
  d::T       # dimension of the matrix
  p::T       # number of free parameters in the covariance matrix, the triangular number d(d+1)÷2
  tril_idx   # cartesian indices of lower triangle
  diag_idx   # rows corresponding to the diagonal elements of the d×d covariance matrix   
end
function CovarianceMatrix(d::Integer)
	tril_idx = tril(trues(d, d))
	diag_idx = [1]
	for i ∈ 2:d
		push!(diag_idx, diag_idx[i-1] + d-(i-1)+1)
	end
	return CovarianceMatrix(d, triangularnumber(d), tril_idx, diag_idx)
end
function (l::CovarianceMatrix)(v, cholesky_only::Bool = false)

	# Extract indices 
	diag_idx = cpu(l.diag_idx)
	tril_idx = l.tril_idx

	d = l.d
	p, K = size(v)
	@assert p == l.p "the number of rows must be the triangular number d(d+1)÷2 = $(l.p)"

	# Ensure that diagonal elements are positive
	L = vcat([i ∈ diag_idx ? softplus(v[i:i, :]) : v[i:i, :] for i ∈ 1:p]...) 
	cholesky_only && return L

	# Insert zeros so that the input v can be transformed into Cholesky factors
	zero_mat = zero(L[1:d, :]) # NB Zygote does not like repeat()
	x = d:-1:1      # number of rows to extract from v
	j = cumsum(x)   # end points of the row-groups of v
	k = j .- x .+ 1 # start point of the row-groups of v
	L̃ = vcat(L[k[1]:j[1], :], [vcat(zero_mat[1:i.-1, :], L[k[i]:j[i], :]) for i ∈ 2:d]...)

	# Reshape to a three-dimensional array of Cholesky factors
	L̃ = reshape(L̃, d, d, K)

	# Batched multiplication and transpose to compute covariance matrices
	Σ = L̃ ⊠ batched_transpose(L̃) # alternatively: PermutedDimsArray(L, (2,1,3)) or permutedims(L, (2, 1, 3))

	# Extract the lower triangle of each matrix
	return Σ[tril_idx, :]
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
using NeuralEstimators, LinearAlgebra, Flux

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
	return CorrelationMatrix(d, triangularnumber(d-1), tril_idx_strict)
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

#NB this is from Flux, but I copied it here because I got an error that it wasn't defined when submitting to CRAN (think it's a recent addition to Flux)
function _size_check(layer, x::AbstractArray, (d, n)::Pair)
  0 < d <= ndims(x) || throw(DimensionMismatch(string("layer ", layer,
    " expects ndims(input) >= ", d, ", but got ", summary(x))))
  size(x, d) == n || throw(DimensionMismatch(string("layer ", layer,
    lazy" expects size(input, $d) == $n, but got ", summary(x))))
end
@non_differentiable _size_check(::Any...)

#TODO document last_only 
#TODO g should be a positional argument in line with standard flux layers 
"""
	DensePositive(layer::Dense; g::Function = relu, last_only::Bool = false)
Wrapper around the standard
[Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) layer that
ensures positive weights (biases are left unconstrained).

This layer can be useful for constucting (partially) monotonic neural networks. 

# Examples
```
using NeuralEstimators, Flux

l = DensePositive(Dense(5 => 2))
x = rand32(5, 64)
l(x)
```
"""
struct DensePositive{L, G}
	layer::L
	g::G
	last_only::Bool
end
DensePositive(layer::Dense; g::Function = Flux.relu, last_only::Bool = false) = DensePositive(layer, g, last_only)
# Simple version of forward pass:
# (d::DensePositive)(x) = d.layer.σ.(Flux.softplus(d.layer.weight) * x .+ d.layer.bias)
# Complex version of forward pass based on Flux's Dense code:
function (d::DensePositive)(x::AbstractVecOrMat)
  a = d.layer # extract the underlying fully-connected layer
  _size_check(a, x, 1 => size(a.weight, 2))
  σ = NNlib.fast_act(a.σ, x) # replaces tanh => tanh_fast
  xT = _match_eltype(a, x)   # fixes Float64 input
  if d.last_only
	  weight = hcat(a.weight[:, 1:end-1], d.g.(a.weight[:, end:end]))
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


#TODO constrain a ∈ [0, 1] and b > 0
"""
	PowerDifference(a, b)
Function ``f(x, y) = |ax - (1-a)y|^b`` for trainable parameters a ∈ [0, 1] and b > 0.

# Examples
```
using NeuralEstimators, Flux

# Generate some data
d = 5
K = 10000
X = randn32(d, K)
Y = randn32(d, K)
XY = (X, Y)
a = 0.2f0
b = 1.3f0
Z = (abs.(a .* X - (1 .- a) .* Y)).^b

# Initialise layer
f = PowerDifference([0.5f0], [2.0f0])

# Optimise the layer
loader = Flux.DataLoader((XY, Z), batchsize=32, shuffle=false)
optim = Flux.setup(Flux.Adam(0.01), f)
for epoch in 1:100
    for (xy, z) in loader
        loss, grads = Flux.withgradient(f) do m
            Flux.mae(m(xy), z)
        end
        Flux.update!(optim, f, grads[1])
    end
end

# Estimates of a and b
f.a
f.b
```
"""
struct PowerDifference{A,B}
	a::A
	b::B
end
PowerDifference() = PowerDifference([0.5f0], [2.0f0])
PowerDifference(a::Number, b::AbstractArray) = PowerDifference([a], b)
PowerDifference(a::AbstractArray, b::Number) = PowerDifference(a, [b])
(f::PowerDifference)(x, y) = (abs.(f.a .* x - (1 .- f.a) .* y)).^f.b
(f::PowerDifference)(tup::Tuple) = f(tup[1], tup[2])


#TODO add further details
#TODO Groups in ResidualBlock (i.e., allow additional arguments to Conv).
"""
	ResidualBlock(filter, in => out; stride = 1)

Basic residual block (see [here](https://en.wikipedia.org/wiki/Residual_neural_network#Basic_block)),
consisting of two sequential convolutional layers and a skip (shortcut) connection
that connects the input of the block directly to the output,
facilitating the training of deep networks.

# Examples
```
using NeuralEstimators
z = rand(16, 16, 1, 1)
b = ResidualBlock((3, 3), 1 => 32)
b(z)
```
"""
struct ResidualBlock{B}
    block::B
end
(b::ResidualBlock)(x) = relu.(b.block(x))
function ResidualBlock(filter, channels; stride = 1)

    layer = Chain(
        Conv(filter, channels; stride = stride, pad=1, bias=false),
        BatchNorm(channels[2], relu),
        Conv(filter, channels[2]=>channels[2]; pad=1, bias=false),
        BatchNorm(channels[2])
        )

    if stride == 1 && channels[1] == channels[2]
        # dimensions match, can add input directly to output
        connection = +
    else
        #TODO options for different dimension matching (padding vs. projection)
        # Projection connection using 1x1 convolution
        connection = Shortcut(
                Chain(
                    Conv((1, 1), channels; stride = stride, bias=false),
                    BatchNorm(channels[2])
                )
            )
    end

    ResidualBlock(SkipConnection(layer, connection))
end
struct Shortcut{S}
    s::S
end
(s::Shortcut)(mx, x) = mx + s.s(x)



