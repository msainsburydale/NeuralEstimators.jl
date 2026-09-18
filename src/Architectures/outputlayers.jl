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
```julia
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
    function Compress(a::T, b::T, k::T) where {T}
        @assert all(b .> a) "All upper bounds b must be strictly greater than lower bounds a"
        new{T}(a, b, k)
    end
end
Compress(a, b) = Compress(float.(a), float.(b), ones(eltype(float.(a)), length(a)))
Compress(a::Number, b::Number) = Compress([float(a)], [float(b)])
(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-l.k .* θ))
Optimisers.trainable(l::Compress) = NamedTuple()

triangularnumber(d) = d*(d+1)÷2

@doc raw"""
    CovarianceMatrix(d)
	(object::CovarianceMatrix)(x::Matrix, cholesky::Bool = false)
Transforms unconstrained input into the parameters of a `d`×`d`
covariance matrix or, if `cholesky = true`, the lower Cholesky factor of a `d`×`d` covariance matrix.

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
```julia
using NeuralEstimators, LinearAlgebra

d = 4
l = CovarianceMatrix(d)
p = d*(d+1)÷2
x = randn(p, 50)

# Returns a matrix of parameters, which can be converted to covariance matrices
Σ = l(x)
Σ = [Symmetric(vectotril(x), :L) for x ∈ eachcol(Σ)]

# Obtain the Cholesky factor directly
L = l(x, true)
L = [LowerTriangular(vectotril(x)) for x ∈ eachcol(L)]
L[1] * L[1]'
```
"""
struct CovarianceMatrix{T1, T2, I <: Integer}
    d::I          # dimension of the matrix
    p::I          # number of free parameters in the covariance matrix, the triangular number d(d+1)÷2
    tril_idx::T1   # cartesian indices of lower triangle
    diag_idx::T2   # rows corresponding to the diagonal elements of the d×d covariance matrix   
end
function CovarianceMatrix(d::Integer)
    tril_idx = tril(trues(d, d))
    diag_idx = [1]
    for i ∈ 2:d
        push!(diag_idx, diag_idx[i - 1] + d-(i-1)+1)
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
    L̃ = vcat(L[k[1]:j[1], :], [vcat(zero_mat[1:(i .- 1), :], L[k[i]:j[i], :]) for i ∈ 2:d]...)

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
Transforms unconstrained input into the parameters of a `d`×`d`
correlation matrix or, if `cholesky = true`, the lower Cholesky factor of a 
`d`×`d` correlation matrix.

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
```julia
using NeuralEstimators, LinearAlgebra

d  = 4
l  = CorrelationMatrix(d)
p  = (d-1)*d÷2
x  = randn(p, 100)

# Returns a matrix of parameters, which can be converted to correlation matrices
R = l(x)
R = map(eachcol(R)) do r
	R = Symmetric(vectotril(r, strict = true), :L)
	R[diagind(R)] .= 1
	R
end

# Obtain the Cholesky factor directly
L = l(x, true)
L = map(eachcol(L)) do x
	# Only the strict lower diagonal elements are returned
	L = LowerTriangular(vectotril(x, strict = true))

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
    x = (d - 1):-1:0           # number of rows to extract from v
    j = cumsum(x[1:(end - 1)])   # end points of the row-groups of v
    k = j .- x[1:(end - 1)] .+ 1 # start points of the row-groups of v
    L = vcat([vcat(zero_mat[1:i, :], v[k[i]:j[i], :]) for i ∈ 1:(d - 1)]...)
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