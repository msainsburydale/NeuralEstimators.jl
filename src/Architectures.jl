
# ---- DeepSet ----

#TODO delete some of the methods (redundant now that we've changed the fields of RatioEstimator)
#TODO Remove ElementwiseAggregator?

@concrete struct ElementwiseAggregator
    a::Any
end
(e::ElementwiseAggregator)(x::A) where {A <: AbstractArray{T, N}} where {T, N} = e.a(x, dims = N)

"""
	(S::AbstractVector)(z)
	(S::Tuple)(z)
Method allows a collection of vector-valued functions to be applied to a single
input `z` and then concatenated, which allows users to provide a collection of
functions as a user-defined summary statistic in [`DeepSet`](@ref) objects.

Examples
```julia
f(z) = rand32(2)
g(z) = rand32(3) .+ z
S = [f, g]
S(1)
```
"""
(S::AbstractVector)(z) = reduce(vcat, [s(z) for s in S])
(S::Tuple)(z) = reduce(vcat, [s(z) for s in S])

@doc raw"""
    DeepSet(ψ, ϕ, a = mean; S = nothing)
	(ds::DeepSet)(Z::Vector{A}) where A <: Any
	(ds::DeepSet)(tuple::Tuple{Vector{A}, Vector{Vector}}) where A <: Any
The DeepSets representation ([Zaheer et al., 2017](https://arxiv.org/abs/1703.06114); [Sainsbury-Dale et al., 2024](https://www.tandfonline.com/doi/full/10.1080/00031305.2023.2249522)),
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}(\mathbf{T}(\mathbf{Z})), \quad
\mathbf{T}(\mathbf{Z}) = \mathbf{a}(\{\boldsymbol{\psi}(\mathbf{Z}_i) : i = 1, \dots, m\}),
```
where 𝐙 ≡ (𝐙₁', …, 𝐙ₘ')' are exchangeable replicates of data, 
`ψ` and `ϕ` are neural networks, and `a` is a permutation-invariant aggregation
function. 

The function `a` must operate on arrays and have a keyword argument `dims` for 
specifying the dimension of aggregation (e.g., `mean`, `sum`, `maximum`, `minimum`, `logsumexp`).

`DeepSet` objects act on data of type `Vector{A}`, where each
element of the vector is associated with one data set (i.e., one set of exchangeable replicates), and where `A` depends on the chosen architecture for `ψ`. 
Exchangeable replicates within each data set are stored in the batch dimension. For example, with data collected over a two-dimensional grid and with `ψ` chosen to be a CNN, `A` should be a 4-dimensional array, 
with replicates stored in the 4ᵗʰ dimension. 

For computational efficiency, 
array data are first concatenated along their final dimension 
(i.e., the replicates dimension) before being passed into the inner network `ψ`, 
thereby ensuring that `ψ` is applied to a single large array, rather than multiple small ones. 

Fixed (non-trainable) transformations of the data can be incorporated alongside the learned summaries via the `S` argument:
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})')'),
```
where `S` is a function (or vector of functions) that maps data to a vector of fixed summary statistics. These are not differentiated through during training. In the case that `ψ` is set to `nothing`, only the fixed summaries will be used. For the common case where summary statistics are precomputed and stored alongside the data, see [`DataSet`](@ref) as an alternative approach.

Set-level inputs (e.g., covariates) ``𝐗`` can be passed
directly into the outer network `ϕ` in the following manner: 
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{X}')'),
```
or, when fixed transformations are also used,
```math
\hat{\boldsymbol{\theta}}(\mathbf{Z}) = \boldsymbol{\phi}((\mathbf{T}(\mathbf{Z})', \mathbf{S}(\mathbf{Z})', \mathbf{X}')').
```
This is done by calling the `DeepSet` object on a
`Tuple{Vector{A}, Vector{Vector}}`, where the first element of the tuple
contains a vector of data sets and the second element contains a vector of
set-level inputs (i.e., one vector for each data set).

!!! note
    `DeepSet` is currently only implemented for the `Flux` backend.

# Examples
```julia
using NeuralEstimators, Flux

# Two data sets containing 3 and 4 replicates
d = 5  # number of parameters in the model
n = 10 # dimension of each replicate
Z = [rand32(n, m) for m ∈ (3, 4)]

# Construct DeepSet object
dₜ = 16  # dimension of neural summary statistic
w  = 32  # width of hidden layers
ψ  = Chain(Dense(n, w, relu), Dense(w, dₜ, relu))
ϕ  = Chain(Dense(dₜ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ)

# Apply DeepSet object to data
ds(Z)

# With fixed transformations S
dₛ = 1   # dimension of fixed summary statistic
ϕ  = Chain(Dense(dₜ + dₛ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ; S = logsamplesize)
ds(Z)

# With set-level inputs 
dₓ = 2 # dimension of set-level inputs 
ϕ  = Chain(Dense(dₜ + dₓ, w, relu), Dense(w, d))
ds = DeepSet(ψ, ϕ)
X  = [rand32(dₓ) for _ ∈ eachindex(Z)]
ds((Z, X))
```
"""
@concrete struct DeepSet
    ψ::Any
    ϕ::Any
    a::Any
    S::Any
end
function DeepSet(ψ, ϕ, a::Function = mean; S = nothing)
    @assert !isnothing(ψ) | !isnothing(S) "At least one of `ψ` or `S` must be given"
    DeepSet(ψ, ϕ, ElementwiseAggregator(a), S)
end
Base.show(io::IO, D::DeepSet) = print(io, "\nDeepSet object with:\nInner network:  $(D.ψ)\nAggregation function:  $(D.a)\nExpert statistics: $(D.S)\nOuter network:  $(D.ϕ)")

# Single data set
function (d::DeepSet)(Z::A) where {A}
    d.ϕ(_deepsetsummaries(d, Z))
end
# Single data set with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractVector{T}} where {T}
    Z, x = tup
    t = _deepsetsummaries(d, Z)
    u = vcat(t, x)
    d.ϕ(u)
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{A, B}} where {A, B <: AbstractMatrix{T}} where {T}
    Z, x = tup
    if size(x, 2) == 1
        # Catches the simple case that the user accidentally passed an Nx1 matrix rather than an N-dimensional vector. 
        d((Z, vec(x)))
    else
        # Designed for situations where we have a fixed data set and want to
        # evaluate the object for many different set-level covariates
        t = _deepsetsummaries(d, Z) # only needs to be computed once
        tx = vcat(repeat(t, 1, size(x, 2)), x) # NB ideally we'd avoid copying t so many times here, using @view
        d.ϕ(tx)
    end
end
# Multiple data sets
function (d::DeepSet)(Z::V) where {V <: AbstractVector{A}} where {A}
    # Stack into a single array before applying the outer network
    d.ϕ(stackarrays(_deepsetsummaries(d, Z)))
end
# Multiple data sets with set-level covariates
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <: AbstractVector{T}} where {T}
    Z, x = tup
    t = _deepsetsummaries(d, Z)
    tx = vcat.(t, x)
    d.ϕ(stackarrays(tx))
end
function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V, M}} where {V <: AbstractVector{A}, M <: AbstractMatrix{T}} where {A, T}
    Z, x = tup
    if size(x, 2) == length(Z)
        # Catches the simple case that the user accidentally passed an NxM matrix rather than an M-dimensional vector of N-vector.
        d((Z, eachcol(x)))
    else
        # Designed for situations where we have a several data sets and we want
        # to evaluate the object for many different set-level covariates
        [d((z, x)) for z in Z]
    end
end

# Single data set
function _deepsetsummaries(d::DeepSet, Z::A) where {A}
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
function _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where {A}
    _deepsetsummaries.(Ref(d), Z)
end

# Multiple data sets: optimised version for array data
function _deepsetsummaries(d::DeepSet, Z::V) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}
    if !isnothing(d.ψ)
        if _first_N_minus_1_dims_identical(Z)
            # Stack Z = [A₁, A₂, ...] into a single large N-dimensional array and then apply the inner network
            ψa = d.ψ(stackarrays(Z))

            # Compute the indices needed for aggregation (i.e., the indicies associated with each Aᵢ in the stacked array)
            mᵢ = size.(Z, N) # number of replicates for every element in Z
            cs = cumsum(mᵢ)
            indices = [(cs[i] - mᵢ[i] + 1):cs[i] for i ∈ eachindex(Z)]

            # Construct the summary statistics
            t = map(indices) do idx
                d.a(getobs(ψa, idx))
            end

            if !isnothing(d.S)
                s = @ignore_derivatives d.S.(Z)
                if !isnothing(d.ψ)
                    t = vcat.(t, s)
                else
                    t = s
                end
            end

            return t
        else
            # Array sizes differ, so therefore cannot stack together; use simple (and slower) broadcasting method (identical to general fallback method defined above)
            return _deepsetsummaries.(Ref(d), Z)
        end
    end
end

function _first_N_minus_1_dims_identical(arrays::Vector{<:AbstractArray})
    # Get the size of the first array up to N-1 dimensions
    first_size = size(arrays[1])[1:(end - 1)]

    # Loop over the remaining arrays and compare their first N-1 dimensions
    for i = 2:length(arrays)
        if size(arrays[i])[1:(end - 1)] != first_size
            return false  # Dimensions do not match
        end
    end

    return true  # All arrays have the same first N-1 dimensions
end

# ---- Output layers ----

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

# ---- Layers ----

"""
    MLP(in::Integer, out::Integer; kwargs...)

A traditional fully-connected multilayer perceptron (MLP) with input dimension `in` and output dimension `out`.

# Keyword arguments
- `depth::Integer = 2`: the number of hidden layers. Use `depth = 0` for a single linear layer with no hidden layers.
- `width::Integer = 128`: the width of each hidden layer.
- `activation = relu`: the activation function used in each hidden layer.
- `output_activation = identity`: the activation function used in the output layer.
- `backend::Union{Nothing, Module} = nothing`: the backend to use for constructing the network (e.g., `Lux` or `Flux`). If `nothing`, the backend is resolved automatically.
"""
function MLP(in::Integer, out::Integer; depth::Integer = 2, width::Integer = 128, activation = relu, output_activation = identity, backend::Union{Nothing, Module} = nothing)
    @assert depth >= 0
    B = _resolvebackend(backend)
    if depth == 0
        layers = Any[B.Dense(in => out, output_activation)]
    else
        layers = []
        push!(layers, B.Dense(in => width, activation))
        append!(layers, [B.Dense(width => width, activation) for _ ∈ 2:depth])
        push!(layers, B.Dense(width => out, output_activation))
    end

    return B.Chain(layers...)
end

"""
    ResidualBlock(filter, in => out; stride = 1, backend = nothing)

Basic residual block (see [here](https://en.wikipedia.org/wiki/Residual_neural_network#Basic_block)),
consisting of two sequential convolutional layers and a skip (shortcut) connection
that connects the input of the block directly to the output,
facilitating the training of deep networks.

# Examples
```julia
using NeuralEstimators, Flux
z = rand(16, 16, 1, 1)
b = ResidualBlock((3, 3), 1 => 32)
b(z)
```
"""
function ResidualBlock(filter, channels; stride = 1, backend::Union{Nothing, Module} = nothing)
    B = _resolvebackend(backend)
    lux = get(Base.loaded_modules, _LUX_UUID, nothing)
    is_lux = !isnothing(lux) && B === lux

    bias_kwarg = is_lux ? :use_bias : :bias
    id = is_lux ? lux.WrappedFunction(identity) : identity

    layer = B.Chain(
        B.Conv(filter, channels; stride = stride, pad = 1, bias_kwarg => false),
        B.BatchNorm(channels[2], relu),
        B.Conv(filter, channels[2] => channels[2]; pad = 1, bias_kwarg => false),
        B.BatchNorm(channels[2])
    )

    connection = if stride == 1 && channels[1] == channels[2]
        +
    else
        projection = B.Chain(
            B.Conv((1, 1), channels; stride = stride, bias_kwarg => false),
            B.BatchNorm(channels[2])
        )
        B.Parallel(+, id, projection)
    end

    return B.Chain(B.SkipConnection(layer, connection), B.relu)
end

# ---- Structs for GNNs: Only compatible with Flux ----

"""
	PowerDifference(a, b)
Function ``f(x, y) = |\\tilde{a}x - (1-\\tilde{a})y|^{\\tilde{b}}``, where
``\\tilde{a} = \\text{sigmoid}(a) \\in (0, 1)`` and ``\\tilde{b} = \\text{softplus}(b) > 0``
are constrained transformations of the trainable parameters `a` and `b`.

# Examples
```julia
using NeuralEstimators

X = rand(5, 100)
Y = rand(5, 100)
f = PowerDifference(0, 1.55)
f(X, Y)   # two arg method
f((X, Y)) # tuple method
```
"""
struct PowerDifference{A, B}
    a::A
    b::B
end
PowerDifference() = PowerDifference([0.0f0], [1.55f0]) # default initial values chosen such that ã = 0.5 and b̃ ≈ 2
PowerDifference(a::Number, b::AbstractArray) = PowerDifference([a], b)
PowerDifference(a::AbstractArray, b::Number) = PowerDifference(a, [b])
(f::PowerDifference)(x, y) = (abs.(sigmoid.(f.a) .* x - (1 .- sigmoid.(f.a)) .* y)) .^ softplus.(f.b)
(f::PowerDifference)(tup::Tuple) = f(tup[1], tup[2])

@doc raw"""
	IndicatorWeights(h_max, n_bins::Integer)
	(w::IndicatorWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\mathbb{I}(h \in B_k) : k = 1, \dots, K)',
```

where $\mathbb{I}(\cdot)$ denotes the indicator function, 
$h \equiv \|\boldsymbol{s} - \boldsymbol{u} \|$ is the spatial distance between $\boldsymbol{s}$ and 
$\boldsymbol{u}$, and $\{B_k : k = 1, \dots, K\}$ is a set of $K =$`n_bins` equally-sized distance bins covering the spatial distances between 0 and `h_max`. 

# Examples 
```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = IndicatorWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct IndicatorWeights{T}
    h_cutoffs::T
end
function IndicatorWeights(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    IndicatorWeights(h_cutoffs)
end
function (l::IndicatorWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    h_cutoffs = l.h_cutoffs
    bins_upper = h_cutoffs[2:end]   # upper bounds of the distance bins
    bins_lower = h_cutoffs[1:(end - 1)] # lower bounds of the distance bins 
    N = [bins_lower[i:i] .< h .<= bins_upper[i:i] for i in eachindex(bins_upper)] # NB avoid scalar indexing by i:i
    N = reduce(vcat, N)
    f32(N)
end
Optimisers.trainable(l::IndicatorWeights) = NamedTuple()

@doc raw"""
	KernelWeights(h_max, n_bins::Integer)
	(w::KernelWeights)(h::Matrix) 
For spatial locations $\boldsymbol{s}$ and  $\boldsymbol{u}$, creates a spatial weight function defined as

```math 
\boldsymbol{w}(\boldsymbol{s}, \boldsymbol{u}) \equiv (\exp(-(h - \mu_k)^2 / (2\sigma_k^2)) : k = 1, \dots, K)',
```

where $h \equiv \|\boldsymbol{s} - \boldsymbol{u}\|$ is the spatial distance between $\boldsymbol{s}$ and $\boldsymbol{u}$, and ${\mu_k : k = 1, \dots, K}$ and ${\sigma_k : k = 1, \dots, K}$ are the means and standard deviations of the Gaussian kernels for each bin, covering the spatial distances between 0 and h_max.

# Examples 
```julia
using NeuralEstimators, GraphNeuralNetworks

h_max = 1
n_bins = 10
w = KernelWeights(h_max, n_bins)
h = rand(1, 30) # distances between 30 pairs of spatial locations 
w(h)
```
"""
struct KernelWeights{T1, T2}
    mu::T1
    sigma::T2
end
function KernelWeights(h_max, n_bins::Integer)
    h_cutoffs = range(0, stop = h_max, length = n_bins+1)
    h_cutoffs = collect(h_cutoffs)
    mu = [(h_cutoffs[i] + h_cutoffs[i + 1]) / 2 for i = 1:n_bins] # midpoints of the intervals 
    sigma = [(h_cutoffs[i + 1] - h_cutoffs[i]) / 4 for i = 1:n_bins] # std dev so that 95% of mass is within the bin 
    mu = f32(mu)
    sigma = f32(sigma)
    KernelWeights(mu, sigma)
end
function (l::KernelWeights)(h::M) where {M <: AbstractMatrix{T}} where {T}
    mu = l.mu
    sigma = l.sigma
    N = [exp.(-(h .- mu[i:i]) .^ 2 ./ (2 * sigma[i:i] .^ 2)) for i in eachindex(mu)] # Gaussian kernel for each bin (NB avoid scalar indexing by i:i)
    N = reduce(vcat, N)
    f32(N)
end
Optimisers.trainable(l::KernelWeights) = NamedTuple()
