"""
Generic function that may be overloaded to implicitly define a statistical model.
Specifically, the user should provide a method `simulate(parameters, m)`
that returns `m` simulated replicates for each element in the given set of
`parameters`.
"""
function simulate end

"""
	simulate(parameters, m, J::Integer)

Simulates `J` sets of `m` independent replicates for each parameter vector in
`parameters` by calling `simulate(parameters, m)` a total of `J` times,
where the method `simulate(parameters, m)` is provided by the user via function
overloading.

# Examples
```
import NeuralEstimators: simulate

p = 2
K = 10
m = 15
parameters = rand(p, K)

# Univariate Gaussian model with unknown mean and standard deviation
simulate(parameters, m) = [θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)]
simulate(parameters, m)
simulate(parameters, m, 2)
```
"""
function simulate(parameters::P, m, J::Integer; args...) where P <: Union{AbstractMatrix, ParameterConfigurations}
	v = [simulate(parameters, m; args...) for i ∈ 1:J]
	if typeof(v[1]) <: Tuple
		z = vcat([v[i][1] for i ∈ eachindex(v)]...)
		x = vcat([v[i][2] for i ∈ eachindex(v)]...)
		v = (z, x)
	else
		v = vcat(v...)
	end
	return v
end

# ---- Gaussian process ----

# returns the number of locations in the field
size(grf::GaussianRandomField) = prod(size(grf.mean))
size(grf::GaussianRandomField, d::Integer) = size(grf)

"""
	simulategaussianprocess(L::Matrix, m = 1)
	simulategaussianprocess(grf::GaussianRandomField, m = 1)

Simulates `m` independent and identically distributed (i.i.d.) realisations from
a mean-zero Gaussian process.

Accepts either the lower Cholesky factor `L` associated with a Gaussian process
or an object from the package [`GaussianRandomFields`](https://github.com/PieterjanRobbe/GaussianRandomFields.jl).

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are
returned as an ``n``x`m` matrix.

# Examples
```
using NeuralEstimators
using Distances
using LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)

# Passing a Cholesky factor:
D = pairwise(Euclidean(), S, S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulategaussianprocess(L)

# Passing a GaussianRandomField with Cholesky decomposition:
using GaussianRandomFields
cov = CovarianceFunction(2, Matern(ρ, ν))
grf = GaussianRandomField(cov, GaussianRandomFields.Cholesky(), S)
simulategaussianprocess(grf)

# Passing a GaussianRandomField with circulant embedding (fast but requires regular grid):
pts = range(0.0, 1.0, 20)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding = 100)
simulategaussianprocess(grf)
```
"""
function simulategaussianprocess(obj::M, m::Integer) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number
	y = [simulategaussianprocess(obj) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

# TODO This should really dispatch on LowerTriangular, Symmetric, and
# UpperTriangular. For backwards compatability, we can keep the following
# method for L::AbstractMatrix.
function simulategaussianprocess(L::M) where M <: AbstractMatrix{T} where T <: Number
	L * randn(T, size(L, 1))
end

function simulategaussianprocess(grf::GaussianRandomField)
	vec(GaussianRandomFields.sample(grf))
end


# TODO add simulateGH()

# ---- Schlather's max-stable model ----

"""
	simulateschlather(L::Matrix, m = 1)
	simulateschlather(grf::GaussianRandomField, m = 1)

Simulates `m` independent and identically distributed (i.i.d.) realisations from
Schlather's max-stable model using the algorithm for approximate simulation given
by [Schlather (2002)](https://link.springer.com/article/10.1023/A:1020977924878).

Accepts either the lower Cholesky factor `L` associated with a Gaussian process
or an object from the package [`GaussianRandomFields`](https://github.com/PieterjanRobbe/GaussianRandomFields.jl).

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are
returned as an ``n``x`m` matrix.

# Keyword arguments
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorithm: small `C` favours computational efficiency, while large `C` favours accuracy. Schlather (2002) recommends the use of `C = 3`.
- `Gumbel = true`: flag indicating whether the data should be log-transformed from the unit Fréchet scale to the `Gumbel` scale.

# Examples
```
using NeuralEstimators
using Distances
using LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)

# Passing a Cholesky factor:
D = pairwise(Euclidean(), S, S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulateschlather(L)

# Passing a GaussianRandomField with Cholesky decomposition:
using GaussianRandomFields
cov = CovarianceFunction(2, Matern(ρ, ν))
grf = GaussianRandomField(cov, GaussianRandomFields.Cholesky(), S)
simulateschlather(grf)

# Passing a GaussianRandomField with circulant embedding (fast but requires regular grid):
pts = range(0.0, 1.0, 20)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding = 100)
simulateschlather(grf)
```
"""
function simulateschlather(obj::M, m::Integer; C = 3.5, Gumbel::Bool = true) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number
	y = [simulateschlather(obj, C = C, Gumbel = Gumbel) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

function simulateschlather(obj::M; C = 3.5, Gumbel::Bool = true) where M <: Union{AbstractMatrix{T}, GaussianRandomField} where T <: Number

	# small hack to get the right eltype
	if typeof(obj) <: GaussianRandomField G = eltype(obj.cov) else G = T end

	n   = size(obj, 1)  # number of observations
	Z   = fill(zero(G), n)
	ζ⁻¹ = randexp(G)
	ζ   = 1 / ζ⁻¹

	# We must enforce E(max{0, Yᵢ}) = 1. It can
	# be shown that this condition is satisfied if the marginal variance of Y(⋅)
	# is equal to 2π. Now, our simulation design embeds a marginal variance of 1
	# into fields generated from the cholesky factors, and hence
	# simulategaussianprocess(L) returns simulations from a Gaussian
	# process with marginal variance 1. To scale the marginal variance to
	# 2π, we therefore need to multiply the field by √(2π).

	# Note that, compared with Algorithm 1.2.2 of Dey DK, Yan J (2016),
	# some simplifications have been made to the code below. This is because
	# max{Z(s), ζW(s)} ≡ max{Z(s), max{0, ζY(s)}} = max{Z(s), ζY(s)}, since
	# Z(s) is initialised to 0 and increases during simulation.

	while (ζ * C) > minimum(Z)
		Y = simulategaussianprocess(obj)
		Y = √(G(2π)) * Y
		Z = max.(Z, ζ * Y)
		E = randexp(G)
		ζ⁻¹ += E
		ζ = 1 / ζ⁻¹
	end

	# Log transform the data from the unit Fréchet scale to the Gumbel scale,
	# which stabilises the variance and helps to prevent neural-network collapse.
	if Gumbel Z = log.(Z) end

	return Z
end

# ---- Miscellaneous functions ----

#NB Currently, second order optimisation methods cannot be used
# straightforwardly because besselk() is not differentiable. In the future, we
# can add an argument to matern() and maternchols(), besselfn = besselk, which
# allows the user to change the bessel function to use adbesselk(), which
# allows automatic differentiation: see https://github.com/cgeoga/BesselK.jl.
@doc raw"""
    matern(h, ρ, ν, σ² = 1)
For two points separated by `h` units, compute the Matérn covariance function,
with range parameter `ρ`, smoothness parameter `ν`, and marginal variance parameter `σ²`.

We use the parametrisation
``C(\|\boldsymbol{h}\|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\boldsymbol{h}\|}{\rho}\right)^\nu K_\nu \left(\frac{\|\boldsymbol{h}\|}{\rho}\right)``,
where ``\Gamma(\cdot)`` is the gamma function, and ``K_\nu(\cdot)`` is the modified Bessel
function of the second kind of order ``\nu``.
"""
function matern(h, ρ, ν, σ² = one(typeof(h)))

	# Note that the `Julia` functions for ``\Gamma(\cdot)`` and ``K_\nu(\cdot)``, respectively `gamma()` and
	# `besselk()`, do not work on the GPU and, hence, nor does `matern()`.

	@assert h >= 0 "h should be non-negative"
	@assert ρ > 0 "ρ should be positive"
	@assert ν > 0 "ν should be positive"

	if h == 0
        C = σ²
    else
		d = h / ρ
        C = σ² * ((2^(1 - ν)) / gamma(ν)) * d^ν * besselk(ν, d)
    end
    return C
end


"""
    maternchols(D, ρ, ν, σ² = 1; stack = true)
Given a distance matrix `D`, constructs the Cholesky factor of the covariance matrix
under the Matérn covariance function with range parameter `ρ`, smoothness
parameter `ν`, and marginal variance `σ²`.

Providing vectors of parameters will yield a three-dimensional array of Cholesky factors (note
that the vectors must of the same length, but a mix of vectors and scalars is
allowed). A vector of distance matrices `D` may also be provided.

If `stack = true`, the Cholesky factors will be "stacked" into a
three-dimensional array (this is only possible if all distance matrices in `D`
are the same size).

# Examples
```
using NeuralEstimators
using LinearAlgebra: norm
n  = 10
S  = rand(n, 2)
D  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ ∈ eachrow(S)]
ρ  = [0.6, 0.5]
ν  = [0.7, 1.2]
σ² = [0.2, 0.4]
maternchols(D, ρ, ν)
maternchols([D], ρ, ν)
maternchols(D, ρ, ν, σ²; stack = false)

S̃  = rand(n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²)
maternchols([D, D̃], ρ, ν, σ²; stack = false)

S̃  = rand(2n, 2)
D̃  = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S̃), sⱼ ∈ eachrow(S̃)]
maternchols([D, D̃], ρ, ν, σ²; stack = false)
```
"""
function maternchols(D, ρ, ν, σ² = one(eltype(D)); stack::Bool = true)

	K = max(length(ρ), length(ν), length(σ²))
	if K > 1
		@assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)]) "`ρ`, `ν`, and `σ²` should be the same length"
		ρ  = _coercetoKvector(ρ, K)
		ν  = _coercetoKvector(ν, K)
		σ² = _coercetoKvector(σ², K)
	end

	# compute Cholesky factorization (exploit symmetry of D to minimise computations)
	# NB surprisingly, found that the parallel Folds.map() is slower than map(). Could try FLoops or other parallelisation packages.
	L = map(1:K) do k
		C = matern.(UpperTriangular(D), ρ[k], ν[k], σ²[k])
		L = cholesky(Symmetric(C)).L
		L = convert(Array, L) # convert from Triangular to Array so that stackarrays() can be used
		L
	end

	# Optionally convert from Vector of Matrices to 3D Array
	if stack
		L = stackarrays(L, merge = false)
	end

	return L
end


function maternchols(D::V, ρ, ν, σ² = one(nested_eltype(D)); stack::Bool = true) where {V <: AbstractVector{A}} where {A <: AbstractArray{T, N}} where {T, N}

	if stack
		@assert length(unique(size.(D))) == 1 "Converting the Cholesky factors from a vector of matrices to a three-dimenisonal array is only possible if the Cholesky factors (i.e., all matrices `D`) are the same size."
	end

	K = max(length(ρ), length(ν), length(σ²))
	if K > 1
		@assert all([length(θ) ∈ (1, K) for θ ∈ (ρ, ν, σ²)]) "`ρ`, `ν`, and `σ²` should be the same length"
		ρ  = _coercetoKvector(ρ, K)
		ν  = _coercetoKvector(ν, K)
		σ² = _coercetoKvector(σ², K)
	end
	@assert length(D) ∈ (1, K)

	# Compute the Cholesky factors
	L = maternchols.(D, ρ, ν, σ², stack = false)

	# L is currently a length-one Vector of Vectors: drop redundant outer vector
	L = stackarrays(L, merge = true)

	# Optionally convert from Vector of Matrices to 3D Array
	if stack
		L = stackarrays(L, merge = false)
	end
	return L
end

# Coerces a single-number or length-one-vector x into a K vector
function _coercetoKvector(x, K)
	@assert length(x) ∈ (1, K)
	if !isa(x, Vector) x = [x] end
	if length(x) == 1  x = repeat(x, K) end
	return x
end


# ---- Potts model ----

#TODO nsims argument

"""
	simulatepotts(grid::Matrix{Int}, β)
	simulatepotts(grid::Matrix{Union{Int, Nothing}}, β)
	simulatepotts(nrows::Int, ncols::Int, num_states::Int, β)

Chequerboard Gibbs sampling for a 2D Potts model.

# Keyword arguments
- num_iterations::Int = 2000
- burn::Int = 1000
- thin::Int = 10
- mask::Union{Matrix{Bool}, Nothing} = nothing

# Examples
```

## Marginal simulation 
using Random
Random.seed!(1234)
nrows, ncols =  10,10
num_states = 5
β = 0.8
simulatepotts(nrows, ncols, num_states, β)

using BenchmarkTools
num_iterations = 200
@belapsed simulatepotts(nrows, ncols, num_states, β, num_iterations = num_iterations)
# sequential: 0.113351459
# sequential, @inbounds: 0.113351459
# threaded: 

## Recreate Fig. 8.8 of Marin & Robert (2007) “Bayesian Core”
using Plots 
grids = [simulatepotts(100, 100, 2, β) for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio=1)
Plots.plot(heatmaps...)

## Conditional simulation 
β = 0.8
complete_grid   = simulatepotts(100, 100, 2, β)      # simulate from the Ising model 
incomplete_grid = removedata(complete_grid, 0.3)     # remove 30% of the pixels at random  
imputed_grid    = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels
"""
function simulatepotts(grid::AbstractMatrix{Int}, β; nsims::Integer = 1, burn::Int = 1000, thin::Int = 10, num_iterations::Int = 2000, mask = nothing)

	# TODO burn
	# TODO thin 
	# TODO return MCMC chain rather than just the end value 
	
	β = β[1] # remove the container if β was passed as a vector or a matrix 

	nrows, ncols = size(grid)
    num_states = maximum(grid) 

    # Define chequerboard patterns
    chequerboard1 = [(i+j) % 2 == 0 for i in 1:nrows, j in 1:ncols]
    chequerboard2 = .!chequerboard1
	if !isnothing(mask)
		@assert size(grid) == size(mask)
		chequerboard1 = chequerboard1 .&& mask 
		chequerboard2 = chequerboard2 .&& mask 
	end

    # Define neighbours offsets (assuming 4-neighbour connectivity)
    neighbour_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Gibbs sampling iterations
	for _ in 1:num_iterations
		for chequerboard in (chequerboard1, chequerboard2)
			for ci in findall(chequerboard)
	
				# Get cartesian coordinates of current pixel
				i, j = Tuple(ci)
	
				# Calculate conditional probabilities Pr(zᵢ | z₋ᵢ, β)
				n = zeros(num_states) # neighbour counts for each state
				for (di, dj) in neighbour_offsets
					ni, nj = i + di, j + dj
					if 1 <= ni <= nrows && 1 <= nj <= ncols
						@inbounds n[grid[ni, nj]] += 1
					end
				end
				probs = exp.(β * n) 
				probs /= sum(probs) # normalise 
				u = rand()
				new_state = findfirst(x -> x > u, cumsum(probs))
		
				# Update grid with new state
				@inbounds grid[i, j] = new_state
			end
		end
	end

    return grid
end



function simulatepotts(nrows::Int, ncols::Int, num_states::Int, β; kwargs...)
	grid = rand(1:num_states, nrows, ncols)
	simulatepotts(grid, β; kwargs...)
end

function simulatepotts(grid::AbstractMatrix{Union{Missing, I}}, β; kwargs...) where I <: Integer 

	# Avoid mutating the user's incomplete grid
	grid = copy(grid) 
	
	# Find the number of states 
	num_states = maximum(skipmissing(grid)) 
	
	# Compute the mask 
	mask = ismissing.(grid)

	# Replace missing entries with random states 
	grid[mask] .= rand(1:num_states, sum(mask))

	# Convert eltype of grid to Int 
	grid = convert(Matrix{I}, grid)

	# Conditionally simulate 
	simulatepotts(grid, β; kwargs..., mask = mask)
end

function simulatepotts(Z::A, β; kwargs...) where A <: AbstractArray{T, N} where {T, N}

  @assert all(size(Z)[3:end] .== 1) "Code for the Potts model is not equipped to handle independent replicates"

  # Save the original dimensions
	dims = size(Z)

	# Convert to matrix and pass to the matrix method
	Z = simulatepotts(Z[:, :], β; kwargs...)

	# Convert Z to the correct dimensions
	Z = reshape(Z, dims[1:end-1]..., :)
end
