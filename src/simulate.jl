"""
	simulategaussian(L::AbstractMatrix, m = 1)
Simulates `m` independent and identically distributed realisations from
a mean-zero multivariate Gaussian random vector with associated lower Cholesky 
factor `L`. 

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are
returned as an ``n``x`m` matrix.

# Examples
```
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulategaussian(L)
```
"""
function simulategaussian(obj::M, m::Integer) where M <: AbstractMatrix{T} where T <: Number
	y = [simulategaussian(obj) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

function simulategaussian(L::M) where M <: AbstractMatrix{T} where T <: Number
	L * randn(T, size(L, 1))
end

#TODO better keyword argument name than Gumbel?
"""
	simulateschlather(L::Matrix, m = 1; C = 3.5, Gumbel::Bool = false)
Simulates `m` independent and identically distributed realisations from
[Schlather's (2002)](https://link.springer.com/article/10.1023/A:1020977924878) max-stable model given the lower Cholesky factor `L` of the covariance matrix of the underlying Gaussian process. 

The function uses the algorithm for approximate simulation given
by [Schlather (2002)](https://link.springer.com/article/10.1023/A:1020977924878).

If `m` is not specified, the simulated data are returned as a vector with
length equal to the number of spatial locations, ``n``; otherwise, the data are 
returned as an ``n``x`m` matrix.

# Keyword arguments
- `C = 3.5`: a tuning parameter that controls the accuracy of the algorithm. Small `C` favours computational efficiency, while large `C` favours accuracy. 
- `Gumbel = true`: flag indicating whether the data should be log-transformed from the unit Fréchet scale to the Gumbel scale.

# Examples
```
using NeuralEstimators, Distances, LinearAlgebra

n = 500
ρ = 0.6
ν = 1.0
S = rand(n, 2)
D = pairwise(Euclidean(), S, dims = 1)
Σ = Symmetric(matern.(D, ρ, ν))
L = cholesky(Σ).L
simulateschlather(L)
```
"""
function simulateschlather(obj::M, m::Integer; kwargs...) where M <: AbstractMatrix{T} where T <: Number
	y = [simulateschlather(obj; kwargs...) for _ ∈ 1:m]
	y = stackarrays(y, merge = false)
	return y
end

function simulateschlather(obj::M; C = 3.5, Gumbel::Bool = false) where M <: AbstractMatrix{T} where T <: Number

	n   = size(obj, 1)  # number of observations
	Z   = fill(zero(T), n)
	ζ⁻¹ = randexp(T)
	ζ   = 1 / ζ⁻¹

	# We must enforce E(max{0, Yᵢ}) = 1. It can
	# be shown that this condition is satisfied if the marginal variance of Y(⋅)
	# is equal to 2π. Now, our simulation design embeds a marginal variance of 1
	# into fields generated from the cholesky factors, and hence
	# simulategaussian(L) returns simulations from a Gaussian
	# process with marginal variance 1. To scale the marginal variance to
	# 2π, we therefore need to multiply the field by √(2π).

	# Note that, compared with Algorithm 1.2.2 of Dey DK, Yan J (2016),
	# some simplifications have been made to the code below. This is because
	# max{Z(s), ζW(s)} ≡ max{Z(s), max{0, ζY(s)}} = max{Z(s), ζY(s)}, since
	# Z(s) is initialised to 0 and increases during simulation.

	while (ζ * C) > minimum(Z)
		Y = simulategaussian(obj)
		Y = √(T(2π)) * Y
		Z = max.(Z, ζ * Y)
		E = randexp(T)
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
Given distance ``\|\boldsymbol{h}\|`` (`h`), computes the Matérn covariance function

```math
C(\|\boldsymbol{h}\|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\boldsymbol{h}\|}{\rho}\right)^\nu K_\nu \left(\frac{\|\boldsymbol{h}\|}{\rho}\right),
```

where `ρ` is a range parameter, `ν` is a smoothness parameter, `σ²` is the marginal variance, 
``\Gamma(\cdot)`` is the gamma function, and ``K_\nu(\cdot)`` is the modified Bessel
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

#TODO could make this more general by using Matern rather than an exponential covariance
@doc raw"""
    paciorek(s, r, ω₁, ω₂, ρ, β)
Given spatial locations `s` and `r`, computes the nonstationary covariance function 
```math
C(\boldsymbol{s}, \boldsymbol{r}) = 
|\boldsymbol{\Sigma}(\boldsymbol{s})|^{1/4}
|\boldsymbol{\Sigma}(\boldsymbol{r})|^{1/4}
\left|\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right|^{-1/2}
C^0\big(\sqrt{Q(\boldsymbol{s}, \boldsymbol{r})}\big), 
```
where $C^0(h) = \exp\{-(h/\rho)^{3/2}\}$ for range parameter $\rho > 0$, 
the matrix $\boldsymbol{\Sigma}(\boldsymbol{s}) = \exp(\beta\|\boldsymbol{s} - \boldsymbol{\omega}\|)\boldsymbol{I}$ 
is a kernel matrix ([Paciorek and Schervish, 2006](https://onlinelibrary.wiley.com/doi/abs/10.1002/env.785)) 
with scale parameter $\beta > 0$ and reference point $\boldsymbol{\omega} \equiv (\omega_1, \omega_2)' \in \mathbb{R}^2$,
and 
```math
Q(\boldsymbol{s}, \boldsymbol{r}) = 
(\boldsymbol{s} - \boldsymbol{r})'
\left(\frac{\boldsymbol{\Sigma}(\boldsymbol{s}) + \boldsymbol{\Sigma}(\boldsymbol{r})}{2}\right)^{-1}
(\boldsymbol{s} - \boldsymbol{r})
``` 
is the squared Mahalanobis distance between $\boldsymbol{s}$ and $\boldsymbol{r}$. 

Note that, in practical applications, the reference point $\boldsymbol{\omega}$ is often taken to be an estimable parameter rather than fixed and known. 
"""
function paciorek(s, r, ω₁, ω₂, ρ, β)

    # Displacement vector
    h = s - r

    # Distance from each point to ω ≡ (ω₁, ω₂)'
    dₛ = sqrt((s[1] - ω₁)^2 + (s[2] - ω₂)^2)
    dᵣ = sqrt((r[1] - ω₁)^2 + (r[2] - ω₂)^2)

    # Scaling factors of kernel matrices, such that Σ(s) = a(s)I
    aₛ = exp(β * dₛ) 
    aᵣ = exp(β * dᵣ) 

    # Several computational efficiencies afforded by use of a diagonal kernel matrix:
    # - the inverse of a diagonal matrix is given by replacing the diagonal elements with their reciprocals
    # - the determinant of a diagonal matrix is equal to the product of its diagonal elements

    # Mahalanobis distance
    Q = 2 * h'h / (aₛ + aᵣ)

    # Explicit version of code 
    # Σₛ_det = aₛ^2   
    # Σᵣ_det = aᵣ^2   
    # C⁰ = exp(-sqrt(Q/ρ)^1.5)
    # logC = 1/4*log(Σₛ_det) + 1/4*log(Σᵣ_det) - log((aₛ + aᵣ)/2) + log(C⁰)
    
    # Numerically stable version of code
    logC = β*dₛ/2 + β*dᵣ/2 - log((aₛ + aᵣ)/2) - (sqrt(Q)/ρ)^1.5

    exp(logC)
end 


"""
    maternchols(D, ρ, ν, σ² = 1; stack = true)
Given a matrix `D` of distances, constructs the Cholesky factor of the covariance matrix
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


"""
	simulatepotts(grid::Matrix{Int}, β)
	simulatepotts(grid::Matrix{Union{Int, Nothing}}, β)
	simulatepotts(nrows::Int, ncols::Int, num_states::Int, β)
Chequerboard Gibbs sampling from a spatial Potts model with parameter `β`>0 (see, e.g., [Sainsbury-Dale et al., 2025, Sec. 3.3](https://arxiv.org/abs/2501.04330), and the references therein).

Approximately independent simulations can be obtained by setting 
`nsims` > 1 or `num_iterations > burn`. The degree to which the 
resulting simulations can be considered independent depends on the 
thinning factor (`thin`) and the burn-in (`burn`).

# Keyword arguments
- `nsims = 1`: number of approximately independent replicates. 
- `num_iterations = 2000`: number of MCMC iterations.
- `burn = num_iterations`: burn-in.
- `thin = 10`: thinning factor.

# Examples
```
using NeuralEstimators 

## Marginal simulation 
β = 0.8
simulatepotts(10, 10, 5, β)

## Marginal simulation: approximately independent samples 
simulatepotts(10, 10, 5, β; nsims = 100, thin = 10)

## Conditional simulation 
β = 0.8
complete_grid   = simulatepotts(50, 50, 2, β)        # simulate marginally from the Ising model 
incomplete_grid = removedata(complete_grid, 0.1)     # remove 10% of the pixels at random  
imputed_grid    = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels

## Multiple conditional simulations 
imputed_grids   = simulatepotts(incomplete_grid, β; num_iterations = 2000, burn = 1000, thin = 10)

## Recreate Fig. 8.8 of Marin & Robert (2007) “Bayesian Core”
using Plots 
grids = [simulatepotts(100, 100, 2, β) for β ∈ 0.3:0.1:1.2]
heatmaps = heatmap.(grids, legend = false, aspect_ratio=1)
Plots.plot(heatmaps...)
```
"""
function simulatepotts(grid::AbstractMatrix{Int}, β; nsims::Int = 1, num_iterations::Int = 2000, burn::Int = num_iterations, thin::Int = 10, mask = nothing)
	
	#TODO Int or Integer?

	@assert burn <= num_iterations
	if burn < num_iterations || nsims > 1
		Z₀ = simulatepotts(grid, β; num_iterations = burn, mask = mask) 
		Z_chain = [Z₀]

		# If the user has left nsims unspecified, determine it based on the other arguments
		# NB num_iterations is ignored in the case that nsims > 1. 
		if nsims == 1  
			nsims = (num_iterations - burn) ÷ thin
		end 
		
		for i in 1:nsims-1 
			z = copy(Z_chain[i])
			z = simulatepotts(z, β; num_iterations = thin, mask = mask)
			push!(Z_chain, z)
		end

		return Z_chain
	end 
	
  	β = β[1] # remove the container if β was passed as a vector or a matrix 
  
  	nrows, ncols = size(grid)
    states = unique(skipmissing(grid))
  	num_states = length(states)

    # Define chequerboard patterns
    chequerboard1 = [(i+j) % 2 == 0 for i in 1:nrows, j in 1:ncols]
    chequerboard2 = .!chequerboard1
  	if !isnothing(mask)
		#TODO check sum(mask) != 0 (return unaltered grid in this case, with a warning)
  		@assert size(grid) == size(mask)
  		chequerboard1 = chequerboard1 .&& mask 
  		chequerboard2 = chequerboard2 .&& mask 
  	end
  	#TODO sum(chequerboard1) == 0 (easy workaround in this case, just iterate over chequerboard2)
  	#TODO sum(chequerboard2) == 0 (easy workaround in this case, just iterate over chequerboard1)
	
    # Define neighbours offsets based on 4-neighbour connectivity
    neighbour_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]

  	for _ in 1:num_iterations
  		for chequerboard in (chequerboard1, chequerboard2)

			# ---- Vectorised version (this implementation doesn't seem to save any time) ----

			# # Compute neighbour counts for each state
			# # NB some wasted computations because we are comuting the neighbour counts for all grid cells, not just the current chequerboard
			# padded_grid = padarray(grid, 1, minimum(states) - 1) # pad grid with label that is outside support 
			# neighbor_counts = zeros(nrows, ncols, num_states)
			# for (di, dj) in neighbour_offsets
			# 	row_indices = 2+di:nrows+di+1
			# 	col_indices = 2+dj:ncols+dj+1
			# 	shifted_grid = padded_grid[row_indices, col_indices]
			# 	for k in 1:num_states
			# 	 	neighbor_counts[:, :, k] .+= (shifted_grid .== states[k])
			# 	end			
			# end

			# # Calculate conditional probabilities
			# probs = exp.(β .* neighbor_counts)
			# probs ./= sum(probs, dims=3)


			# # Sample new states for chequerboard cells
			# cumulative_probs = cumsum(probs, dims=3)
			# rand_matrix = rand(nrows, ncols) 

			# # Indices of chequerboard cells
			# chequerboard_indices = findall(chequerboard)

			# # Sample new states and update grid  
			# sampled_indices = map(i -> findfirst(cumulative_probs[Tuple(i)..., :] .> rand_matrix[Tuple(i)...]), chequerboard_indices)
			# grid[chequerboard] .= states[sampled_indices]


			# ---- Simple version ----


  			for ci in findall(chequerboard)

  				# Get cartesian coordinates of current pixel
  				i, j = Tuple(ci)
  	
  				# Calculate conditional probabilities Pr(zᵢ | z₋ᵢ, β)
  				n = zeros(num_states) # neighbour counts for each state
  				for (di, dj) in neighbour_offsets
  					ni, nj = i + di, j + dj
  					if 1 <= ni <= nrows && 1 <= nj <= ncols
  						state = grid[ni, nj]
  						index = findfirst(x -> x == state, states)
  						n[index] += 1
  					end
  				end
  				probs = exp.(β * n) 
  				probs /= sum(probs) # normalise 
  				u = rand()
  				new_state_index = findfirst(x -> x > u, cumsum(probs))
  				new_state = states[new_state_index]
  		
  				# Update grid with new state
  				grid[i, j] = new_state
  			end
  		end
  	end

    return grid
end

# function padarray(grid, pad_size, pad_value)
#     padded_grid = fill(pad_value, size(grid)[1] + 2*pad_size, size(grid)[2] + 2*pad_size)
#     padded_grid[pad_size+1:end-pad_size, pad_size+1:end-pad_size] .= grid
#     return padded_grid
# end

function simulatepotts(nrows::Int, ncols::Int, num_states::Int, β; kwargs...)
    @assert length(β) == 1
    β = β[1]
    β_crit = log(1 + sqrt(num_states))
    if β < β_crit
        # Random initialization for high temperature
        grid = rand(1:num_states, nrows, ncols)
    else
        # Clustered initialization for low temperature
        cluster_size = max(1, min(nrows, ncols) ÷ 4)  
        clustered_rows = ceil(Int, nrows / cluster_size)
        clustered_cols = ceil(Int, ncols / cluster_size)
        base_grid = rand(1:num_states, clustered_rows, clustered_cols)
        grid = repeat(base_grid, inner=(cluster_size, cluster_size))
        grid = grid[1:nrows, 1:ncols]  # Trim to exact dimensions
    end
    simulatepotts(grid, β; kwargs...)
end

function simulatepotts(grid::AbstractMatrix{Union{Missing, I}}, β; kwargs...) where I <: Integer 

	# Avoid mutating the user's incomplete grid
	grid = copy(grid) 
	
	# Find the number of states 
	states = unique(skipmissing(grid))
	
	# Compute the mask 
	mask = ismissing.(grid)

	# Replace missing entries with random states 
	# TODO might converge faster with a better initialisation
	grid[mask] .= rand(states, sum(mask))

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