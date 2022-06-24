# ---- Parameteric bootstrap ----

"""
	parametricbootstrap(θ̂, parameters::P, m::Integer; B::Integer = 100, use_gpu::Bool = true) where {P <: ParameterConfigurations}

Returns `B` parameteric bootstrap samples of an estimator `θ̂` as a p × `B`
matrix, where p is the number of parameters in the statistical model, based on
simulated data sets of size `m`.

This function requires a method `simulate(parameters::P, m::Integer`).
"""
function parametricbootstrap(θ̂, parameters::P, m::Integer; B::Integer = 100, use_gpu::Bool = true) where {P <: ParameterConfigurations}
	Z̃ = simulate(parameters, m, B)
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃) #TODO Need to add _checkgpu() calls instead of just use_gpu
	return θ̃
end

# ---- Non-parametric bootstrapping ----

"""
	nonparametricbootstrap(θ̂, Z; B::Integer = 100, use_gpu::Bool = true)
	nonparametricbootstrap(θ̂, Z, blocks; B::Integer = 100, use_gpu::Bool = true)

Returns `B` non-parametric bootstrap samples of an estimator `θ̂` as a p × `B`
matrix, where p is the number of parameters in the statistical model.

The argument `blocks` caters for block bootstrapping, and should be an integer
vector specifying the block for each replicate. For example, if we have 5
replicates with the first two replicates corresponding to block 1 and the
remaining replicates corresponding to block 2, then `blocks` should be
[1, 1, 2, 2, 2]. The resampling algorithm tries to produce resampled data sets
of a similar size to the original data, but this can only be achieved exactly if
the blocks are the same length.
"""
function nonparametricbootstrap(θ̂, Z; B::Integer = 100, use_gpu::Bool = true)
	Z̃ = _resample(Z, B)
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃) #TODO Need to add _checkgpu() calls instead of just use_gpu
	return θ̃
end

# ---- Non-parametric block bootstrapping ----


function nonparametricbootstrap(θ̂, Z, blocks; B::Integer = 100, use_gpu::Bool = true)
	Z̃ = _resample(Z, B, blocks)
	θ̃ = use_gpu ? _runondevice(θ̂, Z̃, true) : θ̂(Z̃) #TODO Need to add _checkgpu() calls instead of just use_gpu
	return θ̃
end


# ---- Resampling functions ----

"""
Samples Z with replacement, generating B bootstrap samples with the same sample size
as the original data, Z. The function assumes that the last dimension of Z is
the replicates dimension and, hence, Z is resampled over its last dimension.
Here, blocks is an integer vector specifying the block for each replicate; for
example, if we have 5 replicates with the first two replicates corresponding to
block 1 and the remaining replicates corresponding to block 2, then blocks
should be [1, 1, 2, 2, 2]. The function aims to produce data sets that are of a
similar size to the original data, but this can only be achieved exactly if the
blocks are of equal size.
"""
function _resample(Z::A, B, blocks) where {A <: AbstractArray{T, N}} where {T, N}
	n = size(Z)[end]
	@assert length(blocks) == n "The number of replicates and the length of the blocks vector does not match"
	colons = ntuple(_ -> (:), ndims(Z) - 1)
	unique_blocks = blocks |> unique
	num_blocks = unique_blocks |> length

	# Size of the resampled data set, ñ, is not necessarily close to n.
	# num_fields_each_block = n ÷ num_blocks # only an approximation in most cases
	# Z̃ = map(1:B) do _
	# 	sampled_blocks = rand(unique_blocks, num_fields_each_block)
	# 	idx = vcat([findall(x -> x == i, blocks) for i ∈ sampled_blocks]...)
	# 	# Z[colons..., idx]
	# end

	# Define c ≡ median(block_counts)/2 and d ≡ maximum(block_counts).
	# The following method ensures that ñ ∈ [n - c, n - c + d), where
	# ñ is the size of the resampled data set.
	n = length(blocks)
	block_counts = [count(x -> x == i, blocks) for i ∈ unique_blocks]
	c = median(block_counts) / 2
	Z̃ = map(1:B) do _
		sampled_blocks = Int[]
		ñ = 0
		while ñ < n - c
			push!(sampled_blocks, rand(unique_blocks))
			ñ += block_counts[sampled_blocks[end]]
		end
		idx = vcat([findall(x -> x == i, blocks) for i ∈ sampled_blocks]...)
		Z[colons..., idx]
	end

	return Z̃
end


"""
Samples Z with replacement, generating B bootstrap samples with the same sample
size as the original data, Z. The last dimension of Z is assumed to be the
replicates dimension and, hence, Z is resampled over its last dimension.
"""
function _resample(Z::A, B) where {A <: AbstractArray{T, N}} where {T, N}
	n = size(Z)[end]
	colons = ntuple(_ -> (:), ndims(Z) - 1)
	Z̃ = [Z[colons..., rand(1:n, n)] for _ in 1:B]
	return Z̃
end
