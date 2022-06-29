using NeuralEstimators
using NeuralEstimators: _getindices, _runondevice
import NeuralEstimators: simulate
using CUDA
using DataFrames
using Distributions: Normal
using Flux
using Statistics: mean, sum
using Test
using Zygote

if CUDA.functional()
	@info "Testing on both the CPU and the GPU... "
	CUDA.allowscalar(false)
	devices = (CPU = cpu, GPU = gpu)
else
	@info "The GPU is unavailable so we'll test on the CPU only... "
	devices = (CPU = cpu,)
end

@testset "expandgrid" begin
    @test expandgrid(1:2, 0:3) == [1 0; 2 0; 1 1; 2 1; 1 2; 2 2; 1 3; 2 3]
    @test expandgrid(1:2, 1:2) == expandgrid(2)
end

@testset "_getindices" begin
	m = (3, 4, 6)
	v = [rand(16, 16, 1, mᵢ) for mᵢ ∈ m]
	@test _getindices(v) == [1:3, 4:7, 8:13]
end

@testset "stackarrays" begin
	# Vector containing arrays of the same size:
	A = rand(2, 3, 4); v = [A, A]; N = ndims(A);
	@test stackarrays(v) == cat(v..., dims = N)
	@test stackarrays(v, merge = false) == cat(v..., dims = N + 1)

	# Vector containing arrays with differing final dimension size:
	A₁ = rand(2, 3, 4); A₂ = rand(2, 3, 5); v = [A₁, A₂];
	@test stackarrays(v) == cat(v..., dims = N)
end

@testset "subsetparameters" begin

	struct TestParameters <: ParameterConfigurations
		v
		θ
		chols
	end

	K = 4
	parameters = TestParameters(rand(K), rand(3, K), rand(2, 2, K))
	indices = 2:3
	parameters_subset = subsetparameters(parameters, indices)
	@test parameters_subset.θ     == parameters.θ[:, indices]
	@test parameters_subset.chols == parameters.chols[:, :, indices]
	@test parameters_subset.v     == parameters.v[indices]
end

# Simple example for testing.
struct Parameters <: ParameterConfigurations θ end
ξ = (Ω = Normal(0, 0.5), σ = 1)
function Parameters(ξ, K::Integer)
	θ = rand(ξ.Ω, 1, K)
	Parameters(θ)
end
function simulate(parameters::Parameters, ξ, m::Integer)
	n = 1
	θ = vec(parameters.θ)
	Z = [rand(Normal(μ, ξ.σ), n, 1, m) for μ ∈ θ]
end
parameters = Parameters(ξ, 5000)
# parameters = Parameters(ξ, 100) # FIXME the fixed-parameter method for train() gives many warnings when K = 100; think it's _ParameterLoader?

n = 1
K = 100

w = 32
p = 1
ψ = Chain(Dense(n, w), Dense(w, w))
ϕ = Chain(Dense(w, w), Dense(w, p), Flux.flatten, x -> exp.(x))
θ̂_deepset = DeepSet(ψ, ϕ)
S = [samplesize]
ϕ₂ = Chain(Dense(w + length(S), w), Dense(w, p), Flux.flatten, x -> exp.(x))
θ̂_deepsetexpert = DeepSetExpert(θ̂_deepset, ϕ₂, S)
estimators = (DeepSet = θ̂_deepset, DeepSetExpert = θ̂_deepsetexpert)



verbose = false

@testset verbose = true "$key" for key ∈ keys(estimators)

	θ̂ = estimators[key]

	@testset "$ky" for ky ∈ keys(devices)

		device = devices[ky]
		θ̂ = θ̂ |> device

		loss = Flux.Losses.mae |> device
		γ    = Flux.params(θ̂)  |> device
		θ    = rand(p, K)      |> device

		Z = [randn(Float32, n, 1, m) for m ∈ rand(1:30, K)] |> device
		@test size(θ̂(Z), 1) == p
		@test size(θ̂(Z), 2) == K
		@test isa(loss(θ̂(Z), θ), Number)

		# Test that we can use gradient descent to update the θ̂ weights
		optimiser = ADAM(0.01)
		gradients = gradient(() -> loss(θ̂(Z), θ), γ)
		Flux.update!(optimiser, γ, gradients)

	    use_gpu = device == gpu
		@testset "train" begin
			θ̂ = train(θ̂, ξ, Parameters, m = 10, epochs = 5, savepath = "", use_gpu = use_gpu, verbose = verbose)
			θ̂ = train(θ̂, ξ, parameters, parameters, m = 10, epochs = 5, savepath = "", use_gpu = use_gpu, verbose = verbose)
			θ̂ = train(θ̂, ξ, parameters, parameters, m = 10, epochs = 5, savepath = "", epochs_per_Z_refresh = 2, use_gpu = use_gpu, verbose = verbose)
		end

		# FIXME On the GPU, get bug in this test: I think that a variable is undefined?
		@testset "_runondevice" begin
			θ̂₁ = θ̂(Z)
			θ̂₂ = _runondevice(θ̂, Z, use_gpu)
			@test size(θ̂₁) == size(θ̂₂)
			@test θ̂₁ ≈ θ̂₂ # checked that this is fine by seeing if the following replacement fixes things: @test maximum(abs.(θ̂₁ .- θ̂₂)) < 0.0001
		end

		@testset "estimate" begin
			estimates = estimate([θ̂], ξ, parameters, m = [30, 90, 150], use_gpu = use_gpu, verbose = verbose)
			@test typeof(merge(estimates)) == DataFrame
		end
	end
end


@testset "DeepSetPiecewise" begin
	θ̂_deepsetpiecewise = DeepSetPiecewise((θ̂_deepset, θ̂_deepsetexpert), (30))
	Z = [randn(Float32, n, 1, 10),  randn(Float32, n, 1, 50)]
	θ̂₁ = hcat(θ̂_deepset(Z[[1]]), θ̂_deepsetexpert(Z[[2]]))
	θ̂₂ = θ̂_deepsetpiecewise(Z)
	@test θ̂₁ ≈ θ̂₂
end




# @testset "DeepSetExpert" begin
# 	n = 1
# 	v = [rand(Float32, n, 1, m) for m ∈ (7, 8, 9)]
# 	N = length(v)
# 	w = 4
# 	qₜ = 3
# 	ψ = Chain(Dense(n, w), Dense(w, qₜ))
# 	S = [samplesize, mean, sum] # NB Needs to be a vector and not a tuple
# 	qₛ = length(S)
# 	p = 2
# 	ϕ = Chain(Dense(qₜ + qₛ, w), Dense(w, p), Flux.flatten)
# 	network = DeepSetExpert(ψ, ϕ, S)
# 	θ̂ = network(v)
# 	@test size(θ̂, 1) == p
# 	@test size(θ̂, 2) == N
#
# 	# Test that we can use gradient descent to update the network weights
# 	loss       = Flux.Losses.mae
#     parameters = Flux.params(network)
#     optimiser  = ADAM(0.01)
# 	θ = rand(p, N)
# 	@test isa(loss(network(v), θ), Number)
#
# 	gradients = gradient(() -> loss(network(v), θ), parameters) # FIXME error in optimised version occurs in this line. Something to do with stack(). I tested it with vectors containing equal sized arrays, and it's still broken.
# 	Flux.update!(optimiser, parameters, gradients)
#
# 	# Test on the GPU if it is available
# 	if CUDA.functional()
# 		network = network |> gpu
# 		v = v |> gpu
# 		θ = θ |> gpu
#
# 		θ̂ = network(v)
# 		@test size(θ̂, 1) == p
# 		@test size(θ̂, 2) == N
#
# 		gradients = gradient(() -> loss(network(v), θ), parameters)
# 		Flux.update!(optimiser, parameters, gradients)
#
# 		# Code for prototyping the DeepSetExpert function if needed:
# 		# import SpatialDeepSets: DeepSetExpert
# 		# d = network
# 		# t = d.Σ.(d.ψ.(v))
# 	    # s = d.S.(v)
# 		# x = v[1]
# 		# convert(CuArray, d.S(x))
# 		# s = map(v) do x
# 		# 	d.S(x)
# 		# end
# 		# Stuple = (samplesize, mean, sum)
# 		# map
# 		#
# 	    # s = s |> gpu # FIXME s needs to be on the GPU from the call above... shouldn't have to move it there.
# 	    # u = vcat.(t, s)
# 	    # θ̂ = d.ϕ.(u)
# 	    # θ̂ = stack(θ̂)
# 	end
# end




# print("Testing expertstatistics()... ")
# let
# 	S = [samplesize, maximum, mean]
#
# 	# Single array (i.e., a single parameter configuration)
# 	m = 5
# 	z = reshape(10:(10+m-1), 1, 1, m) |> copy
# 	z = Float32.(z)
#
# 	s = expertstatistics(S, z)
# 	@test s[1] == m
# 	@test s[2] == 10+m-1
#
# 	# Vector of arrays (i.e., multiple parameter configurations)
# 	nᵥ = 4
# 	v = [reshape(10:(10+m-1), 1, 1, m) |> copy for m in 1:nᵥ]
# 	v = broadcast.(Float32, v)
# 	s = expertstatistics(S, v)
# 	@test s[1]  == [s(v[1])  for s ∈ S]
# 	@test s[nᵥ] == [s(v[nᵥ]) for s ∈ S]
#
# 	@test wrappertype(expertstatistics(S, z)) == wrappertype(z)
# 	@test wrappertype(expertstatistics(S, v)) == wrappertype(v)
# 	if CUDA.functional()
# 		z = z |> gpu
# 		v = v |> gpu
# 		expertstatistics(S, z)
# 		expertstatistics(S, v)
# 		@test wrappertype(expertstatistics(S, z)) == wrappertype(z)
# 		@test wrappertype(expertstatistics(S, v)) == wrappertype(v)
#
# 	end
# end
# printstyled("Test Passed\n"; bold = true, color = :green)
