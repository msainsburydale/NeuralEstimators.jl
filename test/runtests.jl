using NeuralEstimators
using NeuralEstimators: _getindices, _runondevice
import NeuralEstimators: simulate
using CUDA
using Distributions: Normal
using Flux
using Statistics: mean, sum
using Test
using Zygote

if CUDA.functional()
	@info "Testing on both the CPU and the GPU... "
	CUDA.allowscalar(false)
else
	@info "The GPU is unavailable so we'll test on the CPU only... "
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


@testset "core" begin
	n = 1
	v = [rand(Float32, n, 1, m) for m ∈ (7, 8, 9)]
	K = length(v)
	w = 4
	ψ = Chain(Dense(n, w), Dense(w, w))
	p = 1
	ϕ = Chain(Dense(w, w), Dense(w, p), Flux.flatten, x -> exp.(x))
	θ̂ = DeepSet(ψ, ϕ)
	@test size(θ̂(v), 1) == p
	@test size(θ̂(v), 2) == K

	# Test that we can use gradient descent to update the θ̂ weights
	loss = Flux.Losses.mae
	γ    = Flux.params(θ̂)
	θ    = rand(p, K)
	@test isa(loss(θ̂(v), θ), Number)

	optimiser = ADAM(0.01)
	gradients = gradient(() -> loss(θ̂(v), θ), γ)
	Flux.update!(optimiser, γ, gradients)

	# test train()
	θ̂ = train(θ̂, ξ, Parameters, m = 10, epochs = 5, savepath = "")
	# parameters = Parameters(ξ, 100)
	parameters = Parameters(ξ, 5000)
	θ̂ = train(θ̂, ξ, parameters, parameters, m = 10, epochs = 5, savepath = "") # FIXME not good seeing so many warnings when K = 100; think it's _ParameterLoader?
	θ̂ = train(θ̂, ξ, parameters, parameters, m = 10, epochs = 5, savepath = "", epochs_per_Z_refresh = 2) # FIXME not good seeing so many warnings; think it's _ParameterLoader?

	# test _runondevice()
	θ̂₁ = θ̂(v)
	θ̂₂ = _runondevice(θ̂, v, false)
	@test size(θ̂₁) == size(θ̂₂)
	@test θ̂₁ ≈ θ̂₂

	# test estimate()
	estimate([θ̂], ξ, parameters, m = [30, 90, 150])

	# Test on the GPU if it is available
	if CUDA.functional()
		θ̂ = θ̂ |> gpu
		v = v |> gpu
		θ = θ |> gpu

		θ̂ = θ̂(v)
		@test size(θ̂, 1) == p
		@test size(θ̂, 2) == N

		gradients = gradient(() -> loss(θ̂(v), θ), γ)
		Flux.update!(optimiser, γ, gradients)
	end
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
#
