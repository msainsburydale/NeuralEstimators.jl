using NeuralEstimators
using NeuralEstimators: _getindices, _runondevice, _incgammalowerunregularised
import NeuralEstimators: simulate
using CUDA
using DataFrames
using Distributions: Normal, cdf, logpdf, quantile
using Flux
using Flux: DataLoader
using Flux: mae
using Graphs
using GraphNeuralNetworks
using LinearAlgebra
using Random: seed!
using SpecialFunctions: gamma
using Statistics
using Statistics: mean, sum
using Test
using Zygote
array(size...; T = Float64) = T.(reshape(1:prod(size), size...) ./ prod(size))

verbose = false # verbose used in the NeuralEstimators code

if CUDA.functional()
	@info "Testing on both the CPU and the GPU... "
	CUDA.allowscalar(false)
	devices = (CPU = cpu, GPU = gpu)
else
	@info "The GPU is unavailable so we'll test on the CPU only... "
	devices = (CPU = cpu,)
end

@testset "loss functions" begin
	p = 3
	K = 10
	θ̂ = array(p, K)
	θ = array(p, K)
	@test quantileloss(θ̂, θ, 0.5) ≈ 0.5 * mae(θ̂, θ)
end

@testset "UtilityFunctions" begin

	@testset "expandgrid" begin
	    @test expandgrid(1:2, 0:3) == [1 0; 2 0; 1 1; 2 1; 1 2; 2 2; 1 3; 2 3]
	    @test expandgrid(1:2, 1:2) == expandgrid(2)
	end

	@testset "_getindices" begin
		m = (3, 4, 6)
		v = [array(16, 16, 1, mᵢ) for mᵢ ∈ m]
		@test _getindices(v) == [1:3, 4:7, 8:13]
	end

	@testset "stackarrays" begin
		# Vector containing arrays of the same size:
		A = array(2, 3, 4); v = [A, A]; N = ndims(A);
		@test stackarrays(v) == cat(v..., dims = N)
		@test stackarrays(v, merge = false) == cat(v..., dims = N + 1)

		# Vector containing arrays with differing final dimension size:
		A₁ = array(2, 3, 4); A₂ = array(2, 3, 5); v = [A₁, A₂];
		@test stackarrays(v) == cat(v..., dims = N)
	end

	# @testset "samplesize" begin
	# 	Z = array(3, 4, 1, 6)
	#     @test inversesamplesize(Z) ≈ 1/samplesize(Z)
	# end

end


@testset "incgamma" begin

	# tests based on the "Special values" section of https://en.wikipedia.org/wiki/Incomplete_gamma_function

	@testset "unregularised" begin

		reg = false
		a = 1.0

		x = a + 0.5 # x < (a + 1)
		@test incgamma(a, x, upper = true, reg = reg) ≈ exp(-x)
		@test incgamma(a, x, upper = false, reg = reg) ≈ 1 - exp(-x)
		@test _incgammalowerunregularised(a, x) ≈ incgamma(a, x, upper = false, reg = reg)

		x = a + 1.5 # x > (a + 1)
		@test incgamma(a, x, upper = true, reg = reg) ≈ exp(-x)
		@test incgamma(a, x, upper = false, reg = reg) ≈ 1 - exp(-x)
		@test _incgammalowerunregularised(a, x) ≈ incgamma(a, x, upper = false, reg = reg)

	end

	@testset "regularised" begin

		reg = true
		a = 1.0

		x = a + 0.5 # x < (a + 1)
		@test incgamma(a, x, upper = false, reg = true) ≈ incgamma(a, x, upper = false, reg = false)  / gamma(a)
		@test incgamma(a, x, upper = true, reg = true) ≈ incgamma(a, x, upper = true, reg = false)  / gamma(a)

		x = a + 1.5 # x > (a + 1)
		@test incgamma(a, x, upper = false, reg = true) ≈ incgamma(a, x, upper = false, reg = false)  / gamma(a)
		@test incgamma(a, x, upper = true, reg = true) ≈ incgamma(a, x, upper = true, reg = false)  / gamma(a)

	end

end

@testset "subsetparameters" begin

	struct TestParameters <: ParameterConfigurations
		v
		θ
		chols
	end

	K = 4
	parameters = TestParameters(array(K), array(3, K), array(2, 2, K))
	indices = 2:3
	parameters_subset = subsetparameters(parameters, indices)
	@test parameters_subset.θ     == parameters.θ[:, indices]
	@test parameters_subset.chols == parameters.chols[:, :, indices]
	@test parameters_subset.v     == parameters.v[indices]
end


@testset "Compress" begin

	p = 3
	a = [0.1, 4, 2]
	b = [0.9, 9, 3]
	l = Compress(a, b)
	K = 10
	θ = array(p, K)
	l(θ)
	@test all([all(a .< x .< b) for x ∈ eachcol(l(θ))])

	n = 20
	Z = array(n, K)
	θ̂ = Chain(Dense(n, 15), Dense(15, p), l)
	@test all([all(a .< x .< b) for x ∈ eachcol(θ̂(Z))])
end


@testset "simulation" begin
	S = array(10, 2, T = Float32)
	D = [norm(sᵢ - sⱼ) for sᵢ ∈ eachrow(S), sⱼ in eachrow(S)]
	ρ = Float32.([0.6, 0.8])
	ν = Float32.([0.5, 0.7])
	L = maternchols(D, ρ, ν)
	σ² = 0.5f0
	L = maternchols(D, ρ, ν, σ²)
	@test maternchols(D, ρ, ν, σ²) == maternchols([D, D], ρ, ν, σ²)
	L₁ = L[:, :, 1]
	m = 5

	@test eltype(simulateschlather(L₁, m)) == Float32
	# @code_warntype simulateschlather(L₁, m)

	@test eltype(simulategaussianprocess(L₁, m)) == Float32
	# @code_warntype simulategaussianprocess(L₁, σ, m)
end


@testset "densities" begin

	# "scaledlogistic"
	@test all(4 .<= scaledlogistic.(-10:10, 4, 5) .<= 5)
	@test all(scaledlogit.(scaledlogistic.(-10:10, 4, 5), 4, 5) .≈ -10:10)
	Ω = (σ = 1:10, ρ = (2, 7))
	Ω = [Ω...] # convert to array since broadcasting over dictionaries and NamedTuples is reserved
	θ = [-10, 15]
	@test all(minimum.(Ω) .<= scaledlogistic.(θ, Ω) .<= maximum.(Ω))
	@test all(scaledlogit.(scaledlogistic.(θ, Ω), Ω) .≈ θ)

	# Check that the pdf is consistent with the cdf using finite differences
	using NeuralEstimators: _schlatherbivariatecdf
	function finitedifference(z₁, z₂, ψ, ϵ = 0.0001)
		(_schlatherbivariatecdf(z₁ + ϵ, z₂ + ϵ, ψ) - _schlatherbivariatecdf(z₁ - ϵ, z₂ + ϵ, ψ) - _schlatherbivariatecdf(z₁ + ϵ, z₂ - ϵ, ψ) + _schlatherbivariatecdf(z₁ - ϵ, z₂ - ϵ, ψ)) / (4 * ϵ^2)
	end
	function finitedifference_check(z₁, z₂, ψ)
		@test abs(finitedifference(z₁, z₂, ψ) - schlatherbivariatedensity(z₁, z₂, ψ; logdensity=false)) < 0.0001
	end
	finitedifference_check(0.3, 0.8, 0.2)
	finitedifference_check(0.3, 0.8, 0.9)
	finitedifference_check(3.3, 3.8, 0.2)
	finitedifference_check(3.3, 3.8, 0.9)

	# Other small tests
	@test schlatherbivariatedensity(3.3, 3.8, 0.9; logdensity = false) ≈ exp(schlatherbivariatedensity(3.3, 3.8, 0.9))
	y = [0.2, 0.4, 0.3]
	n = length(y)
	# construct a diagonally dominant covariance matrix (pos. def. guaranteed via Gershgorins Theorem)
	Σ = array(n, n)
	Σ[diagind(Σ)] .= diag(Σ) + sum(Σ, dims = 2)
	@test gaussiandensity(y, Σ, logdensity = false) ≈ exp(gaussiandensity(y, Σ))
end


@testset "GNNEstimator" begin
	n₁, n₂ = 11, 27
	m₁, m₂ = 30, 50
	d = 1
	g₁ = rand_graph(n₁, m₁, ndata = array(d, n₁, T = Float32))
	g₂ = rand_graph(n₂, m₂, ndata = array(d, n₂, T = Float32))
	g = Flux.batch([g₁, g₂])

	# g is a single large GNNGraph containing the subgraphs
	@test g.num_graphs == 2
	@test g.num_nodes == n₁ + n₂
	@test g.num_edges == m₁ + m₂

	# Greate a mini-batch from g (use integer range to extract multiple graphs)
	@test getgraph(g, 1) == g₁

	# We can pass a single GNNGraph to Flux's DataLoader, and this will iterate over
	# the subgraphs in the expected manner.
	train_loader = DataLoader(g, batchsize=1, shuffle=true)
	for g in train_loader
	    @test g.num_graphs == 1
	end

	# graph-to-graph propagation module
	w = 5
	o = 7
	graphtograph = GNNChain(GraphConv(d => w), GraphConv(w => w), GraphConv(w => o))
	@test graphtograph(g) == Flux.batch([graphtograph(g₁), graphtograph(g₂)])

	# global pooling module
	# We can apply the pooling operation to the whole graph; however, I think this
	# is mainly possible because the GlobalPool with mean is very simple.
	# We may need to do something different for general global pooling layers (e.g.,
	# universal pooling with DeepSets).
	meanpool = GlobalPool(mean)
	h  = meanpool(graphtograph(g))
	h₁ = meanpool(graphtograph(g₁))
	h₂ = meanpool(graphtograph(g₂))
	@test graph_features(h) == hcat(graph_features(h₁), graph_features(h₂))

	# Deep Set module
	w = 32
	p = 3
	ψ₂ = Chain(Dense(o, w, relu), Dense(w, w, relu), Dense(w, w, relu))
	ϕ₂ = Chain(Dense(w, w, relu), Dense(w, p))
	deepset = DeepSet(ψ₂, ϕ₂)

	# Full estimator
	est = GNNEstimator(graphtograph, meanpool, deepset)

	# Test on a single graph containing sub-graphs
	θ̂ = est(g)
	@test size(θ̂, 1) == p
	@test size(θ̂, 2) == 1

	# test on a vector of graphs
	v = [g₁, g₂, Flux.batch([g₁, g₂])]
	θ̂ = est(v)
	@test size(θ̂, 1) == p
	@test size(θ̂, 2) == length(v)

	# test that it can be trained
	K = 10
	Z = [rand_graph(n₁, m₁, ndata = array(d, n₁, T = Float32)) for _ in 1:K]
	θ = array(p, K)
	train(est, θ, θ, Z, Z; batchsize = 2, epochs = 3, verbose = verbose)
end


# Simple example for testing.
struct Parameters <: ParameterConfigurations
	θ
	σ
end
ξ = (Ω = Normal(0, 0.5), σ = 1)
Parameters(K::Integer, ξ) = Parameters(rand(ξ.Ω, 1, K), ξ.σ)
function simulate(parameters::Parameters, m::Integer)
	n = 1
	θ = vec(parameters.θ)
	Z = [rand(Normal(μ, parameters.σ), n, m) for μ ∈ θ]
end
parameters = Parameters(100, ξ)

MLE(Z) = mean.(Z)'
MLE(Z, ξ) = MLE(Z) # the MLE doesn't need ξ, but we include it for testing

n = 1
K = 100
w = 32
p = 1
ψ = Chain(Dense(n, w), Dense(w, w), Flux.flatten)
ϕ = Chain(Dense(w, w), Dense(w, p))
θ̂_deepset = DeepSet(ψ, ϕ)
S = samplesize
ϕₛ = Chain(Dense(w + 1, w), Dense(w, p))
θ̂_deepsetexpert = DeepSetExpert(ψ, ϕₛ, S)
dₓ= 2
estimators = (DeepSet = θ̂_deepset, DeepSetExpert = θ̂_deepsetexpert)


@testset verbose = true "$key" for key ∈ keys(estimators)

	# key = :DeepSet
	θ̂ = estimators[key]

	@testset "$ky" for ky ∈ keys(devices)


		# ky = :CPU
		device = devices[ky]
		θ̂ = θ̂ |> device

		loss = Flux.Losses.mae |> device
		γ    = Flux.params(θ̂)  |> device
		θ    = array(p, K)      |> device

		Z = [array(n, m, T = Float32) for m ∈ rand(29:30, K)] |> device
		@test size(θ̂(Z), 1) == p
		@test size(θ̂(Z), 2) == K
		@test isa(loss(θ̂(Z), θ), Number)

		# Test that we can use gradient descent to update the θ̂ weights
		optimiser = ADAM(0.01)
		gradients = gradient(() -> loss(θ̂(Z), θ), γ)
		Flux.update!(optimiser, γ, gradients)

	    use_gpu = device == gpu
		@testset "train" begin

			# train: single estimator
			θ̂ = train(θ̂, Parameters, simulate, m = 10, epochs = 5, use_gpu = use_gpu, verbose = verbose, ξ = ξ)
			θ̂ = train(θ̂, parameters, parameters, simulate, m = 10, epochs = 5, use_gpu = use_gpu, verbose = verbose)
			θ̂ = train(θ̂, parameters, parameters, simulate, m = 10, epochs = 5, epochs_per_Z_refresh = 2, use_gpu = use_gpu, verbose = verbose)
			θ̂ = train(θ̂, parameters, parameters, simulate, m = 10, epochs = 5, epochs_per_Z_refresh = 1, simulate_just_in_time = true, use_gpu = use_gpu, verbose = verbose)
			Z_train = simulate(parameters, 20);
			Z_val   = simulate(parameters, 10);
			train(θ̂, parameters, parameters, Z_train, Z_val; epochs = 5, use_gpu = use_gpu, verbose = verbose)

			# trainx: Multiple estimators
			trainx(θ̂, Parameters, simulate, [1, 2, 5]; ξ = ξ, epochs = [10, 5, 3], use_gpu = use_gpu, verbose = verbose) 
			trainx(θ̂, parameters, parameters, simulate, [1, 2, 5]; epochs = [10, 5, 3], use_gpu = use_gpu, verbose = verbose)
			trainx(θ̂, parameters, parameters, Z_train, Z_val, [1, 2, 5]; epochs = [10, 5, 3], use_gpu = use_gpu, verbose = verbose)
			Z_train = [simulate(parameters, m) for m ∈ [1, 2, 5]];
			Z_val   = [simulate(parameters, m) for m ∈ [1, 2, 5]];
			trainx(θ̂, parameters, parameters, Z_train, Z_val; epochs = [10, 5, 3], use_gpu = use_gpu, verbose = verbose)

			# Decided not to test the saving functions, because we can't always assume that we have write privledges
			# θ̂ = train(θ̂, parameters, parameters, m = 10, epochs = 5, savepath = "dummy123", use_gpu = use_gpu, verbose = verbose)
			# θ̂ = train(θ̂, parameters, parameters, m = 10, epochs = 5, savepath = "dummy123", use_gpu = use_gpu, verbose = verbose)
			# then rm dummy123 folder
		end

		# FIXME On the GPU, bug in this test
		# @testset "_runondevice" begin
		# 	θ̂₁ = θ̂(Z)
		# 	θ̂₂ = _runondevice(θ̂, Z, use_gpu)
		# 	@test size(θ̂₁) == size(θ̂₂)
		# 	@test θ̂₁ ≈ θ̂₂ # checked that this is fine by seeing if the following replacement fixes things: @test maximum(abs.(θ̂₁ .- θ̂₂)) < 0.0001
		# end

		@testset "assess" begin

			all_m = [10, 20, 30]

			# Method that does not require the user to provide data
			# assessment = assess([θ̂], parameters, m = all_m, use_gpu = use_gpu, verbose = verbose)
			# @test typeof(assessment)         == Assessment
			# @test typeof(assessment.θandθ̂)   == DataFrame
			# @test typeof(assessment.runtime) == DataFrame
			# risk(assessment)
			# risk(assessment, average_over_parameters = false)

			# Method that require the user to provide data: J == 1
			Z_test = [simulate(parameters, m) for m ∈ all_m]
			assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
			@test typeof(assessment)         == Assessment
			@test typeof(assessment.θandθ̂)   == DataFrame
			@test typeof(assessment.runtime) == DataFrame

			# Method that require the user to provide data: J == 5 > 1
			Z_test = [simulate(parameters, m, 5) for m ∈ all_m]
			assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
			@test typeof(assessment)         == Assessment
			@test typeof(assessment.θandθ̂)   == DataFrame
			@test typeof(assessment.runtime) == DataFrame

			# Test that estimators needing invariant model information can be used:
			assess([MLE], parameters, Z_test, verbose = verbose)
			assess([MLE], parameters, Z_test, verbose = verbose, ξ = ξ)
		end

		@testset "bootstrap" begin
			Z = Z[1] # bootstrap functions are designed for a single data set
			bootstrap(θ̂, Parameters(1, ξ), 50; use_gpu = use_gpu)
			bootstrap(θ̂, Z; use_gpu = use_gpu)
			bootstrap(θ̂, [Z]; use_gpu = use_gpu)
			@test_throws Exception bootstrap(θ̂, [Z, Z]; use_gpu = use_gpu)
			bootstrap(θ̂, Z, use_gpu = use_gpu, blocks = rand(1:2, size(Z)[end]))
			confidenceinterval(bootstrap(θ̂, Z; use_gpu = use_gpu))
		end
	end
end

@testset "PiecewiseEstimator" begin
	θ̂_piecewise = PiecewiseEstimator((θ̂_deepset, MLE), (30))
	Z = [array(n, 1, 10, T = Float32), array(n, 1, 50, T = Float32)]
	θ̂₁ = hcat(θ̂_deepset(Z[[1]]), MLE(Z[[2]]))
	θ̂₂ = θ̂_piecewise(Z)
	@test θ̂₁ ≈ θ̂₂
end
