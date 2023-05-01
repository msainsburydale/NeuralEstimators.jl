# NB train() breaks when updated from Flux@v0.13.9 to Flux@v0.13.11

using NeuralEstimators
using NeuralEstimators: _getindices, _runondevice
import NeuralEstimators: simulate
using CUDA
using DataFrames
using Distributions: Normal, cdf, logpdf, quantile
using Flux
using Flux: DataLoader, mae, mse
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
arrayn(size...; T = Float64) = array(size..., T = T) .- mean(array(size..., T = T))
verbose = false # verbose used in NeuralEstimators code (not @testset)

if CUDA.functional()
	@info "Testing on both the CPU and the GPU... "
	CUDA.allowscalar(false)
	devices = (CPU = cpu, GPU = gpu)
else
	@info "The GPU is unavailable so we'll test on the CPU only... "
	devices = (CPU = cpu,)
end

# ---- Stand-alone functions ----

# Start testing with low-level functions, which form the base of the
# dependency tree.
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
	@testset "containertype" begin
		a = rand(3, 4)
		T = Array
		@test containertype(a) == T
		@test containertype(typeof(a)) == T
		@test all([containertype(x) for x ∈ eachcol(a)] .== T)
	end
end

@testset verbose = true "loss functions: $dvc" for dvc ∈ devices

	p = 3
	K = 10
	θ̂ = arrayn(p, K)       |> dvc
	θ = arrayn(p, K) * 0.9 |> dvc

	@testset "kpowerloss" begin
		@test kpowerloss(θ̂, θ, 2; safeorigin = false) ≈ mse(θ̂, θ)
		@test kpowerloss(θ̂, θ, 1; safeorigin = false) ≈ mae(θ̂, θ)
		@test kpowerloss(θ̂, θ, 1; safeorigin = true) ≈ mae(θ̂, θ)
		@test kpowerloss(θ̂, θ, 0.1) >= 0
	end

	@testset "quantileloss" begin
		q = 0.5
		@test quantileloss(θ̂, θ, q) >= 0
		@test quantileloss(θ̂, θ, q) ≈ mae(θ̂, θ)/2

		q = [0.025, 0.975]
		@test_throws Exception quantileloss(θ̂, θ, q)
		θ̂ = arrayn(length(q) * p, K) |> dvc
		@test quantileloss(θ̂, θ, q) >= 0
	end

	@testset "intervalscore" begin
		α = 0.025
		θ̂ = arrayn(2p, K) |> dvc
		@test intervalscore(θ̂, θ, α) >= 0
	end

end

@testset "simulate" begin

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


@testset verbose = true "vectotri: $dvc" for dvc ∈ devices

	d = 4
	n = d*(d+1)÷2

	v = arrayn(n) |> dvc
	L = vectotril(v)
	@test istril(L)
	@test all([cpu(v)[i] ∈ cpu(L) for i ∈ 1:n])
	@test containertype(L) == containertype(v)
	U = vectotriu(v)
	@test istriu(U)
	@test all([cpu(v)[i] ∈ cpu(U) for i ∈ 1:n])
	@test containertype(U) == containertype(v)

	# testing that it works for views of arrays
	V = arrayn(n, 2) |> dvc
	L = [vectotril(v) for v ∈ eachcol(V)]
	@test all(istril.(L))
	@test all(containertype.(L) .== containertype(v))

	# strict variants
	n = d*(d-1)÷2
	v = arrayn(n) |> dvc
	L = vectotril(v; strict = true)
	@test istril(L)
	@test all(L[diagind(L)] .== 0)
	@test all([cpu(v)[i] ∈ cpu(L) for i ∈ 1:n])
	@test containertype(L) == containertype(v)
	U = vectotriu(v; strict = true)
	@test istriu(U)
	@test all(U[diagind(U)] .== 0)
	@test all([cpu(v)[i] ∈ cpu(U) for i ∈ 1:n])
	@test containertype(U) == containertype(v)

end


# ---- Activation functions ----

function testbackprop(l, dvc, p::Integer, K::Integer, d::Integer)
	Z = arrayn(d, K) |> dvc
	θ = arrayn(p, K) |> dvc
	θ̂ = Chain(Dense(d, p), l) |> dvc
	@test isa(gradient(() -> mae(θ̂(Z), θ), Flux.params(θ̂)), Zygote.Grads) # TODO should probably use pullback() like I do in train(). Do this after updating the training functions in line with the recent versions of Flux.
end

@testset verbose = true "Activation functions: $dvc" for dvc ∈ devices

	@testset "Compress" begin
		p = 3
		K = 10
		a = [0.1, 4, 2]
		b = [0.9, 9, 3]
		l = Compress(a, b) |> dvc
		θ = arrayn(p, K)   |> dvc
		θ̂ = l(θ)
		@test size(θ̂) == (p, K)
		@test typeof(θ̂) == typeof(θ)
		@test all([all(a .< cpu(x) .< b) for x ∈ eachcol(θ̂)])
		testbackprop(l, dvc, p, K, 20)
	end

	d = 4
	K = 50
	p = d*(d+1)÷2
	θ = arrayn(p, K) |> dvc

	@testset "CholeskyCovariance" begin
		l = CholeskyCovariance(d) |> dvc
		θ̂ = l(θ)
		@test size(θ̂) == (p, K)
		@test all(θ̂[l.diag_idx, :] .> 0)
		@test typeof(θ̂) == typeof(θ)
		testbackprop(l, dvc, p, K, d)
	end

	@testset "CovarianceMatrix" begin
		l = CovarianceMatrix(d) |> dvc
		θ̂ = l(θ)
		@test size(θ̂) == (p, K)
		@test all(θ̂[l.choleskyparameters.diag_idx, :] .> 0)
		@test typeof(θ̂) == typeof(θ)
		testbackprop(l, dvc, p, K, d)

		Σ = [Symmetric(cpu(vectotril(y)), :L) for y ∈ eachcol(θ̂)]
		Σ = convert.(Matrix, Σ);
		@test all(isposdef.(Σ))
	end

	@testset "CorrelationMatrix" begin
		p = d*(d-1)÷2
		θ = arrayn(p, K) |> dvc
		l = CorrelationMatrix(d) |> dvc
		θ̂ = l(θ)
		@test size(θ̂) == (p, K)
		@test typeof(θ̂) == typeof(θ)

		R = map(eachcol(l(θ))) do y
			R = Symmetric(cpu(vectotril(y; strict=true)), :L)
			R[diagind(R)] .= 1
			R
		end
		@test all(isposdef.(R))

		testbackprop(l, dvc, p, K, d)
	end

	@testset "SplitApply" begin
		p₁ = 2          # number of non-covariance matrix parameters
		p₂ = d*(d+1)÷2  # number of covariance matrix parameters
		p = p₁ + p₂

		a = [0.1, 4]
		b = [0.9, 9]
		l₁ = Compress(a, b)
		l₂ = CovarianceMatrix(d)
		l = SplitApply([l₁, l₂], [1:p₁, p₁+1:p])

		l = l            |> dvc
		θ = arrayn(p, K) |> dvc
		θ̂ = l(θ)
		@test size(θ̂) == (p, K)
		@test typeof(θ̂) == typeof(θ)
		testbackprop(l, dvc, p, K, 20)
	end

end


# ---- Architectures ----

# Expert summary statistic that may be used in DeepSetExpert
S = samplesize

#TODO # Multiple data sets with set-level covariates
# function (d::DeepSet)(tup::Tup) where {Tup <: Tuple{V₁, V₂}} where {V₁ <: AbstractVector{A}, V₂ <: AbstractVector{B}} where {A, B <:



# DeepSet and DeepSetExpert with GraphPropagatePool module
#TODO need to test this on the GPU
@testset "GraphPropagatePool" begin
	n₁, n₂ = 11, 27
	m₁, m₂ = 30, 50
	d = 1
	g₁ = rand_graph(n₁, m₁, ndata = array(d, n₁, T = Float32))
	g₂ = rand_graph(n₂, m₂, ndata = array(d, n₂, T = Float32))
	g  = Flux.batch([g₁, g₂])

	# g is a single large GNNGraph containing subgraphs
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

	# Full estimator couched in Deep Set framework
	w = 32
	p = 3

	@testset verbose = true "GraphPropagatePool: $ds" for ds ∈ ["DeepSet", "DeepSetExpert"]
		ψ = GraphPropagatePool(graphtograph, meanpool)
		ϕ = Chain(Dense(o, w, relu), Dense(w, p))

		# TODO Add if statement for DeepSet or DeepSetExpert
		est = DeepSet(ψ, ϕ)

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

end


# DeepSet and DeepSetExpert with array input data

struct Parameters <: ParameterConfigurations
	θ
	σ
end
parameter_names = ["μ"]
ξ = (Ω = Normal(0, 0.5), σ = 1, parameter_names)
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
ϕₛ = Chain(Dense(w + 1, w), Dense(w, p))
θ̂_deepsetexpert = DeepSetExpert(ψ, ϕₛ, S)
dₓ= 2
estimators = (DeepSet = θ̂_deepset, DeepSetExpert = θ̂_deepsetexpert)


@testset verbose = true "$key" for key ∈ keys(estimators)

	θ̂ = estimators[key]

	@testset "$ky" for ky ∈ keys(devices)

		device = devices[ky]
		θ̂ = θ̂ |> device

		loss = Flux.Losses.mae |> device
		γ    = Flux.params(θ̂)  |> device
		θ    = array(p, K)     |> device

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

			m = 20

			# J == 1
			Z_test = simulate(parameters, m)
			assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
			@test typeof(assessment)         == Assessment
			@test typeof(assessment.df)      == DataFrame
			@test typeof(assessment.runtime) == DataFrame

			# J == 5 > 1
			Z_test = simulate(parameters, m, 5)
			assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
			@test typeof(assessment)         == Assessment
			@test typeof(assessment.df)      == DataFrame
			@test typeof(assessment.runtime) == DataFrame
			@test typeof(merge(assessment, assessment)) == Assessment
			risk(assessment)
			risk(assessment; average_over_parameters = false)
			risk(assessment; average_over_sample_sizes = false)
			risk(assessment; average_over_parameters = false, average_over_sample_sizes = false)

			# Test that estimators needing invariant model information can be used:
			assess([MLE], parameters, Z_test, verbose = verbose)
			assess([MLE], parameters, Z_test, verbose = verbose, ξ = ξ)
		end

		@testset "bootstrap" begin

			# parametric bootstrap functions are designed for a single parameter configuration
			parameters = Parameters(1, ξ)
			m = 20
			B = 400
			Z̃ = simulate(parameters, m, B)
			bootstrap(θ̂, parameters, Z̃; use_gpu = use_gpu)
			bootstrap(θ̂, parameters, m; use_gpu = use_gpu)

			# non-parametric bootstrap is designed for a single parameter configuration and a single data set
			Z = Z̃[1] |> device
			bootstrap(θ̂, Z; use_gpu = use_gpu)
			bootstrap(θ̂, [Z]; use_gpu = use_gpu)
			@test_throws Exception bootstrap(θ̂, [Z, Z]; use_gpu = use_gpu)
			bootstrap(θ̂, Z, use_gpu = use_gpu, blocks = rand(1:2, size(Z)[end]))

			# interval
			θ̃ = bootstrap(θ̂, parameters, m; use_gpu = use_gpu)
			@test size(interval(θ̃)) == (p, 2)
			# @test size(interval(θ̃, θ̂(Z), type = "basic")) == (p, 2) #FIXME broken on the GPU
			@test_throws Exception interval(θ̃, type = "basic")
			@test_throws Exception interval(θ̃, type = "zxcvbnm")

			# Coverage
			ci = interval(θ̃)
			θ  = parameters.θ
			cov = coverage([ci], θ)
			@test length(cov) == p
			@assert all(0 .<= cov .<= 1)
		end
	end
end

# TODO this should be moved into the above loop
@testset "set-level covariates" begin
	n = 10
	p = 4

	w = 32 # width of each layer
	ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));

	Z₁ = rand(n, 3);                  # single set of 3 realisations
	Z₂ = [rand(n, m) for m ∈ (3, 3)]; # two sets each containing 3 realisations
	Z₃ = [rand(n, m) for m ∈ (3, 4)]; # two sets containing 3 and 4 realisations
	θ = rand(p, 2)

	dₓ = 2
	x₁ = rand(dₓ)
	x₂ = [rand(dₓ) for _ ∈ eachindex(Z₂)]

	# regular deepset
	ϕ = Chain(Dense(w + dₓ, w, relu), Dense(w, p));
	θ̂ = DeepSet(ψ, ϕ)
	θ̂((Z₁, x₁))
	θ̂((Z₂, x₂))
	θ̂((Z₃, x₂))

	# Test that training works:
	train(θ̂, θ, θ, (Z₂, x₂), (Z₂, x₂), epochs = 3, batchsize = 2, verbose = verbose)
	train(θ̂, θ, θ, (broadcast(z -> hcat(z, z), Z₂), x₂), (Z₂, x₂), epochs = 3, batchsize = 2, verbose = verbose)
	train(θ̂, θ, θ, (Z₃, x₂), (Z₃, x₂), epochs = 3, batchsize = 2, verbose = verbose)
end

@testset "PiecewiseEstimator" begin
	@test_throws Exception PiecewiseEstimator((θ̂_deepset, MLE), (30, 50))
	@test_throws Exception PiecewiseEstimator((θ̂_deepset, MLE, MLE), (50, 30))
	θ̂_piecewise = PiecewiseEstimator((θ̂_deepset, MLE), (30))
	Z = [array(n, 1, 10, T = Float32), array(n, 1, 50, T = Float32)]
	θ̂₁ = hcat(θ̂_deepset(Z[[1]]), MLE(Z[[2]]))
	θ̂₂ = θ̂_piecewise(Z)
	@test θ̂₁ ≈ θ̂₂
end


@testset "IntervalEstimator" begin
	# Generate some toy data
	n = 2  # bivariate data
	m = 10 # number of independent replicates
	Z = rand(n, m)

	# Create an architecture
	p = 3  # parameters in the model
	w = 8  # width of each layer
	ψ = Chain(Dense(n, w, relu), Dense(w, w, relu));
	ϕ = Chain(Dense(w, w, relu), Dense(w, p));
	architecture = DeepSet(ψ, ϕ)

	# Initialise the interval estimator
	estimator = IntervalEstimator(architecture)

	# Apply the interval estimator
	estimator(Z)
	ci = interval(estimator, Z, parameter_names = ["ρ", "σ", "τ"]) #TODO  suppress these warnings
	@test size(ci[1]) == (p, 2) # FIXME why does this method of interval return a vector??
end
