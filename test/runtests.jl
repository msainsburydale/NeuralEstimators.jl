using NeuralEstimators
import NeuralEstimators: simulate
using NeuralEstimators: _getindices, _runondevice, _check_sizes, _extractθ
using CUDA
using DataFrames
using Distributions: Normal, Uniform, Product, cdf, logpdf, quantile
using Distances
using Flux
using Flux: batch, DataLoader, mae, mse
using GaussianRandomFields
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

oldstd = stdout
redirect_stderr(devnull)

if CUDA.functional()
	@info "Testing on both the CPU and the GPU... "
	CUDA.allowscalar(false)
	devices = (CPU = cpu, GPU = gpu)
else
	@info "The GPU is unavailable so we will test on the CPU only... "
	devices = (CPU = cpu,)
end

# ---- Stand-alone functions ----

# Start testing low-level functions, which form the base of the dependency tree.
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

		## Parameters stored as a simple matrix
		parameters = rand(3, K)
		indices = 2:3
		parameters_subset = subsetparameters(parameters, indices)
		@test size(parameters_subset) == (3, 2)
		@test parameters_subset       == parameters[:, indices]
	end
	@testset "containertype" begin
		a = rand(3, 4)
		T = Array
		@test containertype(a) == T
		@test containertype(typeof(a)) == T
		@test all([containertype(x) for x ∈ eachcol(a)] .== T)
	end

	@test isnothing(_check_sizes(1, 1))
end

@testset "maternclusterprocess" begin

	S = maternclusterprocess()
	@test size(S, 2) == 2

end

@testset "adjacencymatrix" begin

	n = 100
	d = 2
	S = rand(n, d)
	k = 5
	r = 0.3

	# Memory efficient constructors (avoids constructing the full distance matrix D)
	A₁ = adjacencymatrix(S, k)
	A₂ = adjacencymatrix(S, r)

	# Construct from full distance matrix D
	D = pairwise(Euclidean(), S, S, dims = 1)
	Ã₁ = adjacencymatrix(D, k)
	Ã₂ = adjacencymatrix(D, r)

	# Test that the matrices are the same irrespective of which method was used
	@test Ã₁ ≈ A₁
	@test Ã₂ ≈ A₂

	# Randomly selecting k nodes within a node's neighbourhood disc.
	seed!(1); A₃ = adjacencymatrix(S, k, r)
	@test A₃.n == A₃.m == n
	@test length(adjacencymatrix(S, k, 0.02).nzval) < k*n
	seed!(1); Ã₃ = adjacencymatrix(D, k, r)
	@test Ã₃ ≈ A₃

end


@testset "missingdata" begin

	# ---- removedata() ----
	d = 5     # dimension of each replicate
	n = 3     # number of observed elements of each replicate: must have n <= d
	m = 2000  # number of replicates
	Z = rand(d, m)

	removedata(Z, n)
	removedata(Z, d)
	removedata(Z, n; fixed_pattern = true)
	removedata(Z, n; contiguous_pattern = true)
	removedata(Z, n, variable_proportion = true)
	removedata(Z, n; contiguous_pattern = true, fixed_pattern = true)
	removedata(Z, n; contiguous_pattern = true, variable_proportion = true)

	# Passing the proportion of missingness
	p = rand(d)
	removedata(Z, p)
	removedata(Z, p; prevent_complete_missing = false)
	# Check that the probability of missingness is roughly correct:
	mapslices(x -> sum(ismissing.(x))/length(x), removedata(Z, p), dims = 2)
	# Check that none of the replicates contain 100% missing:
	@test !(d ∈ unique(mapslices(x -> sum(ismissing.(x)), removedata(Z, p), dims = 1)))


	# ---- encodedata() ----
	n = 16
	Z = rand(n)
	Z = removedata(Z, 0.25)
	UW = encodedata(Z);
	@test ndims(UW) == 3
	@test size(UW) == (n, 2, 1)

	Z = rand(n, n)
	Z = removedata(Z, 0.25)
	UW = encodedata(Z);
	@test ndims(UW) == 4
	@test size(UW) == (n, n, 2, 1)

	Z = rand(n, n, 1, 1)
	Z = removedata(Z, 0.25)
	UW = encodedata(Z);
	@test ndims(UW) == 4
	@test size(UW) == (n, n, 2, 1)

	m = 5
	Z = rand(n, n, 1, m)
	Z = removedata(Z, 0.25)
	UW = encodedata(Z);
	@test ndims(UW) == 4
	@test size(UW) == (n, n, 2, m)
end



@testset "WeightedGraphConv" begin
	# Construct a spatially-weighted adjacency matrix based on k-nearest neighbours
	# with k = 5, and convert to a graph with random (uncorrelated) dummy data:
	n = 100
	S = rand(n, 2)
	d = 1 # dimension of each observation (univariate data here)
	A = adjacencymatrix(S, 5)
	Z = GNNGraph(A, ndata = rand(d, n))
	layer = WeightedGraphConv(d => 16)
	show(devnull, layer)
	h = layer(Z) # convolved features
	@test size(h.ndata.x) == (16, n)
end

@testset "loss functions: $dvc" for dvc ∈ devices

	p = 3
	K = 10
	θ̂ = arrayn(p, K)       |> dvc
	θ = arrayn(p, K) * 0.9 |> dvc

	@testset "kpowerloss" begin
		@test kpowerloss(θ̂, θ, 2; safeorigin = false, joint=false) ≈ mse(θ̂, θ)
		@test kpowerloss(θ̂, θ, 1; safeorigin = false, joint=false) ≈ mae(θ̂, θ)
		@test kpowerloss(θ̂, θ, 1; safeorigin = true, joint=false) ≈ mae(θ̂, θ)
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

	n = 10
	S = array(n, 2, T = Float32)
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

	# Passing GaussianRandomFields:
	cov = CovarianceFunction(2, Matern(ρ[1], ν[1]))
	grf = GaussianRandomField(cov, GaussianRandomFields.Cholesky(), S)
	y₁  = simulategaussianprocess(L₁)
	y₂  = simulategaussianprocess(grf)
	y₃  = simulateschlather(grf)
	@test length(y₁) == length(y₂) == length(y₃)
	@test size(grf) == size(grf, 1) == n
end

# Testing the function simulate(): Univariate Gaussian model with unknown mean and standard deviation
p = 2
K = 10
m = 15
parameters = rand(p, K)
simulate(parameters, m) = [θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)]
simulate(parameters, m)
simulate(parameters, m, 2)
simulate(parameters, m) = ([θ[1] .+ θ[2] .* randn(1, m) for θ ∈ eachcol(parameters)], rand(2)) # Tuple (used for passing set-level covariate information)
simulate(parameters, m)
simulate(parameters, m, 2)


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
	L  = cholesky(Symmetric(Σ)).L
	@test gaussiandensity(y, L, logdensity = false) ≈ exp(gaussiandensity(y, L))
	@test gaussiandensity(y, Σ) ≈ gaussiandensity(y, L)
	@test gaussiandensity(hcat(y, y), Σ) ≈ 2 * gaussiandensity(y, L)
end


@testset "vectotri: $dvc" for dvc ∈ devices

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
	@test isa(gradient(() -> mae(θ̂(Z), θ), Flux.params(θ̂)), Zygote.Grads) # NB should probably use pullback() like I do in train().
end

@testset "Activation functions: $dvc" for dvc ∈ devices

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
	K = 100
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
		@test all(-1 .<= θ̂ .<= 1)

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

S = samplesize # Expert summary statistic used in DeepSetExpert
parameter_names = ["μ", "σ"]
struct Parameters <: ParameterConfigurations
	θ
end
Ω = Product([Normal(0, 1), Uniform(0.1, 1.5)])
ξ = (Ω = Ω, parameter_names = parameter_names)
K = 100
Parameters(K::Integer, ξ) = Parameters(rand(ξ.Ω, K))
parameters = Parameters(K, ξ)
show(devnull, parameters)
@test size(parameters) == (2, 100)
@test _extractθ(parameters.θ) == _extractθ(parameters)
p = length(parameter_names)

#### Array data

n = 1  # univariate data
simulatearray(parameters::Parameters, m) = [θ[1] .+ θ[2] .* randn(n, m) for θ ∈ eachcol(parameters.θ)]
function simulatorwithcovariates(parameters::Parameters, m)
	Z = simulatearray(parameters, m)
	x = [rand(qₓ) for _ ∈ eachindex(Z)]
	(Z, x)
end
function simulatorwithcovariates(parameters, m, J::Integer)
	v = [simulatorwithcovariates(parameters, m) for i ∈ 1:J]
	z = vcat([v[i][1] for i ∈ eachindex(v)]...)
	x = vcat([v[i][2] for i ∈ eachindex(v)]...)
	(z, x)
end
function simulatornocovariates(parameters::Parameters, m)
	simulatearray(parameters, m)
end
function simulatornocovariates(parameters, m, J::Integer)
	v = [simulatornocovariates(parameters, m) for i ∈ 1:J]
	vcat(v...)
end

# Traditional estimator that may be used for comparison
MLE(Z) = permutedims(hcat(mean.(Z), var.(Z)))
MLE(Z::Tuple) = MLE(Z[1])
MLE(Z, ξ) = MLE(Z) # the MLE doesn't need ξ, but we include it for testing

w  = 32 # width of each layer
qₓ = 2  # number of set-level covariates
m  = 10 # default sample size

@testset "Array data: $arch" for arch ∈ ["DeepSet" "DeepSetExpert"]
	@testset "$covar" for covar ∈ ["no set-level covariates" "set-level covariates"]
		q = w
		if covar == "set-level covariates"
			q = q + qₓ
			simulator = simulatorwithcovariates
		else
			simulator = simulatornocovariates
		end
		if arch == "DeepSet"
			ψ = Chain(Dense(n, w), Dense(w, w), Flux.flatten)
			ϕ = Chain(Dense(q, w), Dense(w, p))
			θ̂ = DeepSet(ψ, ϕ)
		elseif arch == "DeepSetExpert"
			ψ = Chain(Dense(n, w), Dense(w, w), Flux.flatten)
			ϕ = Chain(Dense(q + 1, w), Dense(w, p))
			θ̂ = DeepSetExpert(ψ, ϕ, S)
		end

		show(devnull, θ̂)

		@testset "$dvc" for dvc ∈ devices

			θ̂ = θ̂ |> dvc

			loss = Flux.Losses.mae |> dvc
			γ    = Flux.params(θ̂)  |> dvc
			θ    = array(p, K)     |> dvc

			Z = simulator(parameters, m) |> dvc
			@test size(θ̂(Z), 1) == p
			@test size(θ̂(Z), 2) == K
			@test isa(loss(θ̂(Z), θ), Number)

			# Test that we can update the neural-network parameters
			optimiser = ADAM(0.01)
			gradients = gradient(() -> loss(θ̂(Z), θ), γ)
			@test isa(gradients, Zygote.Grads)
			Flux.update!(optimiser, γ, gradients)

		    use_gpu = dvc == gpu
			@testset "train" begin

				# train: single estimator
				θ̂ = train(θ̂, Parameters, simulator, m = m, epochs = 2, use_gpu = use_gpu, verbose = verbose, ξ = ξ)
				θ̂ = train(θ̂, parameters, parameters, simulator, m = m, epochs = 2, use_gpu = use_gpu, verbose = verbose)
				θ̂ = train(θ̂, parameters, parameters, simulator, m = m, epochs = 2, epochs_per_Z_refresh = 2, use_gpu = use_gpu, verbose = verbose)
				θ̂ = train(θ̂, parameters, parameters, simulator, m = m, epochs = 2, epochs_per_Z_refresh = 1, simulate_just_in_time = true, use_gpu = use_gpu, verbose = verbose)
				Z_train = simulator(parameters, 2m);
				Z_val   = simulator(parameters, m);
				train(θ̂, parameters, parameters, Z_train, Z_val; epochs = 5, use_gpu = use_gpu, verbose = verbose)

				# trainx: Multiple estimators
				trainx(θ̂, Parameters, simulator, [1, 2, 5]; ξ = ξ, epochs = [3, 2, 1], use_gpu = use_gpu, verbose = verbose)
				trainx(θ̂, parameters, parameters, simulator, [1, 2, 5]; epochs = [3, 2, 1], use_gpu = use_gpu, verbose = verbose)
				trainx(θ̂, parameters, parameters, Z_train, Z_val, [1, 2, 5]; epochs = [3, 2, 1], use_gpu = use_gpu, verbose = verbose)
				Z_train = [simulator(parameters, m) for m ∈ [1, 2, 5]];
				Z_val   = [simulator(parameters, m) for m ∈ [1, 2, 5]];
				trainx(θ̂, parameters, parameters, Z_train, Z_val; epochs = [3, 2, 1], use_gpu = use_gpu, verbose = verbose)
			end

			@testset "assess" begin

				# J == 1
				Z_test = simulator(parameters, m)
				assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
				@test typeof(assessment)         == Assessment
				@test typeof(assessment.df)      == DataFrame
				@test typeof(assessment.runtime) == DataFrame

				@test typeof(merge(assessment, assessment)) == Assessment
				risk(assessment)
				risk(assessment, loss = (x, y) -> (x - y)^2)
				risk(assessment; average_over_parameters = false)
				risk(assessment; average_over_sample_sizes = false)
				risk(assessment; average_over_parameters = false, average_over_sample_sizes = false)

				bias(assessment)
				bias(assessment; average_over_parameters = false)
				bias(assessment; average_over_sample_sizes = false)
				bias(assessment; average_over_parameters = false, average_over_sample_sizes = false)

				rmse(assessment)
				rmse(assessment; average_over_parameters = false)
				rmse(assessment; average_over_sample_sizes = false)
				rmse(assessment; average_over_parameters = false, average_over_sample_sizes = false)

				# J == 5 > 1
				Z_test = simulator(parameters, m, 5)
				assessment = assess([θ̂], parameters, Z_test, use_gpu = use_gpu, verbose = verbose)
				@test typeof(assessment)         == Assessment
				@test typeof(assessment.df)      == DataFrame
				@test typeof(assessment.runtime) == DataFrame

				# Test that estimators needing invariant model information can be used:
				assess([MLE], parameters, Z_test, verbose = verbose)
				assess([MLE], parameters, Z_test, verbose = verbose, ξ = ξ)
			end


			@testset "bootstrap" begin

				# parametric bootstrap functions are designed for a single parameter configuration
				pars = Parameters(1, ξ)
				m = 20
				B = 400
				Z̃ = simulator(pars, m, B)
				size(bootstrap(θ̂, pars, Z̃; use_gpu = use_gpu)) == (p, K)
				size(bootstrap(θ̂, pars, simulator, m; use_gpu = use_gpu)) == (p, K)

				if covar == "no set-level covariates" # TODO non-parametric bootstrapping does not work for tuple data
					# non-parametric bootstrap is designed for a single parameter configuration and a single data set
					if typeof(Z̃) <: Tuple
						Z = ([Z̃[1][1]], [Z̃[2][1]]) # NB not ideal that we need to still store these a vectors, given that the estimator doesn't require it
					else
						Z = Z̃[1]
					end
					Z = Z |> dvc

					@test size(bootstrap(θ̂, Z; use_gpu = use_gpu)) == (p, B)
					@test size(bootstrap(θ̂, [Z]; use_gpu = use_gpu)) == (p, B)
					@test_throws Exception bootstrap(θ̂, [Z, Z]; use_gpu = use_gpu)
					@test size(bootstrap(θ̂, Z, use_gpu = use_gpu, blocks = rand(1:2, size(Z)[end]))) == (p, B)

					# interval
					θ̃ = bootstrap(θ̂, pars, simulator, m; use_gpu = use_gpu)
					@test size(interval(θ̃)) == (p, 2)
				end
			end
		end
	end
end


#### Graph data

@testset "GNN" begin

	# Propagation module
    d = 1      # dimension of response variable
    nh = 32    # dimension of node feature vectors
    propagation = GNNChain(GraphConv(d => nh), GraphConv(nh => nh), GraphConv(nh => nh))

    # Readout module
    nt = 32   # dimension of the summary vector for each node
    no = 128  # dimension of the final summary vector for each graph
    readout = UniversalPool(
    	Chain(Dense(nh, nt), Dense(nt, nt)),
    	Chain(Dense(nt, nt), Dense(nt, no))
    	)
	show(devnull, readout)

    # Mapping module
    p = 3     # number of parameters in the statistical model
    w = 64    # width of layers used for the outer network ϕ
    ϕ = Chain(Dense(no, w, relu), Dense(w, w, relu), Dense(w, p))

    # Construct the estimator
    θ̂ = GNN(propagation, readout, ϕ)
	show(devnull, θ̂)

    # Apply the estimator to:
    # 1. a single graph,
    # 2. a single graph with sub-graphs (corresponding to independent replicates), and
    # 3. a vector of graphs (corresponding to multiple spatial data sets, each
    #    possibly containing independent replicates).
    g₁ = rand_graph(11, 30, ndata=rand(d, 11))
    g₂ = rand_graph(13, 40, ndata=rand(d, 13))
    g₃ = batch([g₁, g₂])
    θ̂(g₁)
    θ̂(g₃)
    θ̂([g₁, g₂, g₃])

	@test size(θ̂(g₁)) == (p, 1)
	@test size(θ̂(g₃)) == (p, 1)
	@test size(θ̂([g₁, g₂, g₃])) == (p, 3)
end

# ---- Estimators ----

@testset "initialise_estimator" begin
	p = 2
	initialise_estimator(p, architecture = "DNN")
	initialise_estimator(p, architecture = "GNN")
	initialise_estimator(p, architecture = "CNN", kernel_size = [(10, 10), (5, 5), (3, 3)])

	@test typeof(initialise_estimator(p, architecture = "DNN", estimator_type = "interval")) <: IntervalEstimator
	@test typeof(initialise_estimator(p, architecture = "GNN", estimator_type = "interval")) <: IntervalEstimator
	@test typeof(initialise_estimator(p, architecture = "CNN", kernel_size = [(10, 10), (5, 5), (3, 3)], estimator_type = "interval")) <: IntervalEstimator

	@test_throws Exception initialise_estimator(0, architecture = "DNN")
	@test_throws Exception initialise_estimator(p, d = 0, architecture = "DNN")
	@test_throws Exception initialise_estimator(p, architecture = "CNN")
	@test_throws Exception initialise_estimator(p, architecture = "CNN", kernel_size = [(10, 10), (5, 5)])
end

@testset "PiecewiseEstimator" begin
	@test_throws Exception PiecewiseEstimator((MLE, MLE), (30, 50))
	@test_throws Exception PiecewiseEstimator((MLE, MLE, MLE), (50, 30))
	θ̂_piecewise = PiecewiseEstimator((MLE, MLE), (30))
	show(devnull, θ̂_piecewise)
	Z = [array(n, 1, 10, T = Float32), array(n, 1, 50, T = Float32)]
	θ̂₁ = hcat(MLE(Z[[1]]), MLE(Z[[2]]))
	θ̂₂ = θ̂_piecewise(Z)
	@test θ̂₁ ≈ θ̂₂
end


@testset "IntervalEstimator" begin
	# Generate some toy data and a basic architecture
	d = 2  # bivariate data
	m = 64 # number of independent replicates
	Z = rand(d, m)
	parameter_names = ["ρ", "σ", "τ"]
	p = length(parameter_names)
	w = 8  # width of each layer
	arch = initialise_estimator(p, architecture = "DNN", d = d, width = 8)

	# IntervalEstimator
	estimator = IntervalEstimator(arch)
	estimator = IntervalEstimator(arch, arch)
	θ̂ = estimator(Z)
	@test size(θ̂) == (2p, 1)
	@test all(θ̂[1:p] .< θ̂[(p+1):end])
	ci = interval(estimator, Z)
	ci = interval(estimator, Z, parameter_names = parameter_names)
	@test size(ci[1]) == (p, 2)

	# IntervalEstimator with a compact prior
	min_supp = [25, 0.5, -pi/2]
	max_supp = [500, 2.5, 0]
	estimator = IntervalEstimator(arch, min_supp, max_supp)
	estimator = IntervalEstimator(arch, arch, min_supp, max_supp)
	θ̂ = estimator(Z)
	@test size(θ̂) == (2p, 1)
	@test all(θ̂[1:p] .< θ̂[(p+1):end])
	@test all(min_supp .< θ̂[1:p] .< max_supp)
	@test all(min_supp .< θ̂[p+1:end] .< max_supp)
	ci = interval(estimator, Z)
	ci = interval(estimator, Z, parameter_names = parameter_names)
	@test size(ci[1]) == (p, 2)
end


@testset "NeuralEM" begin

	# Set the prior distribution
	Ω = (τ = Uniform(0.01, 0.3), ρ = Uniform(0.01, 0.3))

	p = length(Ω)    # number of parameters in the statistical model

	# Set the (gridded) spatial domain
	points = range(0.0, 1.0, 16)
	S = expandgrid(points, points)

	# Model information that is constant (and which will be passed into later functions)
	ξ = (
		Ω = Ω,
		ν = 1.0, 	# fixed smoothness
		S = S,
		D = pairwise(Euclidean(), S, S, dims = 1),
		p = p
	)

	# Sampler from the prior
	struct GPParameters <: ParameterConfigurations
		θ
		cholesky_factors
	end

	function GPParameters(K::Integer, ξ)

		# Sample parameters from the prior
		Ω = ξ.Ω
		τ = rand(Ω.τ, K)
		ρ = rand(Ω.ρ, K)

		# Compute Cholesky factors
		cholesky_factors = maternchols(ξ.D, ρ, ξ.ν)

		# Concatenate into a matrix
		θ = permutedims(hcat(τ, ρ))
		θ = Float32.(θ)

		GPParameters(θ, cholesky_factors)
	end

	function simulate(parameters, m::Integer)

		K = size(parameters, 2)
		τ = parameters.θ[1, :]

		Z = map(1:K) do k
			L = parameters.cholesky_factors[:, :, k]
			z = simulategaussianprocess(L, m)
			z = z + τ[k] * randn(size(z)...)
			z = Float32.(z)
			z = reshape(z, 16, 16, 1, :)
			z
		end

		return Z
	end

	function simulateconditional(Z::M, θ, ξ; nsims::Integer = 1) where {M <: AbstractMatrix{Union{Missing, T}}} where T

		# Save the original dimensions
		dims = size(Z)

		# Convert to vector
		Z = vec(Z)

		# Compute the indices of the observed and missing data
		I₁ = findall(z -> !ismissing(z), Z) # indices of observed data
		I₂ = findall(z -> ismissing(z), Z)  # indices of missing data
		n₁ = length(I₁)
		n₂ = length(I₂)

		# Extract the observed data and drop Missing from the eltype of the container
		Z₁ = Z[I₁]
		Z₁ = [Z₁...]

		# Distance matrices needed for covariance matrices
		D   = ξ.D # distance matrix for all locations in the grid
		D₂₂ = D[I₂, I₂]
		D₁₁ = D[I₁, I₁]
		D₁₂ = D[I₁, I₂]

		# Extract the parameters from θ
		τ = θ[1]
		ρ = θ[2]

		# Compute covariance matrices
		ν = ξ.ν
		Σ₂₂ = matern.(UpperTriangular(D₂₂), ρ, ν); Σ₂₂[diagind(Σ₂₂)] .+= τ^2
		Σ₁₁ = matern.(UpperTriangular(D₁₁), ρ, ν); Σ₁₁[diagind(Σ₁₁)] .+= τ^2
		Σ₁₂ = matern.(D₁₂, ρ, ν)

		# Compute the Cholesky factor of Σ₁₁ and solve the lower triangular system
		L₁₁ = cholesky(Symmetric(Σ₁₁)).L
		x = L₁₁ \ Σ₁₂

		# Conditional covariance matrix, cov(Z₂ ∣ Z₁, θ),  and its Cholesky factor
		Σ = Σ₂₂ - x'x
		L = cholesky(Symmetric(Σ)).L

		# Conditonal mean, E(Z₂ ∣ Z₁, θ)
		y = L₁₁ \ Z₁
		μ = x'y

		# Simulate from the distribution Z₂ ∣ Z₁, θ ∼ N(μ, Σ)
		z = randn(n₂, nsims)
		Z₂ = μ .+ L * z

		# Combine the observed and missing data to form the complete data
		Z = map(1:nsims) do l
			z = Vector{T}(undef, n₁ + n₂)
			z[I₁] = Z₁
			z[I₂] = Z₂[:, l]
			z
		end
		Z = stackarrays(Z, merge = false)

		# Convert Z to an array with appropriate dimensions
		Z = reshape(Z, dims..., 1, nsims)

		return Z
	end

	θ = GPParameters(1, ξ)
	Z = simulate(θ, 1)[1][:, :]		# simulate a single gridded field
	Z = removedata(Z, 0.25)			# remove 25% of the data

	neuralMAPestimator = initialise_estimator(p, architecture = "CNN", kernel_size = [(10, 10), (5, 5), (3, 3)], activation_output = exp)
	neuralem = NeuralEM(simulateconditional, neuralMAPestimator)
	θ₀ = mean.([Ω...]) 						# initial estimate, the prior mean
	H = 5
	θ̂   = neuralem(Z, θ₀, ξ = ξ, nsims = H)
	θ̂2  = neuralem([Z, Z], θ₀, ξ = ξ, nsims = H)

	@test size(θ̂)  == (2, 1)
	@test size(θ̂2) == (2, 2)

	## Test initial-value handling
	@test_throws Exception neuralem(Z)
	@test_throws Exception neuralem([Z, Z])
	neuralem = NeuralEM(simulateconditional, neuralMAPestimator, θ₀)
	neuralem(Z, ξ = ξ, nsims = H)
	neuralem([Z, Z], ξ = ξ, nsims = H)
end
