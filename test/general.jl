using NeuralEstimators
using NeuralEstimators: _runondevice, _check_sizes, _extractθ, rowwisenorm, triangularnumber, forward, inverse, _logdensity
using NeuralEstimators: ActNorm, Permutation, AffineCouplingBlock, CouplingLayer
using CUDA
using DataFrames
using Distances
using Flux
using Flux: batch, DataLoader, mae, mse, numobs, getobs
using GraphNeuralNetworks
using LinearAlgebra
using MLUtils
using Optimisers
using Random: seed!
using SparseArrays: nnz
using SpecialFunctions: gamma
using Statistics
using Statistics: mean, sum
using Test
if CUDA.functional()
    @info "Testing on both the CPU and the GPU... "
    CUDA.allowscalar(false)
    devices = (CPU = cpu_device(), GPU = gpu_device())
else
    @info "The GPU is unavailable so we will test on the CPU only... "
    devices = (CPU = cpu_device(),)
end
verbose = false

array(size...; T = Float32) = T.(reshape(1:prod(size), size...) ./ prod(size))
arrayn(size...; T = Float32) = array(size..., T = T) .- mean(array(size..., T = T))

function testbackprop(l, z, dvc)
    l = l |> dvc
    z = z |> dvc
    y = l(z)

    pars = deepcopy(trainables(l))
    optimiser = Optimisers.setup(Optimisers.Adam(), l)
    ∇ = Flux.gradient(l -> mae(l(z), similar(y)), l)
    Optimisers.update!(optimiser, l, ∇[1])
    @test trainables(l) != pars

    pars = deepcopy(trainables(l))
    ls, ∇ = Flux.withgradient(l -> mae(l(z), similar(y)), l)
    Optimisers.update!(optimiser, l, ∇[1])
    @test trainables(l) != pars
end

@testset "Utility functions" begin
    @testset "expandgrid" begin
        @test expandgrid(1:2, 0:3) == [1 0; 2 0; 1 1; 2 1; 1 2; 2 2; 1 3; 2 3]
        @test expandgrid(1:2, 1:2) == expandgrid(2)
    end
    @testset "stackarrays" begin
        # Vector containing arrays of the same size:
        A = array(2, 3, 4);
        v = [A, A];
        N = ndims(A);
        @test stackarrays(v) == cat(v..., dims = N)
        @test stackarrays(v, merge = false) == cat(v..., dims = N + 1)

        # Vector containing arrays with differing final dimension size:
        A₁ = array(2, 3, 4);
        A₂ = array(2, 3, 5);
        v = [A₁, A₂];
        @test stackarrays(v) == cat(v..., dims = N)
    end
    @testset "containertype" begin
        a = rand(3, 4)
        T = Array
        @test containertype(a) == T
        @test containertype(typeof(a)) == T
        @test all([containertype(x) for x ∈ eachcol(a)] .== T)
    end

    @testset "DataSet" begin
        K = 10
        Z = [randn(2, 5) for _ = 1:K]  # K data sets, each 2×5
        S = randn(3, K)                  # 3 expert summaries per data set

        # ---- Construction ----
        @testset "construction" begin
            ds = DataSet(Z, S)
            @test ds.Z === Z
            @test ds.S === S

            # Without expert summaries
            ds_no_s = DataSet(Z)
            @test ds_no_s.Z === Z
            @test isnothing(ds_no_s.S)

            # Assert mismatch between numobs(Z) and size(S, 2)
            @test_throws AssertionError DataSet(Z, randn(3, K + 1))
        end

        # ---- MLUtils interface ----
        @testset "numobs and getobs" begin
            ds = DataSet(Z, S)
            @test numobs(ds) == K

            # getobs: single index
            ds1 = getobs(ds, 1)
            @test numobs(ds1) == 1
            @test ds1.S == S[:, 1:1]

            # getobs: range
            ds_sub = getobs(ds, 1:3)
            @test numobs(ds_sub) == 3
            @test ds_sub.S == S[:, 1:3]
        end

        # ---- getindex ----
        @testset "getindex" begin
            ds = DataSet(Z, S)
            ds1 = ds[1]
            @test numobs(ds1) == 1

            ds_sub = ds[1:3]
            @test numobs(ds_sub) == 3
        end

        # ---- utility methods ----
        @testset "numberreplicates" begin
            ds = DataSet(Z, S)
            @test numberreplicates(ds) == numberreplicates(Z)
        end

        @testset "subsetreplicates" begin
            ds = DataSet(Z, S)
            ds_sub = subsetreplicates(ds, 1:3)
            @test numobs(ds_sub) == K
            @test all(numberreplicates(ds_sub) .== 3)
        end

        # ---- f32 ----
        @testset "f32" begin
            ds = DataSet(Z)
            ds32 = f32(ds)
            @test eltype(ds32.Z[1]) == Float32
            ds = DataSet(Z, S)
            ds32 = f32(ds)
            @test eltype(ds32.Z[1]) == Float32
            @test eltype(ds32.S) == Float32
        end

        # ---- forward pass ----
        @testset "forward pass" begin
            n, num_summaries = 2, 4
            ψ = Chain(Dense(n, 8, relu))
            ϕ = Chain(Dense(8, num_summaries))
            network = DeepSet(ψ, ϕ)
            estimator = PointEstimator(network)

            ds = DataSet(Z, S) |> f32  # S has 3 rows, network outputs 4 → vcat gives 7
            t = estimator(ds)
            @test size(t, 1) == num_summaries + size(S, 1)  # 4 + 3 = 7
            @test size(t, 2) == K

            # Without expert summaries: output size unchanged
            ds_no_s = DataSet(Z) |> f32 
            t_no_s = estimator(ds_no_s)
            @test size(t_no_s, 1) == num_summaries
            @test size(t_no_s, 2) == K
        end
    end

    @test isnothing(_check_sizes(1, 1))

    @testset "maternclusterprocess" begin
        S = maternclusterprocess()
        @test size(S, 2) == 2
        S = maternclusterprocess(unit_bounding_box = true)
        @test size(S, 2) == 2
    end

    @testset "adjacencymatrix" begin
        n = 100
        d = 2
        S = rand(Float32, n, d)
        k = 5
        r = 0.3

        # Memory efficient constructors (avoids constructing the full distance matrix D)
        A = A₁ = adjacencymatrix(S, k)
        A₂ = adjacencymatrix(S, r)
        @test eltype(A₁) == Float32
        @test eltype(A₂) == Float32
        @test eltype(A) == Float32

        # Construct from full distance matrix D
        D = pairwise(Euclidean(), S, S, dims = 1)
        Ã₁ = adjacencymatrix(D, k)
        Ã₂ = adjacencymatrix(D, r)
        @test eltype(Ã₁) == Float32
        @test eltype(Ã₂) == Float32

        # Test that the matrices are the same irrespective of which method was used
        @test Ã₁ ≈ A₁
        @test Ã₂ ≈ A₂

        # Randomly selecting k nodes within a node's neighbourhood disc
        seed!(1);
        A₃ = adjacencymatrix(S, k, r)
        @test A₃.n == A₃.m == n
        @test length(adjacencymatrix(S, k, 0.02).nzval) < k*n
        seed!(1);
        Ã₃ = adjacencymatrix(D, k, r)
        @test Ã₃ ≈ A₃

        # Test that the number of neighbours is correct
        f(A) = collect(mapslices(nnz, A; dims = 1))
        @test all(f(adjacencymatrix(S, k)) .== k)
        @test all(1 .<= f(adjacencymatrix(S, r, k; random = true)) .<= k)
        @test all(1 .<= f(adjacencymatrix(S, r, k; random = false)) .<= k+1)
        @test all(f(adjacencymatrix(S, 2.0, k; random = true)) .== k)
        @test all(f(adjacencymatrix(S, 2.0, k; random = false)) .== k+1)

        # Gridded locations (useful for checking functionality in the event of ties)
        pts = range(0, 1, length = 10)
        S = expandgrid(pts, pts)
        @test all(f(adjacencymatrix(S, k)) .== k)
        @test all(1 .<= f(adjacencymatrix(S, r, k; random = true)) .<= k)
        @test all(1 .<= f(adjacencymatrix(S, r, k; random = false)) .<= k+1)
        @test all(f(adjacencymatrix(S, 2.0, k; random = true)) .== k)
        @test all(f(adjacencymatrix(S, 2.0, k; random = false)) .== k+1)

        # Check that k > n doesn't cause an error
        n = 3
        d = 2
        S = rand(n, d)
        adjacencymatrix(S, k)
        adjacencymatrix(S, r, k)
        D = pairwise(Euclidean(), S, S, dims = 1)
        adjacencymatrix(D, k)
        adjacencymatrix(D, r, k)
    end

    @testset "spatialgraph" begin
        # Number of replicates, and spatial dimension
        m = 5  # number of replicates
        d = 2  # spatial dimension

        # Spatial locations fixed for all replicates
        n = 1000
        S = rand(n, d)
        Z = rand(n, m)
        g = spatialgraph(S)
        @test g.num_nodes == n
        g = spatialgraph(g, Z)
        g = spatialgraph(S, Z)

        # Spatial locations varying between replicates
        n = rand(500:1000, m)
        S = rand.(n, d)
        Z = rand.(n)
        g = spatialgraph(S)
        @test g.num_nodes == sum(n)
        g = spatialgraph(g, Z)
        g = spatialgraph(S, Z)

        # Mutlivariate processes: spatial locations fixed for all replicates
        q = 2 # bivariate spatial process
        n = 1000
        S = rand(n, d)
        Z = rand(q, n, m)
        g = spatialgraph(S)
        @test g.num_nodes == n
        g = spatialgraph(g, Z)
        g = spatialgraph(S, Z)

        # Mutlivariate processes: spatial locations varying between replicates
        n = rand(500:1000, m)
        S = rand.(n, d)
        Z = rand.(q, n)
        g = spatialgraph(S)
        @test g.num_nodes == sum(n)
        g = spatialgraph(g, Z)
        g = spatialgraph(S, Z)
    end

    # @testset "Missing data" begin

    #     # removedata()
    #     d = 5     # dimension of each replicate
    #     n = 3     # number of observed elements of each replicate: must have n <= d
    #     m = 2000  # number of replicates
    #     p = rand(d)

    #     Z = rand(d)
    #     removedata(Z, n)
    #     removedata(Z, p[1])
    #     removedata(Z, p)

    #     Z = rand(d, m)
    #     removedata(Z, n)
    #     removedata(Z, d)
    #     removedata(Z, n; fixed_pattern = true)
    #     removedata(Z, n; contiguous_pattern = true)
    #     removedata(Z, n; contiguous_pattern = true, fixed_pattern = true)
    #     # removedata(Z, p) #TODO errors
    #     # removedata(Z, p; prevent_complete_missing = false) # TODO errors
    #     # Check that the probability of missingness is roughly correct:
    #     mapslices(x -> sum(ismissing.(x))/length(x), removedata(Z, p), dims = 2)
    #     # Check that none of the replicates contain 100% missing:
    #     @test !(d ∈ unique(mapslices(x -> sum(ismissing.(x)), removedata(Z, p), dims = 1)))

    #     # encodedata()
    #     n = 16
    #     Z = rand(n)
    #     Z = removedata(Z, 0.25)
    #     UW = encodedata(Z)
    #     @test ndims(UW) == 1
    #     @test size(UW) == (2n,)

    #     Z = rand(n, n)
    #     Z = removedata(Z, 0.25)
    #     UW = encodedata(Z)
    #     @test ndims(UW) == 2
    #     @test size(UW) == (2n, n)

    #     Z = rand(n, n, 3, 5)
    #     Z = removedata(Z, 0.25)
    #     UW = encodedata(Z)
    #     @test ndims(UW) == 4
    #     @test size(UW) == (n, n, 6, 5)
    # end

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
end

@testset "User-defined summary statistics: $dvc" for dvc ∈ devices
    # 5 replicates of a 3-dimensional vector
    d, m = 3, 5
    z = rand(d, m) |> dvc
    @test samplesize(z) == m
    @test length(samplecovariance(z)) == triangularnumber(d)
    @test length(samplecorrelation(z)) == triangularnumber(d-1)

    # vector input
    z = rand(d) |> dvc
    @test samplesize(z) == 1
    @test_throws Exception samplecovariance(z)
    @test_throws Exception samplecorrelation(z)

    # neighbourhood variogram
    θ = 0.1                                 # true range parameter
    n = 100                                 # number of spatial locations
    S = rand(n, 2)                          # spatial locations
    D = pairwise(Euclidean(), S, dims = 1)  # distance matrix
    Σ = exp.(-D ./ θ)                       # covariance matrix
    L = cholesky(Symmetric(Σ)).L            # Cholesky factor
    m = 5                                   # number of replicates
    Z = L * randn(n, m)                     # simulated data
    r = 0.15                                # radius of neighbourhood set
    g = spatialgraph(S, Z, r = r) |> dvc
    nv = NeighbourhoodVariogram(r, 10) |> dvc
    nv(g)
    @test length(nv(g)) == 10
    @test all(nv(g) .>= 0)
end

@testset "Loss functions: $dvc" for dvc ∈ devices
    d = 3
    K = 10
    θ̂ = arrayn(d, K) |> dvc
    θ = arrayn(d, K) * 0.9 |> dvc

    @testset "kpowerloss" begin
        @test kpowerloss(θ̂, θ, 2; safeorigin = false, joint = false) ≈ mse(θ̂, θ)
        @test kpowerloss(θ̂, θ, 1; safeorigin = false, joint = false) ≈ mae(θ̂, θ)
        @test kpowerloss(θ̂, θ, 1; safeorigin = true, joint = false) ≈ mae(θ̂, θ)
        @test kpowerloss(θ̂, θ, 0.1) >= 0
    end

    @testset "quantileloss" begin
        q = 0.5
        @test quantileloss(θ̂, θ, q) >= 0
        @test quantileloss(θ̂, θ, q) ≈ mae(θ̂, θ)/2

        q = [0.025, 0.975]
        @test_throws Exception quantileloss(θ̂, θ, q)
        θ̂ = arrayn(length(q) * d, K) |> dvc
        @test quantileloss(θ̂, θ, q) >= 0
    end

    @testset "intervalscore" begin
        α = 0.025
        θ̂ = arrayn(2d, K) |> dvc
        @test intervalscore(θ̂, θ, α) >= 0
    end
end

@testset "Approximate distributions: $dvc" for dvc ∈ devices
    for d = 1:5
        dstar = 2d
        K = 10
        θ = rand32(d, K) |> dvc
        TZ = rand32(dstar, K) |> dvc

        @testset "ActNorm: $args" for args in (d, (3.0 * ones(d), 2.0 * ones(d)))
            an = ActNorm(args...) |> dvc
            U, log_det_J = forward(an, θ)
            @test size(U) == (d, K)
            @test length(log_det_J) == 1
            @test log_det_J == sum(log.(abs.(an.scale)))
            X = inverse(an, U)
            @test size(X) == (d, K)
            @test θ ≈ X
        end

        @testset "Permutation" begin
            perm = Permutation(d) |> dvc
            U = forward(perm, θ)
            @test size(U) == (d, K)
            X = inverse(perm, U)
            @test size(X) == (d, K)
            @test θ == X
        end

        @testset "AffineCouplingBlock" begin
            d₁ = div(d, 2)
            d₂ = div(d, 2) + (d % 2 != 0 ? 1 : 0)
            layer = AffineCouplingBlock(d₁, dstar, d₂; backend = Flux) |> dvc
            θ1 = θ[1:d₁, :]
            θ2 = θ[(d₁ + 1):end, :]
            U2, log_det_J2 = forward(layer, θ2, θ1, TZ)
            @test size(U2) == (d₂, K)
            @test size(log_det_J2) == (1, K)
            X2 = inverse(layer, θ1, U2, TZ)
            @test size(X2) == (d₂, K)
            @test θ2 ≈ X2
        end

        @testset "CouplingLayer" begin
            layer = CouplingLayer(d, dstar; backend = Flux) |> dvc
            U, log_det_J = forward(layer, θ, TZ)
            @test size(U) == (d, K)
            @test size(log_det_J) == (1, K)
            X = inverse(layer, U, TZ)
            @test size(X) == (d, K)
            @test θ ≈ X
        end

        @testset "NormalisingFlow" begin
            flow = NormalisingFlow(d, dstar) |> dvc

            # forward pass
            U, log_det_J = forward(flow, θ, TZ)
            @test size(U) == (d, K)
            @test size(log_det_J) == (1, K)

            # backward/inverse pass
            X = inverse(flow, U, TZ)
            @test size(X) == (d, K)
            @test maximum(abs.(θ - X)) < 1e-4

            # density evaluation (employs forward pass, used during training)
            dens = _logdensity(flow, θ, TZ)
            @test size(dens) == (1, K)

            # sampling (employs backward/inverse pass, used during inference)
            N = 100
            samples = sampleposterior(flow, TZ, N; device = dvc)
            @test length(samples) == K
            @test size(samples[1]) == (d, N)
        end
    end
end

@testset "Layers: $dvc" for dvc ∈ devices
    @testset "ResidualBlock" begin
        n = 10
        ch = 4
        z = rand32(n, n, 1, 1)
        l = ResidualBlock((3, 3), 1 => ch)
        l = l |> dvc
        z = z |> dvc
        y = l(z)
        @test size(y) == (n, n, ch, 1)
        testbackprop(l, z, dvc)
    end

    @testset "SpatialGraphConv" begin
        n = 100
        m = 5
        S = rand(n, 2)
        Z = rand(n, m)
        g = spatialgraph(S, Z)
        ch = 10
        l = SpatialGraphConv(1 => ch)
        l = l |> dvc
        g = g |> dvc
        y = l(g)
        @test size(y.ndata.Z) == (ch, m, n)
        # Back propagation
        pars = deepcopy(trainables(l))
        optimiser = Optimisers.setup(Optimisers.Adam(), l)
        ∇ = Flux.gradient(l -> mae(l(g).ndata.Z, similar(y.ndata.Z)), l)
        Optimisers.update!(optimiser, l, ∇[1])
        @test trainables(l) != pars

        # GNNSummary
        propagation = Chain(SpatialGraphConv(1 => ch), SpatialGraphConv(ch => ch))
        readout = GlobalPool(mean)
        ψ = GNNSummary(propagation, readout)
        ψ = ψ |> dvc
        g = g |> dvc
        y = ψ(g)
        @test size(y) == (ch, m)
        testbackprop(ψ, g, dvc)
    end

    @testset "Spatial weight functions: $Weights" for Weights ∈ [IndicatorWeights, KernelWeights]
        n = 30
        h = rand(1, n)
        n_bins = 10
        w = Weights(1, n_bins)
        w = w |> dvc
        h = h |> dvc
        y = w(h)
        @test size(y) == (n_bins, n)
        # Spatial weight functions do not have trainable parameters: avoid warnings by testing back prop with a simple chain
        l = Chain(w, Dense(n_bins, 10))
        testbackprop(l, h, dvc)
    end
end

@testset "Output layers: $dvc" for dvc ∈ devices
    function testbackprop(l, dvc, p::Integer, K::Integer, d::Integer)
        Z = arrayn(d, K) |> dvc
        θ = arrayn(p, K) |> dvc
        θ̂ = Chain(Dense(d, p), l) |> dvc
        Flux.gradient(θ̂ -> mae(θ̂(Z), θ), θ̂)
    end

    @testset "Compress" begin
        Compress(1, 2)
        p = 3
        K = 10
        a = Float32.([0.1, 4, 2])
        b = Float32.([0.9, 9, 3])
        l = Compress(a, b) |> dvc
        θ = arrayn(p, K) |> dvc
        θ̂ = l(θ)
        @test size(θ̂) == (p, K)
        @test typeof(θ̂) == typeof(θ)
        @test all([all(a .< cpu(x) .< b) for x ∈ eachcol(θ̂)])
        testbackprop(l, dvc, p, K, 20)
    end

    @testset "CovarianceMatrix" begin
        d = 4
        K = 100
        p = d*(d+1)÷2
        θ = arrayn(p, K) |> dvc

        l = CovarianceMatrix(d) |> dvc
        θ̂ = l(θ)
        @test_throws Exception l(vcat(θ, θ))
        @test size(θ̂) == (p, K)
        @test length(l(θ[:, 1])) == p
        @test typeof(θ̂) == typeof(θ)

        Σ = [Symmetric(cpu(vectotril(x)), :L) for x ∈ eachcol(θ̂)]
        Σ = convert.(Matrix, Σ);
        @test all(isposdef.(Σ))

        L = l(θ, true)
        L = [LowerTriangular(cpu(vectotril(x))) for x ∈ eachcol(L)]
        @test all(Σ .≈ L .* permutedims.(L))

        testbackprop(l, dvc, p, K, d)
    end

    A = rand(5, 4)
    @test rowwisenorm(A) == mapslices(norm, A; dims = 2)

    @testset "CorrelationMatrix" begin
        d = 4
        K = 100
        p = d*(d-1)÷2
        θ = arrayn(p, K) |> dvc
        l = CorrelationMatrix(d) |> dvc
        θ̂ = l(θ)
        @test_throws Exception l(vcat(θ, θ))
        @test size(θ̂) == (p, K)
        @test length(l(θ[:, 1])) == p
        @test typeof(θ̂) == typeof(θ)
        @test all(-1 .<= θ̂ .<= 1)

        R = map(eachcol(l(θ))) do x
            R = Symmetric(cpu(vectotril(x; strict = true)), :L)
            R[diagind(R)] .= 1
            R
        end
        @test all(isposdef.(R))

        L = l(θ, true)
        L = map(eachcol(L)) do x
            L = LowerTriangular(cpu(vectotril(x, strict = true)))
            L[diagind(L)] .= sqrt.(1 .- rowwisenorm(L) .^ 2)
            L
        end
        @test all(R .≈ L .* permutedims.(L))

        testbackprop(l, dvc, p, K, d)
    end
end

@testset "DeepSet: $dvc" for dvc ∈ devices
    # Test
    # - with and without expert summary statistics
    # - with and without set-level inputs
    # - common data formats
    n = 10     # dimension of each data replicate
    M = (3, 4) # number of replicates in each data set
    w = 32     # width of each hidden layer
    d = 5      # output dimension
    dₜ = 16    # dimension of neural summary statistic
    for S in (nothing, samplesize)
        for dₓ in (0, 2) # dimension of set-level inputs
            for data in ("unstructured", "grid", "graph")
                dₛ = isnothing(S) ? 0 : 1 # dimension of expert summary statistic
                if data == "unstructured"
                    Z = [rand32(n, m) for m ∈ M]
                    ψ = Chain(Dense(n, w), Dense(w, dₜ), MLUtils.flatten)
                elseif data == "grid"
                    Z = [rand32(10, 10, 1, m) for m ∈ M]
                    ψ = Chain(Conv((5, 5), 1 => dₜ), GlobalMeanPool(), MLUtils.flatten)
                elseif data == "graph"
                    Z = [spatialgraph(rand(100, 2), rand(100, m)) for m ∈ (4, 4)] #TODO doesn't work for variable number of replicates i.e., m ∈ M; also, this can break when n is taken to be small like n=5 (run it many times and you will eventually see ERROR: AssertionError: DataStore: data[e] has 1 observations, but n = 0)
                    propagation = Chain(SpatialGraphConv(1 => 16), SpatialGraphConv(16 => dₜ))
                    readout = GlobalPool(mean)
                    ψ = GNNSummary(propagation, readout)
                end
                ϕ = Chain(Dense(dₜ + dₛ + dₓ, w, relu), Dense(w, d)) # outer network
                ds = DeepSet(ψ, ϕ; S = S)
                show(devnull, ds)
                if dₓ > 0
                    X = [rand32(dₓ) for _ ∈ eachindex(Z)]
                    input = (Z, X)
                else
                    input = Z
                end
                # Forward evaluation
                y = ds(input)
                @test size(y) == (d, length(M))
                # Basic back propagation
                testbackprop(ds, input, dvc)
            end
        end
    end
end

# ---- Estimators ----

struct Parameters{A} <: AbstractParameterSet
    θ::A
end

d, m = 2, 5  # dimension of θ and number of replicates
sampler(K, d = nothing) = NamedMatrix(μ = randn(Float32, K), σ = rand(Float32, K)) # NB dummy argument d just to check (keyword)  arguments can be passed
simulator(θ::AbstractVector, m::Integer) = θ["μ"] .+ θ["σ"] .* sort(randn(Float32, m))
simulator(θ::AbstractMatrix, m::Integer) = reduce(hcat, simulator.(eachcol(θ), m))

K = 35
θ = sampler(K)
Z = simulator(θ, m)

@testset "PointEstimator" begin
    network = Chain(Dense(m, 16, gelu), Dense(16, d))
    estimator = PointEstimator(network)
    show(devnull, estimator)

    @testset "$dvc" for dvc ∈ devices

        use_gpu = dvc == gpu
        
        # Forward pass
        @test size(estimate(estimator, Z)) == (d, K)

        @testset "train" begin
            testbackprop(estimator, Z, dvc)
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,))
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), savepath = "testing-path")
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), simulate_just_in_time = true)
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), freeze_summary_network = true)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, savepath = "testing-path")
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 4, epochs_per_Z_refresh = 2, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 3, epochs_per_Z_refresh = 1, simulate_just_in_time = true, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 4, epochs_per_Z_refresh = 2, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
            Z_train = Z_val = simulator(θ, m);
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose, savepath = "testing-path")
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose)
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
        end

        @testset "assess" begin
            Z_test = simulator(θ, m)
            assessment = assess([estimator], θ, Z_test, use_gpu = use_gpu, verbose = verbose)
            assessment = assess(estimator, θ, Z_test, use_gpu = use_gpu)

            @test typeof(assessment) == Assessment
            @test typeof(assessment.estimates) == DataFrame
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
        end

        @testset "bootstrap" begin
            # parametric bootstrap functions are designed for a single parameter configuration
            B = 40
            parameters = sampler(1)            
            Z_sims = reduce(hcat, [simulator(parameters, m) for _ in 1:B])
            @test size(bootstrap(estimator, parameters, Z_sims; use_gpu = use_gpu)) == (d, B)
            @test size(bootstrap(estimator, parameters, simulator, m; B=B, use_gpu = use_gpu)) == (d, B)
        end
    end
end

@testset "IntervalEstimator" begin
    num_summaries = 3d
    summary_network = Chain(Dense(m, 64, relu), Dense(64, 64, relu), Dense(64, num_summaries))
    min_supp = [-1.0, -1.0]
    max_supp = [1.0, 1.0]
    c = Compress(min_supp, max_supp)
    estimator = IntervalEstimator(summary_network, d; num_summaries = num_summaries, c = c)
    show(devnull, estimator)

    # Forward pass
    θ̂ = estimator(Z)
    ci = interval(estimator, Z)
    @test size(θ̂) == (2d, K)
    @test all(min_supp .< θ̂[1:d, :] .< θ̂[(d + 1):end, :] .< max_supp)
    @test all([size(c) == (d, 2) for c in ci])

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose)

    # Assessment
    assessment = assess(estimator, θ, Z)
    coverage(assessment)
    coverage(assessment; average_over_parameters = true)
    coverage(assessment; average_over_sample_sizes = false)
    coverage(assessment; average_over_parameters = true, average_over_sample_sizes = false)

    intervalscore(assessment)
    intervalscore(assessment; average_over_parameters = true)
    intervalscore(assessment; average_over_sample_sizes = false)
    intervalscore(assessment; average_over_parameters = true, average_over_sample_sizes = false)
end

@testset "QuantileEstimator: marginal" begin
    num_summaries = 3d
    summary_network = Chain(Dense(m, 64, relu), Dense(64, 64, relu), Dense(64, num_summaries))
    probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    estimator = QuantileEstimator(summary_network, d; num_summaries = num_summaries, probs = probs)

    # Forward pass
    θ̂ = estimator(Z)
    @test size(θ̂) == (length(probs) * d, K)

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose)

    # Assessment
    assessment = assess(estimator, θ, Z)

    # Inference
    z = simulator(sampler(1), m)
    estimate(estimator, Z)
    quantiles(estimator, Z)
end

@testset "QuantileEstimator: full conditionals" begin

    # Initialise estimators respectively targetting quantiles of μ∣Z,σ and σ∣Z,μ
    num_summaries = 3d
    summary_network = Chain(Dense(m, 64, relu), Dense(64, 64, relu), Dense(64, num_summaries))
    τ = [0.05, 0.25, 0.5, 0.75, 0.95]
    q₁ = QuantileEstimator(summary_network, d; num_summaries = num_summaries, probs = τ, i = 1)
    q₂ = QuantileEstimator(summary_network, d; num_summaries = num_summaries, probs = τ, i = 2)

    # Forward pass
    θ₋ᵢ = 0.5f0
    θ̂ = q₁(Z, θ₋ᵢ)
    @test size(θ̂) == (length(τ), K)

    # Training
    q₁ = train(q₁, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose)
    q₂ = train(q₂, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose)

    # Inference: Estimate quantiles of μ∣Z,σ with σ = 0.5
    z = simulator(sampler(1), m)
    θ₋ᵢ = [0.5f0;]
    estimate(q₁, (z, θ₋ᵢ))
    quantiles(q₁, (z, θ₋ᵢ))
end

@testset "RatioEstimator" begin

    num_summaries = 3d
    summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries)

    # Forward pass
    r = estimator(Z, θ)
    @test size(r) == (1, K)

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = false)

    # Inference (grid-based)
    grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space
    z = getobs(Z, 1:1)
    logratio(estimator, z; grid = grid)                # log of likelihood-to-evidence ratios
    samples = sampleposterior(estimator, z; grid = grid)         # posterior sample
    @test size(samples) == (2, 1000)

    # Assessment (grid-based)
    assessment = assess(estimator, θ, Z; grid = grid)
end

@testset "PosteriorEstimator" begin
    for approxdist in [NormalisingFlow, GaussianMixture]
        num_summaries = 3d
        summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
        q = approxdist(d, num_summaries)
        estimator = PosteriorEstimator(summary_network, d; num_summaries = num_summaries, q = approxdist) # convenience constructor
        estimator = PosteriorEstimator(summary_network, q)
        estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = false)
        @test numdistributionalparams(estimator) == numdistributionalparams(q)
        samples = sampleposterior(estimator, Z) # posterior draws
        @test all([size(s) == (d, 1000) for s in samples])
        posteriormean(estimator, Z)   # point estimate
        posteriormedian(estimator, Z) # point estimate
        posteriorquantile(estimator, Z, [0.1, 0.5]) # quantiles
        assessment = assess(estimator, θ, Z)
    end
end

# ---- Wrappers and helper functions for NeuralEstimators ----

@testset "Ensemble: $dvc" for dvc ∈ devices
    # Architecture of each ensemble component
    function initestimator()
        network = Chain(Dense(m, 16, gelu), Dense(16, 2))
        PointEstimator(network)
    end

    # Initialise ensemble
    J = 2 # ensemble size
    estimators = [initestimator() for j = 1:J]
    ensemble = Ensemble(estimators)
    ensemble[1]
    @test length(ensemble) == J

    # Training
    ensemble = train(ensemble, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose, use_gpu = dvc == gpu)

    # Assessment
    assessment = assess(ensemble, θ, Z)
    rmse(assessment)

    # Apply to data
    estimate(ensemble, Z)
end

@testset "PiecewiseEstimator" begin

    n = 2    # bivariate data
    d = 3    # dimension of parameter vector
    w = 128  # width of each hidden layer
    ψ₁ = Chain(Dense(n, w, relu), Dense(w, w, relu));
    ϕ₁ = Chain(Dense(w, w, relu), Dense(w, d));
    θ̂₁ = PointEstimator(DeepSet(ψ₁, ϕ₁))
    ψ₂ = Chain(Dense(n, w, relu), Dense(w, w, relu));
    ϕ₂ = Chain(Dense(w, w, relu), Dense(w, d));
    θ̂₂ = PointEstimator(DeepSet(ψ₂, ϕ₂))
    θ̂ = PiecewiseEstimator([θ̂₁, θ̂₂], 30)
    Z = [rand32(n, m) for m ∈ (10, 50)]
    θ̂(Z)
    #estimate(θ̂, Z) #NB last time I checked, breaks on the GPU

    @test_throws Exception PiecewiseEstimator((θ̂₁, θ̂₂), (30, 50))
    @test_throws Exception PiecewiseEstimator((θ̂₁, θ̂₂, θ̂₁), (50, 30))
    θ̂_piecewise = PiecewiseEstimator((θ̂₁, θ̂₂), (30))
    show(devnull, θ̂_piecewise)
    est1 = hcat(θ̂₁(Z[[1]]), θ̂₂(Z[[2]]))
    est2 = θ̂_piecewise(Z)
    @test est1 ≈ est2
end


@testset "EM" begin
    d = 2    # number of parameters in the statistical model

    # Set the (gridded) spatial domain
    points = range(0.0, 1.0, 16)
    S = expandgrid(points, points)

    # Model information that is constant (and which will be passed into later functions)
    ξ = (
        ν = 1.0, # fixed smoothness
        S = S,
        D = pairwise(Euclidean(), S, S, dims = 1),
        d = d
    )

    struct GPParameters <: AbstractParameterSet
        θ::Any
        cholesky_factors::Any
    end

    function GPParameters(K::Integer, ξ)

        # Sample parameters from the prior
        τ = 0.3 * rand(K)
        ρ = 0.3 * rand(K)

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
            z = simulategaussian(L, m)
            z = z + τ[k] * randn(size(z)...)
            z = Float32.(z)
            z = reshape(z, 16, 16, 1, :)
            z
        end

        return Z
    end

    function simulateconditional(Z::M, θ; nsims::Integer = 1, ξ) where {M <: AbstractMatrix{Union{Missing, T}}} where {T}

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
        D = ξ.D # distance matrix for all locations in the grid
        D₂₂ = D[I₂, I₂]
        D₁₁ = D[I₁, I₁]
        D₁₂ = D[I₁, I₂]

        # Extract the parameters from θ
        τ = θ[1]
        ρ = θ[2]

        # Compute covariance matrices
        ν = ξ.ν
        Σ₂₂ = matern.(UpperTriangular(D₂₂), ρ, ν);
        Σ₂₂[diagind(Σ₂₂)] .+= τ^2
        Σ₁₁ = matern.(UpperTriangular(D₁₁), ρ, ν);
        Σ₁₁[diagind(Σ₁₁)] .+= τ^2
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
    Z = simulate(θ, 1)[1][:, :]# simulate a single gridded field
    Z = removedata(Z, 0.25)# remove 25% of the data

    # Construct neural MAP estimator
    ψ = Chain(
        Conv((10, 10), 1 => 16,  relu),
        Conv((5, 5),  16 => 32,  relu),
        Conv((3, 3),  32 => 64, relu),
        Flux.flatten
    )
    ϕ = Chain(Dense(64, 256, relu), Dense(256, d, exp))
    network = DeepSet(ψ, ϕ)
    neuralMAPestimator = PointEstimator(network)

    # EM object
    neuralem = EM(simulateconditional, neuralMAPestimator)
    θ₀ = [0.15, 0.15]# initial estimate, the prior mean
    H = 5
    θ̂ = neuralem(Z, θ₀, nsims = H, ξ = ξ).estimate
    θ̂2 = neuralem([Z, Z], θ₀, nsims = H, ξ = ξ)

    @test size(θ̂) == (2, 1)
    @test size(θ̂2) == (2, 2)

    ## Test initial-value handling
    @test_throws Exception neuralem(Z)
    @test_throws Exception neuralem([Z, Z])
    neuralem = EM(simulateconditional, neuralMAPestimator, θ₀)
    neuralem(Z, nsims = H, ξ = ξ)
    neuralem([Z, Z], nsims = H, ξ = ξ)

    ## Test edge cases (no missingness and complete missingness)
    Z = simulate(θ, 1)[1]# simulate a single gridded field
    # @test_logs (:warn,) neuralem(Z, θ₀, ξ = ξ, nsims = H)
    @test_throws Exception neuralem(Z, θ₀, ξ = ξ, nsims = H)
    Z₁ = removedata(Z₁, 1.0)
    @test_throws Exception neuralem(Z₁, θ₀, nsims = H, ξ = ξ)
end


# ---- Misc. ----

@testset "Model-specific functions" begin
    @testset "Simulation" begin
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
        @test eltype(simulategaussian(L₁, m)) == Float32

        ## Potts model
        β = 0.7
        complete_grid = simulatepotts(n, n, 2, 0.99)      # simulate marginally from the Ising model
        complete_grid = simulatepotts(n, n, 2, β)         # simulate marginally from the Ising model
        @test size(complete_grid) == (n, n)
        @test length(unique(complete_grid)) == 2
        incomplete_grid = removedata(complete_grid, 0.1)     # remove 10% of the pixels at random
        imputed_grid = simulatepotts(incomplete_grid, β)  # conditionally simulate over missing pixels
        observed_idx = findall(!ismissing, incomplete_grid)
        @test incomplete_grid[observed_idx] == imputed_grid[observed_idx]
    end

    @testset "Densities" begin

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
            @test abs(finitedifference(z₁, z₂, ψ) - schlatherbivariatedensity(z₁, z₂, ψ; logdensity = false)) < 0.0001
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
        L = cholesky(Symmetric(Σ)).L
        @test gaussiandensity(y, L, logdensity = false) ≈ exp(gaussiandensity(y, L))
        @test gaussiandensity(y, Σ) ≈ gaussiandensity(y, L)
        @test gaussiandensity(hcat(y, y), Σ) ≈ 2 * gaussiandensity(y, L)
    end
end
