using NeuralEstimators
using NeuralEstimators: _check_sizes, _extractθ, rowwisenorm, triangularnumber, forward, inverse, _logdensity
using NeuralEstimators: _TrainDisplay, _status!, _finishline!, _epoch_status, _bar!, _refresh_status!, _clear_refresh!, _fit_line
using NeuralEstimators: ActNorm, Permutation, AffineCouplingBlock, CouplingLayer
using NeuralEstimators: ChebPlan, ChebPlan_on, chebnodes, chebfit, chebintegrate, chebintegral, chebdefinite, chebint_ab, chebintmatrix
using NeuralEstimators: chebval_ab, chebval_ab_batched, cheblogq, invert_cdf_batched, chebsample, default_bisection_iters
using AdvancedHMC, ForwardDiff, LogDensityProblems # loads the AdvancedHMC extension (NUTS sampling for RatioEstimator)
using CairoMakie
using CUDA
using DataFrames
using Distances
using FFTW
using Flux
using Flux: batch, DataLoader, mae, mse, numobs, getobs, f32
using GraphNeuralNetworks
using LinearAlgebra
using MLUtils
using Optimisers
using Random: seed!
using SparseArrays: nnz, rowvals, nzrange, nonzeros
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

    # NB the target must be well defined: similar(y) is uninitialised memory, which on the GPU can
    # be a recycled buffer equal to the output, in which case the gradient of the MAE is zero
    target = randn(Float32, size(y)) |> dvc

    pars = deepcopy(trainables(l))
    optimiser = Optimisers.setup(Optimisers.Adam(), l)
    ∇ = Flux.gradient(l -> mae(l(z), target), l)
    Optimisers.update!(optimiser, l, ∇[1])
    @test trainables(l) != pars

    pars = deepcopy(trainables(l))
    ls, ∇ = Flux.withgradient(l -> mae(l(z), target), l)
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
        A = array(2, 3, 4)
        v = [A, A]
        N = ndims(A)
        @test stackarrays(v) == cat(v..., dims = N)
        @test stackarrays(v, merge = false) == cat(v..., dims = N + 1)

        # Vector containing arrays with differing final dimension size:
        A₁ = array(2, 3, 4)
        A₂ = array(2, 3, 5)
        v = [A₁, A₂]
        @test stackarrays(v) == cat(v..., dims = N)

        # Many arrays (the arrays must not be splatted into cat(), which is slow)
        v = [array(3, m) for m ∈ rand(2:9, 256)]
        @test stackarrays(v) == reduce(hcat, v)
        v = [array(3, 4) for _ ∈ 1:256]
        @test stackarrays(v) == reduce(hcat, v)

        # Differentiability
        v = [rand32(3, m) for m ∈ (4, 5, 6)]
        @test all(Flux.gradient(v -> sum(abs2, stackarrays(v)), v)[1] .≈ map(x -> 2x, v))
    end
    @testset "containertype" begin
        a = rand(3, 4)
        T = Array
        @test containertype(a) == T
        @test containertype(typeof(a)) == T
        @test all([containertype(x) for x ∈ eachcol(a)] .== T)
    end

    @testset "DataAndSummaries" begin
        K = 10
        Z = [randn(2, 5) for _ = 1:K]  # K data sets, each 2×5
        S = randn(3, K)                  # 3 expert summaries per data set

        # ---- Construction ----
        @testset "construction" begin
            ds = DataAndSummaries(Z, S)
            @test ds.Z === Z
            @test ds.S === S

            # The expert summary statistics are required
            @test_throws MethodError DataAndSummaries(Z)

            # Assert mismatch between numobs(Z) and size(S, 2)
            @test_throws AssertionError DataAndSummaries(Z, randn(3, K + 1))
        end

        # ---- MLUtils interface ----
        @testset "numobs and getobs" begin
            ds = DataAndSummaries(Z, S)
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
            ds = DataAndSummaries(Z, S)
            ds1 = ds[1]
            @test numobs(ds1) == 1

            ds_sub = ds[1:3]
            @test numobs(ds_sub) == 3
        end

        # ---- utility methods ----
        @testset "numberreplicates" begin
            ds = DataAndSummaries(Z, S)
            @test numberreplicates(ds) == numberreplicates(Z)
        end

        @testset "subsetreplicates" begin
            ds = DataAndSummaries(Z, S)
            ds_sub = subsetreplicates(ds, 1:3)
            @test numobs(ds_sub) == K
            @test all(numberreplicates(ds_sub) .== 3)
        end

        # ---- f32 ----
        @testset "f32" begin
            ds = DataAndSummaries(Z, S)
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

            ds = DataAndSummaries(Z, S) |> f32  # S has 3 rows, network outputs 4 → vcat gives 7
            t = estimator(ds)
            @test size(t, 1) == num_summaries + size(S, 1)  # 4 + 3 = 7
            @test size(t, 2) == K
        end
    end

    @testset "PackedReplicates" begin
        n = 2
        Z_eq = [array(n, 4) for _ = 1:3]
        Z_var = [array(n, m) for m in (3, 5, 4)]

        @testset "construction" begin
            for Z in (Z_eq, Z_var)
                P = PackedReplicates(Z)
                @test P.data == stackarrays(Z)
                @test P.sample_sizes == [size(z, ndims(z)) for z in Z]
                @test size(P.data) == (n, sum(P.sample_sizes))
                @test numobs(P) == length(Z)
            end

            P = PackedReplicates(Z_var)
            show(devnull, P)
            @test_throws ArgumentError PackedReplicates(P.data, [1, 1])
            @test_throws ArgumentError PackedReplicates(AbstractArray[])
            @test_throws ArgumentError PackedReplicates(Z_var; max_sample_size = 4)
        end

        @testset "padding" begin
            P = PackedReplicates(Z_var; max_sample_size = 8)
            show(devnull, P)
            @test P.sample_sizes == [3, 5, 4]
            @test size(P.data) == (n, 8 * 3)
            @test size(P.mask) == (8, 3)
            @test P.data[:, 1:3] == Z_var[1]
            @test all(P.data[:, 4:8] .== 0)
            @test P.mask[1:3, 1] == ones(Float32, 3)
            @test all(iszero, P.mask[4:8, 1])
            @test all(isone, P.mask[1:5, 2])
            @test all(iszero, P.mask[6:8, 2])

            P1 = getobs(P, 1)
            @test numobs(P1) == 1
            @test P1.sample_sizes == [3]
            @test size(P1.data) == (n, 8)
            @test P1.data[:, 1:3] == Z_var[1]
            @test size(P1.mask) == (8, 1)

            P_sub = getobs(P, 1:2)
            @test P_sub.sample_sizes == [3, 5]
            @test size(P_sub.data) == (n, 16)
            @test P_sub.data[:, 1:3] == Z_var[1]
            @test P_sub.data[:, 9:13] == Z_var[2]

            P_shuf = getobs(P, [3, 1])
            @test P_shuf.sample_sizes == [4, 3]
            @test P_shuf.data[:, 1:4] == Z_var[3]
            @test P_shuf.data[:, 9:11] == Z_var[1]

            P12 = getobs(P, 1:2)
            P3 = getobs(P, 3)
            Pj = joinobs(P12, P3)
            @test Pj.sample_sizes == P.sample_sizes
            @test Pj.data == P.data
            @test Pj.mask == P.mask
            @test_throws ArgumentError joinobs(P, PackedReplicates(Z_var))

            P_rep = subsetreplicates(P, 1:2)
            @test numobs(P_rep) == 3
            @test P_rep.sample_sizes == [2, 2, 2]
            @test size(P_rep.mask) == (8, 3)
            @test P_rep.data[:, 1:2] == Z_var[1][:, 1:2]
            @test P_rep.data[:, 9:10] == Z_var[2][:, 1:2]
        end

        @testset "numobs, getobs, getindex" begin
            P = PackedReplicates(Z_var)
            @test numobs(P) == 3

            P1 = getobs(P, 1)
            @test numobs(P1) == 1
            @test P1.sample_sizes == [3]
            @test P1.data == Z_var[1]

            P_sub = getobs(P, 1:2)
            @test numobs(P_sub) == 2
            @test P_sub.sample_sizes == [3, 5]
            @test P_sub.data == stackarrays(Z_var[1:2])

            P_vec = getobs(P, [1, 2])
            @test P_vec.sample_sizes == [3, 5]
            @test P_vec.data == P_sub.data

            P_shuf = getobs(P, [3, 1])
            @test P_shuf.sample_sizes == [4, 3]
            @test P_shuf.data == stackarrays(Z_var[[3, 1]])

            @test P[1].data == P1.data
            @test P[1:2].data == P_sub.data
        end

        @testset "joinobs" begin
            P = PackedReplicates(Z_var)
            P12 = getobs(P, 1:2)
            P3 = getobs(P, 3)
            Pj = joinobs(P12, P3)
            @test Pj.sample_sizes == P.sample_sizes
            @test Pj.data == P.data
        end

        @testset "numberreplicates and subsetreplicates" begin
            P = PackedReplicates(Z_var)
            @test numberreplicates(P) == [3, 5, 4]

            P_sub = subsetreplicates(P, 1:2)
            @test numobs(P_sub) == 3
            @test P_sub.sample_sizes == [2, 2, 2]
            @test P_sub.data == stackarrays([z[:, 1:2] for z in Z_var])

            P1 = subsetreplicates(P, 1)
            @test P1.sample_sizes == [1, 1, 1]
            @test size(P1.data, 2) == 3
        end

        @testset "f32" begin
            Z = [randn(n, m) for m in (3, 5, 4)]
            P = PackedReplicates(Z)
            P32 = f32(P)
            @test eltype(P32.data) == Float32
            @test P32.sample_sizes == P.sample_sizes
            @test isnothing(P32.mask)

            Ppad = PackedReplicates(Z; max_sample_size = 8)
            Ppad32 = f32(Ppad)
            @test eltype(Ppad32.data) == Float32
            @test eltype(Ppad32.mask) == Float32
            @test Ppad32.sample_sizes == Ppad.sample_sizes
        end

        @testset "device" begin
            P = PackedReplicates(Z_var) |> f32
            Ppad = PackedReplicates(Z_var; max_sample_size = 8) |> f32
            for dvc in devices
                Pdev = P |> dvc
                @test typeof(Pdev.data) == typeof(P.data |> dvc)
                @test Array(Pdev.data) == P.data
                @test Pdev.sample_sizes == P.sample_sizes
                @test isnothing(Pdev.mask)

                Ppaddev = Ppad |> dvc
                @test typeof(Ppaddev.data) == typeof(Ppad.data |> dvc)
                @test typeof(Ppaddev.mask) == typeof(Ppad.mask |> dvc)
                @test Array(Ppaddev.data) == Ppad.data
                @test Array(Ppaddev.mask) == Ppad.mask
                @test Ppaddev.sample_sizes == Ppad.sample_sizes
            end
        end

        @testset "DataLoader" begin
            P = PackedReplicates(Z_var)
            loader = DataLoader(P; batchsize = 2)
            batch = first(loader)
            @test batch isa PackedReplicates
            @test numobs(batch) == 2
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
        # NB the STORED neighbours of node i. findall(!iszero, A[:, i]) must not be used: a
        # zero-distance edge between coincident locations is stored explicitly, and would be
        # silently skipped
        nbrs(A, i) = rowvals(A)[nzrange(A, i)]
        n = 100
        d = 2
        S = rand(Float32, n, d)
        k = 5
        r = 0.3

        A = A₁ = adjacencymatrix(S, k)
        A₂ = adjacencymatrix(S, r)
        @test eltype(A₁) == Float32
        @test eltype(A₂) == Float32
        @test eltype(A) == Float32

        # Check the neighbourhoods against a brute-force reference built from the full
        # distance matrix (the neighbours of location i are stored in the column A[:, i])
        D = pairwise(Euclidean(), S, S, dims = 1)
        for i ∈ 1:n
            @test sort(nbrs(A₁, i)) == sort(partialsortperm(D[i, :], 2:(k + 1)))
            @test sort(nbrs(A₂, i)) == sort(setdiff(findall(<(r), D[i, :]), i))
        end

        # Randomly selecting k nodes within a node's neighbourhood disc
        seed!(1)
        A₃ = adjacencymatrix(S, k, r)
        @test A₃.n == A₃.m == n
        @test length(adjacencymatrix(S, k, 0.02).nzval) < k*n
        # the selected neighbours must be a subset of the full r-disc neighbourhood
        for i ∈ 1:n
            @test issubset(nbrs(A₃, i), findall(<=(r), D[i, :]))
        end

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
        @test size(adjacencymatrix(S, k)) == (n, n)
        @test all(f(adjacencymatrix(S, k)) .== n - 1)   # every other location, and no more
        @test size(adjacencymatrix(S, r, k)) == (n, n)
        @test size(adjacencymatrix(S, r, k; random = false)) == (n, n)

        # Coincident locations must be treated as neighbours of one another, rather than being
        # discarded along with the self loops. Previously the zero distance between two
        # distinct but co-located points was removed by dropzeros!, which left them with too
        # few neighbours and, in the r method, left them completely isolated
        @testset "coincident locations" begin
            S = [0.0 0.0; 0.0 0.0; 1.0 0.0; 0.5 0.5; 0.2 0.9; 0.7 0.3]
            k = 3
            A = adjacencymatrix(S, k)
            @test all(f(A) .== k)                       # still exactly k neighbours
            @test 2 ∈ nbrs(A, 1)                        # node 2 is co-located with node 1 ...
            @test 1 ∈ nbrs(A, 2)                        # ... and the relation is mutual
            @test nnz(A) == size(S, 1) * k              # the zero-distance edges are retained
            Ar = adjacencymatrix(S, 0.6)
            @test all(f(Ar) .>= 1)                      # no isolated nodes
            @test 2 ∈ nbrs(Ar, 1)

            # more than k+1 coincident locations: the self match is not necessarily returned
            # by the neighbour search at all, so filtering it out by index is what keeps the
            # neighbour count correct
            S = zeros(10, 2)
            S[:, 1] .= 0.0
            @test all(f(adjacencymatrix(S, 3)) .== 3)
            @test all(f(adjacencymatrix(S, 8)) .== 8)
        end

        # A non-Euclidean metric, e.g. great-circle distance for longitude-latitude data
        @testset "metric keyword" begin
            seed!(1)
            n = 60
            S = hcat(360 * rand(n) .- 180, 180 * rand(n) .- 90)
            hav = Haversine(6371.0)
            A = adjacencymatrix(S, 5; metric = hav)
            @test all(f(A) .== 5)
            # values are great-circle distances, and match a brute-force reference
            D = pairwise(hav, permutedims(S))
            for i ∈ 1:n
                @test sort(nbrs(A, i)) == sort(partialsortperm(D[i, :], 2:6))
            end
            @test maximum(A.nzval) > 100                # kilometres, not degrees
            Ar = adjacencymatrix(S, 2000.0; metric = hav)
            for i ∈ 1:n
                @test sort(nbrs(Ar, i)) == sort(setdiff(findall(<(2000.0), D[i, :]), i))
            end
            # the index must be chosen to suit the metric: a ball tree is only valid for a
            # true metric, so a semimetric has to fall back to an exhaustive search
            treename(m) = nameof(typeof(NeuralEstimators._spatialindex(permutedims(S), m)))
            @test treename(Euclidean()) == :KDTree
            @test treename(hav) == :BallTree
            @test treename(SqEuclidean()) == :BruteTree
        end

        # A precomputed distance matrix is no longer accepted, and should say so rather than
        # being silently misread as n locations in n dimensions
        seed!(1)
        S = rand(Float32, 20, 2)
        @test_throws ArgumentError adjacencymatrix(pairwise(Euclidean(), S, S, dims = 1), 5)
        @test_throws ArgumentError adjacencymatrix(pairwise(Euclidean(), S, S, dims = 1), 0.3)
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

    @testset "Missing data" begin

        # removedata()
        d = 5     # dimension of each replicate
        n = 3     # number of observed elements of each replicate: must have n <= d
        m = 50    # number of replicates #NB removedata(Z, p) gives stack overflow when m is large... not fixing because this function is low priority, and may even be removed in the future
        p = rand(d)

        Z = rand(d)
        removedata(Z, n)
        removedata(Z, p[1])
        removedata(Z, p)

        Z = rand(d, m)
        removedata(Z, n)
        removedata(Z, d)
        removedata(Z, n; fixed_pattern = true)
        removedata(Z, n; contiguous_pattern = true)
        removedata(Z, n; contiguous_pattern = true, fixed_pattern = true)
        removedata(Z, p)
        removedata(Z, p; prevent_complete_missing = false)
        # Check that the probability of missingness is roughly correct:
        mapslices(x -> sum(ismissing.(x))/length(x), removedata(Z, p), dims = 2)
        # Check that none of the replicates contain 100% missing:
        @test !(d ∈ unique(mapslices(x -> sum(ismissing.(x)), removedata(Z, p), dims = 1)))

        # encodedata()
        n = 16
        Z = rand(n)
        Z = removedata(Z, 0.25)
        UW = encodedata(Z)
        @test ndims(UW) == 1
        @test size(UW) == (2n,)

        Z = rand(n, n)
        Z = removedata(Z, 0.25)
        UW = encodedata(Z)
        @test ndims(UW) == 2
        @test size(UW) == (2n, n)

        Z = rand(n, n, 3, 5)
        Z = removedata(Z, 0.25)
        UW = encodedata(Z)
        @test ndims(UW) == 4
        @test size(UW) == (n, n, 6, 5)
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
end

@testset "Training display" begin
    io = IOBuffer()
    d = _TrainDisplay(io, false)
    _status!(d, "hello")
    @test String(take!(io)) == "hello\n"

    _status!(d, "phase"; transient = true)
    @test String(take!(io)) == ""

    _finishline!(d)
    @test String(take!(io)) == ""

    d = _TrainDisplay(io, true)
    _status!(d, "hello")
    @test String(take!(io)) == "\rhello\e[K"

    _status!(d, "phase"; transient = true)
    @test String(take!(io)) == "\rphase\e[K"

    _finishline!(d)
    @test String(take!(io)) == "\n"

    # A transient status is erased before a message is logged, which then starts on a line of its own
    _status!(d, "phase"; transient = true)
    take!(io)
    NeuralEstimators._clearstatus!(d)
    @test String(take!(io)) == "\r\r\e[K" && d.nlines == 0
    NeuralEstimators._clearstatus!(d)
    @test String(take!(io)) == "" # nothing to erase

    msg = _epoch_status(6, 100, 0.049, 0.054, 0.046, 2, 5, 5e-4, 0.114)
    @test occursin("6/100", msg)
    @test occursin("Training risk: 0.049", msg)
    @test occursin("Validation risk: 0.054", msg)
    @test occursin("Best: 0.046", msg)
    @test occursin("Epochs since improvement: 2/5", msg)
    @test occursin("5.00E-04", msg)
    @test occursin("0.114 seconds", msg)

    d = _TrainDisplay(true; io = IOBuffer())
    @test d.overwrite == false
    d = _TrainDisplay(false; io = IOBuffer())
    @test d.overwrite == false

    io = IOBuffer()
    d = _TrainDisplay(io, false)
    _status!(d, "header")
    take!(io)
    _bar!(d, 4, 100, 45, 100)
    @test String(take!(io)) == ""

    io = IOBuffer()
    d = _TrainDisplay(io, true)
    _status!(d, "header")
    take!(io)
    _bar!(d, 4, 100, 45, 100)
    out1 = String(take!(io))
    @test occursin("%|", out1)
    @test occursin("45/100", out1)
    _bar!(d, 4, 100, 100, 100)
    out2 = String(take!(io))
    @test occursin("\e[A", out2)
    @test occursin("%|", out2)
    @test occursin("100/100", out2)

    io = IOBuffer()
    d = _TrainDisplay(io, false)
    _refresh_status!(d, :data)
    @test String(take!(io)) == ""
    _refresh_status!(d, :data, 2.0)
    @test String(take!(io)) == "Refreshing training data... finished in 2.0 seconds.\n"
    _refresh_status!(d, :parameters, 0.412)
    @test String(take!(io)) == "Refreshing training parameters... finished in 0.412 seconds.\n"
    _refresh_status!(d, :data, 2.0; first = true)
    @test String(take!(io)) == "Simulating training data... finished in 2.0 seconds.\n"
    _refresh_status!(d, :parameters, 0.412; first = true)
    @test String(take!(io)) == "Simulating training parameters... finished in 0.412 seconds.\n"

    io = IOBuffer()
    d = _TrainDisplay(io, true)
    _status!(d, "header")
    take!(io)
    _refresh_status!(d, :parameters; first = true)
    out = String(take!(io))
    @test occursin("Simulating training parameters...", out)
    @test !occursin("finished in", out)
    @test !occursin("Refreshing", out)
    _refresh_status!(d, :parameters, 0.412; first = true)
    _refresh_status!(d, :data, 2.105; first = true)
    out = String(take!(io))
    @test occursin("Simulating training parameters... finished in 0.412 seconds.", out)
    @test occursin("Simulating training data... finished in 2.105 seconds.", out)
    @test occursin("\e[A", out)
    _bar!(d, 4, 100, 45, 100)
    out = String(take!(io))
    @test occursin("%|", out)
    @test occursin("45/100", out)
    @test occursin("finished in 0.412 seconds.", out)
    @test findfirst("%|", out) < findfirst("Simulating", out)
    _refresh_status!(d, :data, 1.5)
    out = String(take!(io))
    @test occursin("Refreshing training data... finished in 1.5 seconds.", out)
    @test findfirst("%|", out) < findfirst("Refreshing", out)
    _clear_refresh!(d)
    @test isempty(d.param_status)
    @test isempty(d.data_status)
    out = String(take!(io))
    @test occursin("header", out)
    @test occursin("%|", out)
    @test !occursin("Refreshing", out)
    @test !occursin("Simulating", out)
    _finishline!(d)
    out = String(take!(io))
    @test occursin("header", out)
    @test occursin("\n", out)
    @test !occursin("%|", out)

    @test _fit_line("hello", 80) == "hello"
    @test _fit_line("hello", 3) == "hel"
    @test _fit_line("hello", 0) == ""
    @test _fit_line("█"^10 * "░"^10, 8) == "█"^8
    @test textwidth(_fit_line("█"^10 * "░"^10, 8)) == 8

    buf = IOBuffer()
    io = IOContext(buf, :displaysize => (24, 20))
    d = _TrainDisplay(io, true)
    long = "abcdefghijklmnopqrstuvwxyz"
    _status!(d, long)
    out = String(take!(buf))
    @test occursin("abcdefghijklmnopqrst", out)
    @test !occursin("uvwxyz", out)
    @test textwidth(_fit_line(long, 20)) == 20

    buf = IOBuffer()
    io = IOContext(buf, :displaysize => (24, 80))
    d = _TrainDisplay(io, true)
    _status!(d, "hello")
    take!(buf)
    d.term_cols = 200
    d.nlines = 1
    _status!(d, "hello")
    out = String(take!(buf))
    @test startswith(out, "\n")
    @test occursin("hello", out)
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

    # empirical variogram (CPU only; not device-aware)
    if dvc == cpu_device()
        n_bins = 20
        nx = ny = 8
        Zsmall = randn(nx, ny)
        Dsmall = pairwise(Euclidean(), expandgrid(1:nx, 1:ny), dims = 1)
        v_pair = variogram(vec(Zsmall), Dsmall; n_bins)
        @test length(v_pair) == n_bins
        @test all(x -> isnan(x) || x >= 0, v_pair)

        v_fft = variogram(Zsmall; n_bins)
        @test isapprox(v_pair, v_fft; rtol = 1e-8, nans = true)

        Zstack = randn(nx, ny, 3)
        v_fft_stack = variogram(Zstack; n_bins)
        @test size(v_fft_stack) == (n_bins, 3)
        v_pair_stack = reduce(hcat, (variogram(vec(Zstack[:, :, k]), Dsmall; n_bins) for k = 1:3))
        @test isapprox(v_pair_stack, v_fft_stack; rtol = 1e-8, nans = true)

        v_pair_full = variogram(vec(Zsmall), Dsmall; n_bins, maxlag = 1)
        v_fft_full = variogram(Zsmall; n_bins, maxlag = 1)
        @test isapprox(v_pair_full, v_fft_full; rtol = 1e-8, nans = true)

        @test_throws ArgumentError variogram(vec(Zsmall), Dsmall; maxlag = 0)
        @test_throws ArgumentError variogram(vec(Zsmall), Dsmall; maxlag = 1.1)
        @test_throws ArgumentError variogram(Zsmall; maxlag = 0)
        @test_throws ArgumentError variogram(Zsmall; maxlag = 1.1)
    end
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

# Precision matrix Σ⁻¹ = T'D⁻²T of each component of a GaussianMixture, constructed explicitly from
# the distributional parameters κ; a J × K matrix of d × d matrices
function mixtureprecisions(q::GaussianMixture, κ::AbstractMatrix)
    d, J, K = q.d, q.num_components, size(κ, 2)
    _, _, σ, t = NeuralEstimators.distributionparameters(q, κ)
    σ = reshape(σ, d, J, K)
    subdiagonal = [CartesianIndex(i, j) for i = 2:d for j = 1:(i - 1)] # row by row
    map(Iterators.product(1:J, 1:K)) do (j, k)
        T = Matrix{Float64}(I, d, d)
        isnothing(t) || (T[subdiagonal] = reshape(t, :, J, K)[:, j, k])
        T' * Diagonal(σ[:, j, k] .^ -2) * T
    end
end

@testset "Approximate distributions: $dvc" for dvc ∈ devices
    for d = 1:5
        dstar = 2d
        K = 10
        θ = rand32(d, K) |> dvc
        TZ = rand32(dstar, K) |> dvc

        # NB the two-argument constructor takes (log_scale, bias)
        @testset "ActNorm: $args" for args in (d, (3.0 * ones(d), 2.0 * ones(d)))
            an = ActNorm(args...) |> dvc
            U, log_det_J = forward(an, θ)
            @test size(U) == (d, K)
            @test length(log_det_J) == 1
            @test log_det_J == sum(an.log_scale)
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

        # Regression test for the NaN-gradient bug fixed in v0.4.0. With well-separated
        # mixture logits, a softmax weight underflows to zero in Float32; forming the log
        # weights as log(softmax(logits)) then gives -Inf and a NaN gradient, while the
        # log-density itself stays finite. The log-density must therefore be finite AND
        # differentiable here.
        # NB the log weights are formed separately for diagonal and dense covariance matrices
        @testset "GaussianMixture: separated logits (diagonal = $diagonal)" for diagonal in (true, false)
            J = 10
            q = GaussianMixture(d, dstar; num_components = J, diagonal = diagonal, backend = Flux) |> dvc
            @test all(isfinite, NeuralEstimators._logdensity(q, θ, TZ))

            # Push the mixture logits far apart, as training does. The weights head is
            # the first branch of the Parallel block at the end of the inference network.
            weights_head = q.inference_network.layers[end].layers[1]
            copyto!(weights_head.bias,
                    eltype(weights_head.bias).(collect(range(0, -200; length = J))))

            # The log-density stayed finite even with the bug, so the gradient is the
            # assertion that matters. Differentiating with respect to the summary
            # statistics exercises the whole inference network, log weights included.
            @test all(isfinite, NeuralEstimators._logdensity(q, θ, TZ))
            g = Flux.gradient(z -> sum(NeuralEstimators._logdensity(q, θ, z)), TZ)[1]
            @test all(isfinite, g)
        end

        # With diagonal = false, the covariance matrix of each component is dense, and it is
        # parameterised through its inverse, Σ⁻¹ = T'D⁻²T, where T is unit lower triangular.
        # NB when d = 1 there are no sub-diagonal elements, and the mixture is the diagonal one
        @testset "GaussianMixture: dense covariance matrices" begin
            # Give T sizeable sub-diagonal elements, so that the tests below can detect errors in
            # how T is applied. The corresponding head is the last branch of the Parallel block.
            function dense_mixture(J)
                q = GaussianMixture(d, dstar; num_components = J, diagonal = false, backend = Flux)
                d > 1 && fill!(q.inference_network.layers[end].layers[4].bias, 1)
                return q
            end

            J = 3
            q_cpu = dense_mixture(J)
            @test numdistributionalparams(q_cpu) == (1 + 2d + d * (d - 1) ÷ 2) * J
            @test numdistributionalparams(GaussianMixture(d, dstar; num_components = J, backend = Flux)) == numdistributionalparams(q_cpu) # dense by default
            κ = Float64.(q_cpu.inference_network(cpu(TZ)))
            @test size(κ, 1) == numdistributionalparams(q_cpu)

            # The log-density agrees with a reference computed from the precision matrices
            logits, μ = NeuralEstimators.distributionparameters(q_cpu, κ)
            Λ = mixtureprecisions(q_cpu, κ)
            reference = map(1:K) do k
                log_w = Flux.NNlib.logsoftmax(logits[:, k])
                Flux.NNlib.logsumexp(map(1:J) do j
                    Δ = Float64.(cpu(θ)[:, k]) - reshape(μ, d, J, K)[:, j, k]
                    log_w[j] + (logdet(Λ[j, k]) - d * log(2π) - Δ' * Λ[j, k] * Δ) / 2
                end)
            end
            q = q_cpu |> dvc
            dens = NeuralEstimators._logdensity(q, θ, TZ)
            @test size(dens) == (1, K)
            @test vec(cpu(dens)) ≈ reference rtol = 1e-4

            # The log-density is differentiable, and the head for the sub-diagonal elements is trained
            g = Flux.gradient(z -> sum(NeuralEstimators._logdensity(q, θ, z)), TZ)[1]
            @test all(isfinite, g)
            if d > 1
                ∇ = Flux.gradient(m -> sum(NeuralEstimators._logdensity(m, θ, TZ)), q)[1]
                ∇t = cpu(∇.inference_network.layers[end].layers[4].weight)
                @test all(isfinite, ∇t)
                @test !iszero(∇t)
            end

            # The sampler is consistent with the density: the samples have the mean and covariance
            # matrix of the mixture, m = Σⱼ wⱼμⱼ and Σⱼ wⱼ(Σⱼ + μⱼμⱼ') - mm', where Σⱼ = (Tⱼ'Dⱼ⁻²Tⱼ)⁻¹
            # NB a mixture with a single component is the distribution constructed by Gaussian()
            @testset "sampling: $num_components component(s)" for num_components in (1, 3)
                qs = dense_mixture(num_components)
                κs = Float64.(qs.inference_network(cpu(TZ)[:, 1:1]))
                logits_s, μs = NeuralEstimators.distributionparameters(qs, κs)
                w = Flux.NNlib.softmax(vec(logits_s))
                μs = reshape(μs, d, num_components)
                Σs = inv.(vec(mixtureprecisions(qs, κs)))
                m = sum(w[j] * μs[:, j] for j = 1:num_components)
                V = sum(w[j] * (Σs[j] + μs[:, j] * μs[:, j]') for j = 1:num_components) - m * m'
                N = 200_000
                samples = sampleposterior(qs |> dvc, TZ[:, 1:1], N)
                @test size(samples) == (d, N, 1)
                @test all(abs.(vec(mean(samples, dims = 2)) .- m) .< 6 .* sqrt.(diag(V) ./ N))
                @test cov(samples[:, :, 1], dims = 2) ≈ V rtol = 0.05
            end
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
            @test size(samples) == (d, N, K)
        end
    end
end

# Gaussian is a convenience constructor for a GaussianMixture with a single component and, by default,
# a dense covariance matrix, so its density and sampler are tested with those of GaussianMixture above
@testset "Gaussian" begin
    d, dstar = 3, 6
    num_dense = numdistributionalparams(GaussianMixture(d, dstar; num_components = 1, diagonal = false, backend = Flux))
    num_diagonal = numdistributionalparams(GaussianMixture(d, dstar; num_components = 1, diagonal = true, backend = Flux))
    @test num_dense > num_diagonal

    q = Gaussian(d, dstar; backend = Flux)
    @test q isa GaussianMixture
    @test q.num_components == 1
    @test numdistributionalparams(q) == num_dense # dense covariance matrix by default
    @test numdistributionalparams(Gaussian(d, dstar; diagonal = true, backend = Flux)) == num_diagonal
    @test Gaussian(d, dstar; num_components = 5, backend = Flux).num_components == 1 # fixed to one
    @test numdistributionalparams(Gaussian(d; num_summaries = dstar, backend = Flux)) == num_dense # keyword num_summaries
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
        target = randn(Float32, size(y.ndata.Z)) |> dvc # NB not similar(), see testbackprop()
        ∇ = Flux.gradient(l -> mae(l(g).ndata.Z, target), l)
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
        Σ = convert.(Matrix, Σ)
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
    # - with and without conditioning on sample size
    # - common data formats
    n = 10     # dimension of each data replicate
    M = (3, 4) # number of replicates in each data set
    w = 32     # width of each hidden layer
    d = 5      # output dimension
    dₜ = 16    # dimension of neural summary statistic
    for condition_on_sample_size in (false, true)
        for data in ("unstructured", "grid", "graph")
            dₛ = condition_on_sample_size ? 1 : 0
            if data == "unstructured"
                Z = [rand32(n, m) for m ∈ M]
                ψ = Chain(Dense(n, w), Dense(w, dₜ), MLUtils.flatten)
            elseif data == "grid"
                Z = [rand32(10, 10, 1, m) for m ∈ M]
                ψ = Chain(Conv((5, 5), 1 => dₜ), GlobalMeanPool(), MLUtils.flatten)
            elseif data == "graph"
                Z = [spatialgraph(rand(100, 2), rand(100, m)) for m ∈ M] #NB this can break when n is taken to be small like n=5 (run it many times and you will eventually see ERROR: AssertionError: DataStore: data[e] has 1 observations, but n = 0)
                propagation = Chain(SpatialGraphConv(1 => 16), SpatialGraphConv(16 => dₜ))
                readout = GlobalPool(mean)
                ψ = GNNSummary(propagation, readout)
            end
            ϕ = Chain(Dense(dₜ + dₛ, w, relu), Dense(w, d)) # outer network
            ds = DeepSet(ψ, ϕ; condition_on_sample_size)
            show(devnull, ds)
            # Forward evaluation
            y = ds(Z)
            @test size(y) == (d, length(M))
            if data == "graph"
                P = NeuralEstimators.PackedGraphs(Z)
                @test ds(P) ≈ y
                testbackprop(ds, P, dvc)
            else
                P = PackedReplicates(Z)
                @test ds(P) ≈ y
                testbackprop(ds, P, dvc)
            end
            # Basic back propagation
            testbackprop(ds, Z, dvc)
        end
    end
end

@testset "DeepSet aggregation: $dvc" for dvc ∈ devices
    # The replicates of all data sets are aggregated in a single vectorised call: check that this
    # agrees with applying the DeepSet to each data set separately, in both value and gradient
    n = 10     # dimension of each data replicate
    w = 32     # width of each hidden layer
    d = 5      # output dimension
    dₜ = 16    # dimension of neural summary statistic
    logsumexp = Flux.NNlib.logsumexp
    customaggregator(x; dims) = mean(x, dims = dims) # aggregation function without a segmented implementation
    aggregators = (mean, sum, maximum, minimum, logsumexp, customaggregator)

    @testset "a = $a" for a ∈ aggregators
        for M ∈ ((3, 3, 3), (3, 4, 7)) # equal and varying sample sizes
            for condition_on_sample_size ∈ (false, true)
                dₛ = condition_on_sample_size ? 1 : 0
                ψ = Chain(Dense(n, w, relu), Dense(w, dₜ, relu))
                ϕ = Chain(Dense(dₜ + dₛ, w, relu), Dense(w, d))
                ds = DeepSet(ψ, ϕ, a; condition_on_sample_size) |> dvc
                Z = [rand32(n, m) for m ∈ M] |> dvc

                # Forward evaluation
                @test ds(Z) ≈ reduce(hcat, [ds(z) for z ∈ Z])

                P = PackedReplicates(Z)
                @test ds(P) ≈ ds(Z)

                # Back propagation
                ∇ = Flux.gradient(ds -> sum(abs2, ds(Z)), ds)[1]
                ∇ᵣ = Flux.gradient(ds -> sum(abs2, reduce(hcat, [ds(z) for z ∈ Z])), ds)[1]
                ∇ₚ = Flux.gradient(ds -> sum(abs2, ds(P)), ds)[1]
                @test all(isapprox.(trainables(∇), trainables(∇ᵣ), rtol = 1.0f-3))
                @test all(isapprox.(trainables(∇), trainables(∇ₚ), rtol = 1.0f-3))

                if a !== customaggregator
                    Ppad = PackedReplicates(Z; max_sample_size = maximum(M))
                    @test ds(Ppad) ≈ ds(Z)
                    ∇ₚₚ = Flux.gradient(ds -> sum(abs2, ds(Ppad)), ds)[1]
                    @test all(isapprox.(trainables(∇), trainables(∇ₚₚ), rtol = 1.0f-3))
                end
            end
        end
    end
end

@testset "DeepSet graph aggregation: $dvc" for dvc ∈ devices
    # A batch of graphs is packed into a single supergraph before being moved to the device,
    # padding the replicate dimension where necessary. Check that the packed path agrees with
    # applying the DeepSet to each data set separately, in both value and gradient
    dₜ = 8     # dimension of neural summary statistic
    w = 32     # width of each hidden layer
    d = 3      # output dimension
    logsumexp = Flux.NNlib.logsumexp

    mkψ() = GNNSummary(Chain(SpatialGraphConv(1 => 16), SpatialGraphConv(16 => dₜ)), GlobalPool(mean))
    # Replicates stored in the node features (spatial locations fixed over replicates)
    mkfeatures(ms) = [spatialgraph(rand(60, 2), rand(60, m)) for m ∈ ms]
    # Replicates stored as subgraphs (spatial locations varying between replicates)
    function mksubgraphs(ms)
        map(collect(ms)) do m
            n = rand(50:60, m)
            spatialgraph([rand(nᵢ, 2) for nᵢ ∈ n], [rand(nᵢ) for nᵢ ∈ n])
        end
    end

    @testset "numberreplicates" begin
        # NB a singleton replicate dimension in the node features is not the replicate axis:
        # when the replicates are stored as subgraphs, the count comes from the subgraphs
        @test collect(numberreplicates.(mkfeatures((1, 4)))) == [1, 4]
        @test collect(numberreplicates.(mksubgraphs((1, 5)))) == [1, 5]
    end

    @testset "PackedGraphs" begin
        P = NeuralEstimators.PackedGraphs(mkfeatures((3, 3)))
        @test P.layout isa NeuralEstimators.ReplicatesInFeatures
        @test isnothing(P.mask)                      # equal replicates need no padding
        @test numobs(P) == 2
        @test numberreplicates(P) == [3, 3]
        show(devnull, P)

        P = NeuralEstimators.PackedGraphs(mkfeatures((3, 5)))
        @test size(P.mask) == (5, 2)
        @test vec(sum(P.mask, dims = 1)) == Float32[3, 5]
        @test samplesize(P) == Float32[3, 5]
        @test logsamplesize(P) ≈ log.(Float32[3, 5])
        @test_throws ArgumentError getobs(P, 1)

        @test NeuralEstimators.PackedGraphs(mksubgraphs((3, 5))).layout isa NeuralEstimators.ReplicatesInSubgraphs
        @test isnothing(NeuralEstimators.PackedGraphs(mksubgraphs((3, 5))).mask)

        # sample_sizes must stay on the host when the object is moved to the device
        P = NeuralEstimators.PackedGraphs(mkfeatures((3, 5))) |> dvc
        @test P.sample_sizes isa Vector{Int}
    end

    @testset "equivalence: $layout, a = $(nameof(a)), cond = $cond" for layout ∈ (:features, :subgraphs),
        a ∈ (mean, sum, maximum, minimum, logsumexp),
        cond ∈ (false, true)

        ms = (3, 4, 1, 7)
        Z = layout === :features ? mkfeatures(ms) : mksubgraphs(ms)
        ϕ = Chain(Dense(dₜ + Int(cond), w, relu), Dense(w, d))
        ds = DeepSet(mkψ(), ϕ, a; condition_on_sample_size = cond)
        y = ds(Z)
        @test size(y) == (d, length(ms))
        @test y ≈ reduce(hcat, [ds([z]) for z ∈ Z]) rtol = 1e-4
        # the gradients must agree too
        tgt = randn(Float32, d, length(ms))
        g1 = trainables(Flux.gradient(m -> mae(m(Z), tgt), ds)[1])
        g2 = trainables(Flux.gradient(m -> mae(reduce(hcat, [m([z]) for z ∈ Z]), tgt), ds)[1])
        @test all(isapprox.(g1, g2; rtol = 1e-3, atol = 1e-6))
    end

    @testset "padded gradients are finite" begin
        Z = mkfeatures((2, 9))   # unequal replicates, so the batch is padded
        ds = DeepSet(mkψ(), Chain(Dense(dₜ, w, relu), Dense(w, d)))
        tgt = randn(Float32, d, 2)
        grads = trainables(Flux.gradient(m -> mae(m(Z), tgt), ds)[1])
        @test !isempty(grads)
        @test all(x -> all(isfinite, x), grads)
    end

    @testset "padding invariance" begin
        # the result for a data set must not depend on how much the batch was padded
        Z = mkfeatures((2, 3))
        ds = DeepSet(mkψ(), Chain(Dense(dₜ, w, relu), Dense(w, d)))
        y = ds(Z)
        @test ds(vcat(Z, mkfeatures((12,))))[:, 1:2] ≈ y rtol = 1e-4
    end

    @testset "aggregation restricted on the padded path" begin
        ds = DeepSet(mkψ(), Chain(Dense(dₜ, w, relu), Dense(w, d)), median)
        @test_throws ArgumentError ds(NeuralEstimators.PackedGraphs(mkfeatures((2, 5)))) # padded
        @test ds(mkfeatures((4, 4))) isa AbstractMatrix       # equal replicates, no mask
    end

    @testset "_replicategroups" begin
        rg = Base.get_extension(NeuralEstimators, :NeuralEstimatorsGNNExt)._replicategroups
        padded(m, s, groups) = sum(maximum(m[g]) * sum(s[g]) for g ∈ groups)
        @test rg([3, 3, 3], [10, 10, 10]) == [[1, 2, 3]]     # equal m: a single group
        @test rg([1, 1, 50, 50], fill(10, 4)) == [[1, 2], [3, 4]]
        @test rg([7], [10]) == [[1]]
        for _ ∈ 1:20
            K = rand(1:40)
            m = rand(1:100, K)
            s = rand(100:1000, K)
            groups = rg(m, s)
            @test 1 ≤ length(groups) ≤ 4
            @test sort(reduce(vcat, groups)) == 1:K           # a partition of the batch
            @test padded(m, s, groups) ≤ padded(m, s, [1:K])  # never worse than a single group
        end
    end

    @testset "grouping by the number of replicates" begin
        ms = (3, 40, 1, 38, 2, 41)
        Z = mkfeatures(ms)
        G = NeuralEstimators._packbatch(Z)
        @test G isa NeuralEstimators.GroupedPackedGraphs
        @test length(G.groups) > 1
        @test numobs(G) == length(ms)
        @test numberreplicates(G) == collect(ms)
        @test samplesize(G) == Float32.(collect(ms))
        @test_throws ArgumentError getobs(G, 1)
        show(devnull, G)
        # equal replicates and replicates as subgraphs are packed as before
        @test NeuralEstimators._packbatch(mkfeatures((3, 3))) isa NeuralEstimators.PackedGraphs
        @test NeuralEstimators._packbatch(mksubgraphs((1, 5))) isa NeuralEstimators.PackedGraphs
        # the order must stay on the host when the object is moved to the device
        @test (G |> dvc).order isa Vector{Int}

        # grouped, single padded supergraph, and one data set at a time must all agree,
        # in value and in gradient (including the expert statistic from conditioning on m)
        for cond ∈ (false, true)
            ϕ = Chain(Dense(dₜ + Int(cond), w, relu), Dense(w, d))
            ds = DeepSet(mkψ(), ϕ; condition_on_sample_size = cond)
            P = NeuralEstimators.PackedGraphs(Z)
            y = ds(G)
            @test size(y) == (d, length(ms))
            @test y ≈ ds(P) rtol = 1e-4
            @test y ≈ ds(Z) rtol = 1e-4
            @test y ≈ reduce(hcat, [ds([z]) for z ∈ Z]) rtol = 1e-4
            # NB gradients are compared in double precision: in single precision, the gradient of
            # the bias in the final layer of ψ, a sum over every node and replicate in the batch,
            # carries rounding error of order 1e-4 that depends on how the batch is packed
            Z64 = f64.(Z)
            G64 = NeuralEstimators._packbatch(Z64)
            P64 = NeuralEstimators.PackedGraphs(Z64)
            ds64 = f64(ds)
            tgt = randn(d, length(ms))
            g1 = trainables(Flux.gradient(m -> mae(m(G64), tgt), ds64)[1])
            g2 = trainables(Flux.gradient(m -> mae(m(P64), tgt), ds64)[1])
            @test all(isapprox.(g1, g2; rtol = 1e-6))
            testbackprop(ds, G, dvc)
        end
    end

    @testset "mixed storage layouts are rejected" begin
        Z = vcat(mkfeatures((3,)), mksubgraphs((4,)))
        @test_throws ArgumentError NeuralEstimators.PackedGraphs(Z)
    end

    @testset "_packbatch is inert away from graph data" begin
        Z = [rand32(5, m) for m ∈ (2, 3)]
        @test NeuralEstimators._packbatch(Z) === Z
        @test NeuralEstimators._packbatch(rand32(3, 4)) isa Matrix
        @test NeuralEstimators._packbatch(PackedReplicates(Z)) isa PackedReplicates
        @test NeuralEstimators._packbatch((Z, rand32(2, 2)))[1] === Z
        # graph batches are packed, including inside DataAndSummaries
        @test NeuralEstimators._packbatch(mkfeatures((2, 2))) isa NeuralEstimators.PackedGraphs
        dS = DataAndSummaries(mkfeatures((2, 2)), rand32(1, 2))
        @test NeuralEstimators._packbatch(dS).Z isa NeuralEstimators.PackedGraphs
    end

    @testset "_aggregatemiddle" begin
        # unit tests with plain arrays, independent of any graph machinery
        R = rand32(4, 3, 5)
        a = NeuralEstimators.ElementwiseAggregator(mean)
        @test NeuralEstimators._aggregatemiddle(a, R, nothing) ≈ dropdims(mean(R, dims = 2); dims = 2)
        mask = Float32[1 1 1 1 1; 1 1 1 1 1; 0 1 0 1 0]   # third replicate missing for sets 1, 3, 5
        got = NeuralEstimators._aggregatemiddle(a, R, mask)
        want = reduce(hcat, [mean(R[:, findall(!iszero, mask[:, k]), k], dims = 2) for k ∈ 1:5])
        @test got ≈ want
        @test_throws ArgumentError NeuralEstimators._aggregatemiddle(NeuralEstimators.ElementwiseAggregator(median), R, mask)
    end

    @testset "SpatialGraphConv: equivalence with the unoptimised formulation, in = $inch, m = $m" for inch ∈ (1, 6), m ∈ (1, 4)
        # Γ is applied as a single matrix multiplication over the flattened replicate and node
        # dimensions rather than with batched_mul over the nodes, and the edge weights are
        # broadcast over the replicates rather than repeated. Both are meant to be exactly
        # equivalent to the straightforward formulation, which is reproduced here
        batched_mul = Flux.NNlib.batched_mul
        normalise = Base.get_extension(NeuralEstimators, :NeuralEstimatorsGNNExt).normalise_edge_neighbors
        function unoptimised(l, g, x)
            mᵢ = size(x, 2)
            e = :e ∈ keys(g.edata) ? g.edata.e : permutedims(g.graph[3])
            isa(e, AbstractVector) && (e = permutedims(e))
            w̃ = normalise(g, l.w(e))
            isa(w̃, AbstractVector) && (w̃ = permutedims(w̃))
            isa(w̃, AbstractMatrix) && (w̃ = reshape(w̃, size(w̃, 1), 1, size(w̃, 2)))
            w̃ = repeat(w̃, 1, mᵢ, 1)
            msg = apply_edges((xi, xj, ww) -> ww .* l.f(xi, xj), g, x, x, w̃)
            h̄ = aggregate_neighbors(g, +, msg)
            return l.g.(batched_mul(l.Γ1, x) .+ batched_mul(l.Γ2, h̄) .+ l.b)
        end

        n = 40
        Z = inch == 1 ? rand(n, m) : rand(inch, n, m)
        g = spatialgraph(rand(n, 2), Z)
        l = SpatialGraphConv(inch => 5)
        x = g.ndata.Z
        @test size(l(g).ndata.Z) == (5, m, n)
        @test l(g, x) ≈ unoptimised(l, g, x) rtol = 1e-5
        tgt = randn(Float32, size(l(g, x))...)
        g1 = trainables(Flux.gradient(ll -> mae(ll(g, x), tgt), l)[1])
        g2 = trainables(Flux.gradient(ll -> mae(unoptimised(ll, g, x), tgt), l)[1])
        @test all(isapprox.(g1, g2; rtol = 1e-4, atol = 1e-7))
    end

    @testset "PowerDifference: fused broadcast matches the materialised form" begin
        # The subtraction is dotted so that the whole expression is a single fused broadcast
        # rather than three edge-sized temporaries; the arithmetic must be untouched
        materialised(f, x, y) = (abs.(sigmoid.(f.a) .* x - (1 .- sigmoid.(f.a)) .* y)) .^ softplus.(f.b)
        X = rand(Float32, 5, 100)
        Y = rand(Float32, 5, 100)
        for f ∈ (PowerDifference(), PowerDifference([0.5f0], [2.0f0]), PowerDifference(randn(Float32, 5), [0.75f0]))
            @test f(X, Y) ≈ materialised(f, X, Y)
            @test f((X, Y)) ≈ f(X, Y)
            tgt = randn(Float32, size(f(X, Y))...)
            g1 = trainables(Flux.gradient(ff -> mae(ff(X, Y), tgt), f)[1])
            g2 = trainables(Flux.gradient(ff -> mae(materialised(ff, X, Y), tgt), f)[1])
            @test all(isapprox.(g1, g2; rtol = 1e-5, atol = 1e-7))
        end
    end

    @testset "SpatialGraphConv: untraced non-trainable w, $(nameof(Weights)), m = $m" for Weights ∈ (KernelWeights, IndicatorWeights), m ∈ (1, 4)
        # The spatial weight function depends only on the fixed spatial information, so when
        # it holds no trainable parameters its evaluation is kept off the AD tape. That must
        # not change the forward value, and must not perturb any gradient that does exist
        GNNExt = Base.get_extension(NeuralEstimators, :NeuralEstimatorsGNNExt)
        normalise = GNNExt.normalise_edge_neighbors
        q = 10
        w = Weights(1.0, q)   # h_max covers the neighbour distances, so no empty neighbourhood
        @test GNNExt._wtrainable(w) == false
        @test isempty(trainables(w))

        n = 40
        g = spatialgraph(rand(n, 2), rand(n, m))
        l = SpatialGraphConv(1 => 5, w = w, w_out = q)
        x = g.ndata.Z

        # reference that differentiates through w, i.e. the behaviour before the change
        function traced(l, g, x)
            e = :e ∈ keys(g.edata) ? g.edata.e : permutedims(g.graph[3])
            isa(e, AbstractVector) && (e = permutedims(e))
            w̃ = GNNExt.coerce3Darray(normalise(g, l.w(e)))
            msg = apply_edges((xi, xj, ww) -> ww .* l.f(xi, xj), g, x, x, w̃)
            h̄ = aggregate_neighbors(g, +, msg)
            return l.g.(GNNExt._densemul(l.Γ1, x) .+ GNNExt._densemul(l.Γ2, h̄) .+ l.b)
        end

        @test l(g, x) ≈ traced(l, g, x)
        @test all(isfinite, l(g, x))
        tgt = randn(Float32, size(l(g, x))...)
        ∇1 = Flux.gradient(ll -> mae(ll(g, x), tgt), l)[1]
        ∇2 = Flux.gradient(ll -> mae(traced(ll, g, x), tgt), l)[1]
        # Compare the genuinely trainable parameters one by one. NB trainables() must not be
        # used on the gradient objects themselves: the traced version additionally carries
        # tangents for w's own fields, which Optimisers.trainable(::typeof(w)) == NamedTuple()
        # discards on the model but which are still present on the tangent
        for get ∈ (∇ -> ∇.Γ1, ∇ -> ∇.Γ2, ∇ -> ∇.b, ∇ -> ∇.f.a, ∇ -> ∇.f.b)
            @test get(∇1) ≈ get(∇2) rtol = 1e-5
            @test any(!iszero, get(∇1))   # ignoring w must not zero the gradients that matter
        end
        # the weight function itself receives no tangent, which is the point of the change
        @test isnothing(∇1.w) || all(isnothing, values(∇1.w))
    end

    @testset "SpatialGraphConv: a trainable w is still differentiated" begin
        # The gate is dispatch-based, so the default Chain weight function must fall through
        # to the differentiated branch and produce non-zero gradients for its parameters
        GNNExt = Base.get_extension(NeuralEstimators, :NeuralEstimatorsGNNExt)
        l = SpatialGraphConv(1 => 5)
        @test GNNExt._wtrainable(l.w) == true
        g = spatialgraph(rand(40, 2), rand(40, 3))
        x = g.ndata.Z
        tgt = randn(Float32, size(l(g, x))...)
        ∇ = Flux.gradient(ll -> mae(ll(g, x), tgt), l)[1]
        @test !isnothing(∇.w)   # a tangent is built for w, i.e. it was differentiated

        # The default w must never return an identically zero weight for every edge: that
        # would zero h̄ and hence the whole Γ2 h̄ term, and with a relu output (which was the
        # default before) relu's zero gradient means it could never recover. Its output layer
        # therefore uses softplus. Several seeds, since the failure was initialisation-dependent
        for seed ∈ 1:6
            seed!(seed)
            lᵢ = SpatialGraphConv(1 => 5)
            gᵢ = spatialgraph(rand(40, 2), rand(40, 3))
            @test all(>(0), lᵢ.w(permutedims(gᵢ.graph[3])))
            tgtᵢ = randn(Float32, size(lᵢ(gᵢ, gᵢ.ndata.Z))...)
            ∇ᵢ = Flux.gradient(ll -> mae(ll(gᵢ, gᵢ.ndata.Z), tgtᵢ), lᵢ)[1]
            @test any(gⱼ -> any(!iszero, gⱼ), trainables(∇ᵢ.w))
        end

        # A non-degenerate trainable w, to assert the gradient is actually non-zero. NB the
        # default w cannot be used for this: its output layer inherits the activation g
        # (relu by default), which for many initialisations clamps every edge weight to
        # exactly zero, leaving w with an identically zero gradient
        wt = Chain(Dense(1 => 8, tanh), Dense(8 => 1, softplus))
        @test GNNExt._wtrainable(wt) == true
        lt = SpatialGraphConv(1 => 5, w = wt, w_out = 1)
        @test all(!iszero, lt.w(permutedims(g.graph[3])))
        tgt2 = randn(Float32, size(lt(g, x))...)
        ∇t = Flux.gradient(ll -> mae(ll(g, x), tgt2), lt)[1]
        @test any(gᵢ -> any(!iszero, gᵢ), trainables(∇t.w))
    end
end

@testset "Graph data: estimator integration (variable m)" begin
    # End-to-end training on graph data with a varying number of replicates, through the code
    # paths that wrap the batch differently: a tuple input (PosteriorEstimator, guarding the
    # _packbatch(::Tuple) method) and DataAndSummaries
    d = 2
    dₜ = 8
    w = 16
    K = 12
    ψ = GNNSummary(Chain(SpatialGraphConv(1 => 16), SpatialGraphConv(16 => dₜ)), GlobalPool(mean))
    mknet() = DeepSet(deepcopy(ψ), Chain(Dense(dₜ, w, relu), Dense(w, dₜ)))
    Z = [spatialgraph(rand(40, 2), rand(40, mᵢ)) for mᵢ ∈ rand(1:6, K)]
    θ = rand32(d, K)
    kw = (epochs = 2, batchsize = 4, verbose = false, use_gpu = false)

    est = PointEstimator(mknet(), d; num_summaries = dₜ)
    est = train(est, θ, θ, Z, Z; kw...)
    @test size(estimate(est, Z; use_gpu = false)) == (d, K)
    @test assess(est, θ, Z; use_gpu = false) isa Assessment
    assessment = assess(est, θ, Z; probs = [0.05, 0.95], B = 5, use_gpu = false) # non-parametric bootstrap intervals
    @test all(["lower", "upper", "α"] .∈ Ref(names(assessment.estimates)))
    @test nrow(coverage(assessment)) == d
    @test size(bootstrap(est, spatialgraph(rand(40, 2), rand(40, 8)); B = 5, use_gpu = false), 1) == d

    # expert summaries are concatenated to the summary network's output, so num_summaries = dₜ + 1
    dat = DataAndSummaries(Z, rand32(1, K))
    est = PointEstimator(mknet(), d; num_summaries = dₜ + 1)
    est = train(est, θ, θ, dat, dat; kw...)
    @test size(estimate(est, dat; use_gpu = false)) == (d, K)

    # tuple input
    post = PosteriorEstimator(mknet(), NormalisingFlow(d, dₜ))
    post = train(post, θ, θ, Z, Z; kw...)
    @test size(sampleposterior(post, Z; N = 10, use_gpu = false)) == (d, 10, K)
end

@testset "DeepSet convenience constructor: $dvc" for dvc ∈ devices
    n = 10
    M = (3, 4)
    w = 32
    dₜ = 16
    d = 5
    Z = [rand32(n, m) for m ∈ M]
    ψ = Chain(Dense(n, w, relu), Dense(w, dₜ, relu))

    ds = DeepSet(ψ; latent_dim = dₜ, output_dim = d)
    y = ds(Z)
    @test size(y) == (d, length(M))
    testbackprop(ds, Z, dvc)

    ds = DeepSet(ψ; latent_dim = dₜ, output_dim = d, condition_on_sample_size = true)
    y = ds(Z)
    @test size(y) == (d, length(M))
    testbackprop(ds, Z, dvc)

    ds = DeepSet(ψ; latent_dim = dₜ, output_dim = d, width = 16)
    y = ds(Z)
    @test size(y) == (d, length(M))
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
        @test infer(estimator, Z) == estimate(estimator, Z)

        @testset "train" begin
            testbackprop(estimator, Z, dvc)
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,))
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), savepath = "testing-path")
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), simulate_just_in_time = true)
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,), freeze_summary_network = true)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, savepath = "testing-path")
            estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 2, epochs_per_refresh = 2, use_gpu = use_gpu, verbose = verbose, sampler_args = (d,))
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 4, epochs_per_refresh = 2, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 3, epochs_per_refresh = 1, simulate_just_in_time = true, use_gpu = use_gpu, verbose = verbose)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 1, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
            estimator = train(estimator, θ, θ, simulator, simulator_args = m, epochs = 4, epochs_per_refresh = 2, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
            Z_train = Z_val = simulator(θ, m)
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose, savepath = "testing-path")
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose)
            train(estimator, θ, θ, Z_train, Z_val; epochs = 1, use_gpu = use_gpu, verbose = verbose, freeze_summary_network = true)
            p = plotrisk()
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

            @test plot(assessment) isa Figure
        end

        @testset "bootstrap" begin
            # parametric bootstrap functions are designed for a single parameter configuration
            B = 40
            parameters = sampler(1)
            Z_sims = reduce(hcat, [simulator(parameters, m) for _ = 1:B])
            @test size(bootstrap(estimator, parameters, Z_sims; use_gpu = use_gpu)) == (d, B)
            @test size(bootstrap(estimator, parameters, simulator, m; B = B, use_gpu = use_gpu)) == (d, B)
        end
    end
end

@testset "IntervalEstimator" begin
    num_summaries = 3d
    summary_network = Chain(Dense(m, 64, relu), Dense(64, 64, relu), Dense(64, num_summaries))
    lower, upper = [-Inf, 0], [Inf, 1] # μ ∈ ℝ and σ ∈ (0, 1), the support of the prior given by sampler()
    estimator = IntervalEstimator(summary_network, d; num_summaries = num_summaries, lower = lower, upper = upper)
    show(devnull, estimator)

    # Forward pass
    θ̂ = estimator(Z)
    ci = interval(estimator, Z)
    @test size(θ̂) == (2d, K)
    @test all(lower .<= θ̂[1:d, :] .<= θ̂[(d + 1):end, :] .<= upper)
    @test all([size(c) == (d, 2) for c in ci])

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose)

    # Assessment
    assessment = assess(estimator, θ, Z)
    coverage(assessment)
    coverage(assessment; average_over_parameters = true)
    coverage(assessment; average_over_sample_sizes = false)
    coverage(assessment; average_over_parameters = true, average_over_sample_sizes = false)
    @test plot(assessment) isa Figure

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
    @test plot(assessment) isa Figure

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

@testset "Early stopping" begin
    # Gradient ascent (a negative learning rate) makes the validation risk increase, so training
    # must stop early. NB lr_schedule = nothing since learning-rate schedules reject negative learning rates.
    epochs, stopping_epochs = 20, 2
    Z_train = Z_val = simulator(θ, m)
    scenarios = (
        (args = (θ, θ, Z_train, Z_val), kwargs = (;)),
        (args = (θ, θ, simulator), kwargs = (simulator_args = m,)),
        (args = (sampler, simulator), kwargs = (simulator_args = m, K = K))
    )
    for scenario in scenarios, v in (false, true)
        savepath = mktempdir()
        estimator = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)))
        redirect_stdio(stdout = devnull, stderr = devnull) do
            train(estimator, scenario.args...; scenario.kwargs...,
                optimiser = Descent(-1.0f-1), lr_schedule = nothing,
                epochs = epochs, stopping_epochs = stopping_epochs, savepath = savepath, verbose = v)
        end
        @test size(loadrisk(savepath), 1) < epochs + 1 # initial risk + one row per completed epoch
    end
end

@testset "RatioEstimator" begin
    num_summaries = 3d
    summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    estimator = RatioEstimator(summary_network, d; num_summaries = num_summaries)

    # Forward pass
    r = estimator(Z, θ)
    @test size(r) == (1, K)

    # Training pairs: each data set is paired with its own parameters (label 1) and with independent parameters (label 0),
    # with the labels in the rows of the output, so that the training forward pass computes its summary statistics once
    est = RatioEstimator(summary_network, d; num_summaries = num_summaries, standardise_data = Standardiser(Z), standardise_parameters = Standardiser(θ.array))
    (Zs, θs, θ̃s), labels = NeuralEstimators._inputoutput(est, Z, θ.array)
    @test numobs(Zs) == numobs(θs) == numobs(θ̃s) == K
    @test labels == vcat(ones(Float32, 1, K), zeros(Float32, 1, K))
    @test sortslices(θ̃s; dims = 2) == sortslices(θs; dims = 2) # without a sampler, a permutation of the parameters...
    @test θ̃s != θs                                             # ...that pairs the data sets with other parameters
    ref =RatioEstimator(est.summary_network, est.summary_network_θ, est.inference_network; standardise_data = false, standardise_parameters = false)
    @test NeuralEstimators._mapstandardisers(NeuralEstimators._mark, est)((Zs, θs, θ̃s)) ≈ vcat(ref(Zs, θs), ref(Zs, θ̃s))
    @test est(Z, θ.array, θ.array[:, end:-1:1]) ≈ vcat(est(Z, θ.array), est(Z, θ.array[:, end:-1:1]))
    @test size(NeuralEstimators._inputoutput(RatioEstimator(summary_network, d; num_summaries, sampler), Z, θ.array)[1][3]) == size(θ) # with a sampler

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = false)

    # Data stored in an array with three or more dimensions (e.g., images for a CNN)
    simulator4(θ) = reshape(reduce(hcat, [θ[1, k] .+ θ[2, k] .* randn(Float32, 16) for k in 1:size(θ, 2)]), 4, 4, 1, :)
    cnn = Chain(Conv((2, 2), 1 => 4, relu), Flux.flatten, Dense(36, num_summaries))
    trained = train(RatioEstimator(cnn, d; num_summaries = num_summaries), sampler, simulator4; K = 64, epochs = 1, verbose = false)
    @test all(isfinite, trained(simulator4(θ), θ.array))

    # Inference (grid-based)
    grid = expandgrid(0:0.01:1, 0:0.01:1)'  # fine gridding of the parameter space
    z = getobs(Z, 1:1)
    logratio(estimator, z; grid = grid)                # log of likelihood-to-evidence ratios
    samples = sampleposterior(estimator, z; grid = grid)         # posterior sample
    @test size(samples) == (d, 1000, 1)
    seed!(1)
    samples1 = sampleposterior(estimator, z; grid = grid, N = 50)
    seed!(1)
    samples2 = infer(estimator, z; grid = grid, N = 50)
    @test samples1 == samples2

    # Assessment (grid-based)
    assessment = assess(estimator, θ, Z; grid = grid)

    # NUTS sampling (AdvancedHMC extension)
    lower, upper = [-3.0f0, 0.0f0], [3.0f0, 1.0f0]
    samples = sampleposterior(estimator, z; lower = lower, upper = upper, N = 30, warmup = 30)
    @test size(samples) == (d, 30, 1)
    @test all(lower .<= minimum(samples; dims = (2, 3))) && all(maximum(samples; dims = (2, 3)) .<= upper)
    samples = sampleposterior(estimator, getobs(Z, 1:2); lower = lower, upper = upper, N = 30, warmup = 30, logprior = θ -> -sum(abs2, θ))
    @test size(samples) == (d, 30, 2)
    @test_throws AssertionError sampleposterior(estimator, z; grid = grid, lower = lower, upper = upper)
    @test_throws AssertionError sampleposterior(estimator, z; lower = upper, upper = lower)
end

@testset "TelescopingRatioEstimator" begin
    num_summaries = 3d
    summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    estimator = TelescopingRatioEstimator(
        summary_network, d; num_summaries = num_summaries, sampler = sampler
    )

    # Forward pass: one logit per head
    r = estimator(Z, θ)
    @test size(r) == (d, K)

    # Training
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = false)

    lower, upper = [0.0f0, 0.0f0], [1.0f0, 1.0f0]
    grid = expandgrid(0:0.01:1, 0:0.01:1)'
    z = getobs(Z, 1:1)

    lr = logratio(estimator, z; grid = grid)
    @test size(lr) == (1, size(grid, 2))

    # Sequential Chebyshev sampling (default degree)
    samples = sampleposterior(estimator, z; lower = lower, upper = upper)
    @test size(samples) == (d, 1000, 1)
    @test all(lower .<= minimum(samples; dims = (2, 3)))
    @test all(maximum(samples; dims = (2, 3)) .<= upper)

    seed!(1)
    samples1 = sampleposterior(estimator, z; lower = lower, upper = upper, N = 50)
    seed!(1)
    samples2 = infer(estimator, z; lower = lower, upper = upper, N = 50)
    @test samples1 == samples2

    lp = logposterior(estimator, grid, z; lower = lower, upper = upper)  # default :raw
    @test size(lp) == (size(grid, 2),)

    # Normalised log-density of the sequential Chebyshev approximation, including points outside the box
    θpoints = hcat(grid[:, 1:10], [2.0f0, 0.5f0])
    lp = logposterior(estimator, θpoints, z; lower = lower, upper = upper, method = :chebyshev, degree = 16)
    @test size(lp) == (11,)
    @test all(isfinite, lp[1:10])
    @test lp[11] == -Inf
    logpriors = [x -> -x^2, x -> zero(x)]
    lp = logposterior(estimator, θpoints, getobs(Z, 1:2); lower = lower, upper = upper, method = :chebyshev, degree = 16, logpriors = logpriors)
    @test length(lp) == 2
    lp = logposterior(estimator, θpoints, getobs(Z, 1:2); lower = lower, upper = upper, logpriors = logpriors)
    @test length(lp) == 2 && lp[1][11] == -Inf
    @test_throws AssertionError logposterior(estimator, θpoints, z; lower = lower, upper = upper, method = :other)

    assessment = assess(estimator, θ, Z; lower = lower, upper = upper)
end

@testset "TelescopingRatioEstimator: concentrated posteriors" begin
    # Gaussian conditionals much narrower than the box [-4, 4]², given to the sampler as head logits (the parameters
    # are not standardised, so the rows of the head inputs hold the summary z of Z followed by θ₁, ..., θᵢ):
    # θ₁ | Z ~ N(z, σ₁²) and θ₂ | θ₁, Z ~ N(θ₁, σ₂²)
    σ₁, σ₂ = 0.003f0, 0.002f0
    estimator = TelescopingRatioEstimator(2; num_summaries = 1, sampler = K -> rand(Float32, 2, K), standardise_parameters = false)
    headfun(i, X) = i == 1 ? -(X[2:2, :] .- X[1:1, :]) .^ 2 ./ 2σ₁^2 : -(X[3:3, :] .- X[2:2, :]) .^ 2 ./ 2σ₂^2
    tzs = Float32[0.3 -1.2]  # two data sets
    lower, upper = [-4, -4], [4, 4]
    N = 5000
    seed!(1)
    θ = first(NeuralEstimators._sequential_core(estimator, headfun, tzs, lower, upper, N, nothing, 128, nothing))
    for (k, μ) in enumerate(tzs)
        θ₁, θ₂ = θ[1, (k - 1) * N .+ (1:N)], θ[2, (k - 1) * N .+ (1:N)]
        @test abs(mean(θ₁) - μ) < 0.1σ₁
        @test std(θ₁) ≈ σ₁ rtol = 0.05
        @test abs(mean(θ₂ .- θ₁)) < 0.1σ₂
        @test std(θ₂ .- θ₁) ≈ σ₂ rtol = 0.05
    end

    # Normalised log density: exact near the mode, and -Inf where the sampler assigns no mass
    θpoints = Float32[0.3 0.303 0.0 0.3; 0.3 0.301 0.0 0.5]
    lq = last(NeuralEstimators._sequential_core(estimator, headfun, tzs[:, 1:1], lower, upper, 0, θpoints, 128, nothing; logq = true))
    lmode = -log(2π * σ₁ * σ₂)
    @test lq[1:2] ≈ [lmode, lmode - 1] atol = 0.01
    @test lq[3:4] == [-Inf, -Inf]

    # Mass at a bound: the zoomed interval keeps the bound exactly, so that the draws stay in the box and the log density
    # at the bound is finite (recomputing the bound from the midpoint and half-width rounds -1.07f0 outwards, 0.05f0 inwards)
    estimator₁ = TelescopingRatioEstimator(1; num_summaries = 1, sampler = K -> rand(Float32, 1, K), standardise_parameters = false)
    exponential(_, X) = -(X[2:2, :] .- X[1:1, :]) ./ 1f-6  # θ | Z ~ z + Exp(10⁶)
    θ = first(NeuralEstimators._sequential_core(estimator₁, exponential, Float32[-1.07;;], [-1.07], [3.69], N, nothing, 128, nothing))
    @test all(θ .>= -1.07f0)
    lq = last(NeuralEstimators._sequential_core(estimator₁, exponential, Float32[0.05;;], [0.05], [0.5], 0, Float32[0.05;;], 128, nothing; logq = true))
    @test isfinite(only(lq))

    # A NaN or Inf from logpriors (here at the lower bound) is an error rather than silently corrupting the draws
    @test_throws AssertionError NeuralEstimators._sequential_core(estimator, headfun, tzs, [0, 0], [1, 1], 10, nothing, 128, [x -> 0 * log(x), x -> zero(x)])
    @test_throws AssertionError NeuralEstimators._sequential_core(estimator, headfun, tzs, [0, 0], [1, 1], 10, nothing, 128, [x -> -log(x) / 2, x -> zero(x)])
end

@testset "Chebyshev approximation" begin
    a, b = -1.0, 2.0
    plan = ChebPlan(a, b; degree = 32)
    @test plan.nodes == chebnodes(32, a, b)
    @test first(plan.nodes) ≈ b && last(plan.nodes) ≈ a
    @test default_bisection_iters(Float64) == 53
    @test default_bisection_iters(Float32) == 24

    # Two test densities on [a, b] with closed-form integrals and inverse CDFs: exp(x) and the uniform density
    F = hcat(exp.(plan.nodes), ones(length(plan.nodes)))
    masses = [exp(b) - exp(a), b - a]
    invcdfs = (u -> log(exp(a) + u * masses[1]), u -> a + u * masses[2])

    # Fitting and evaluation
    c = chebfit(plan, F[:, 1])
    x = collect(range(a, b; length = 25))
    @test chebval_ab(x, c, a, b) ≈ exp.(x)
    C = chebfit(plan, F)
    @test chebval_ab_batched([0.5, 0.5], C, a, b) ≈ [exp(0.5), 1.0]
    X = hcat(x, x)
    @test chebval_ab_batched(X, C, a, b) ≈ hcat(exp.(x), ones(length(x)))
    @test_throws DimensionMismatch chebval_ab_batched(hcat(X, x), C, a, b)

    # Integration
    @test chebint_ab(c, a, b) ≈ chebintegrate(plan, c)
    @test chebintmatrix(32, a, b) ≈ plan.Mint
    @test chebintegral(plan, c) ≈ masses[1]
    @test chebintegral(plan, C) ≈ masses
    CI = chebintegrate(plan, C)
    @test chebdefinite(CI, a, b) ≈ masses

    # Normalised log-density: one point per density (vector) and several points per density (matrix)
    @test cheblogq([0.5, 0.5], C, masses, a, b) ≈ [0.5 - log(masses[1]), -log(masses[2])]
    @test cheblogq(X, C, masses, a, b) ≈ hcat(x .- log(masses[1]), fill(-log(masses[2]), length(x)))

    # Inverse-CDF sampling: two uniforms per density, then one per density
    u = [0.1, 0.8, 0.3, 0.6]
    @test invert_cdf_batched(CI, a, b, u) ≈ [invcdfs[1](0.1), invcdfs[1](0.8), invcdfs[2](0.3), invcdfs[2](0.6)]
    @test_throws DimensionMismatch invert_cdf_batched(CI, a, b, u[1:3])
    @test chebsample(plan, F, [0.25, 0.75]) ≈ [invcdfs[1](0.25), invcdfs[2](0.75)]
    @test_throws DimensionMismatch chebsample(plan, F, [0.25, 0.5, 0.75])
    U = [0.1 0.3; 0.8 0.6]
    @test chebsample(plan, F, U) ≈ [invcdfs[1](0.1) invcdfs[2](0.3); invcdfs[1](0.8) invcdfs[2](0.6)]
    @test_throws DimensionMismatch chebsample(plan, F, [U U])

    # Moving the plan to another array type
    plan32 = ChebPlan(a, b; degree = 32, T = Float32)
    @test ChebPlan_on(Array, plan32).D == plan32.D
    if CUDA.functional()
        gpu_plan = ChebPlan_on(CuArray, plan32)
        @test gpu_plan.D isa CuArray
        samples = Array(chebsample(gpu_plan, CuArray(Float32.(F)), CuArray(Float32[0.25, 0.75])))
        @test samples ≈ Float32[invcdfs[1](0.25), invcdfs[2](0.75)] rtol = 1e-4
    end
end

@testset "PosteriorEstimator" begin
    for approxdist in [NormalisingFlow, GaussianMixture, Gaussian]
        num_summaries = 3d
        summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
        q = approxdist(d, num_summaries)
        estimator = PosteriorEstimator(summary_network, d; num_summaries = num_summaries, q = approxdist) # convenience constructor
        estimator = PosteriorEstimator(summary_network, q)
        estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, verbose = false)
        @test numdistributionalparams(estimator) == numdistributionalparams(q)
        samples = sampleposterior(estimator, Z) # posterior draws
        @test size(samples) == (d, 1000, K)
        seed!(1)
        samples1 = sampleposterior(estimator, Z; N = 50)
        seed!(1)
        samples2 = infer(estimator, Z; N = 50)
        @test samples1 == samples2
        posteriormean(estimator, Z)   # point estimate
        posteriormedian(estimator, Z) # point estimate
        posteriorquantile(estimator, Z, [0.1, 0.5]) # quantiles
        assessment = assess(estimator, θ, Z)
        @test plot(assessment) isa Figure
    end
end

@testset "plot(assessment)" begin
    # Point estimates: one estimator, then several estimators in the same panels or in rows of their own
    point = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)))
    assessment = assess(point, θ, Z; use_gpu = false)
    @test plot(assessment; plots = :recovery, ncols = 1, figure = (; size = (400, 600)), axis = (; xgridvisible = false)) isa Figure
    @test_throws ArgumentError plot(assessment; plots = :ecdf)
    merged = assess([point, point], θ, Z; estimator_names = ["A", "B"], use_gpu = false, verbose = false)
    @test plot(merged) isa Figure
    @test plot(merged; grid = true) isa Figure
    @test plot(assessment; transpose = true) isa Figure

    # Posterior samples: a subset of the plots, and several estimators
    posterior = PosteriorEstimator(Chain(Dense(m, 16, gelu), Dense(16, 3d)), d; num_summaries = 3d, q = Gaussian)
    assessment = assess(posterior, θ, Z; N = 100, use_gpu = false)
    @test plot(assessment; plots = (:ecdf, :recovery), difference = false, prob = 0.9) isa Figure
    @test plot(assessment; plots = "zscore") isa Figure
    @test_throws ArgumentError plot(assessment; plots = :calibration)
    @test_throws ArgumentError plot(assessment; plots = Symbol[])
    merged = assess([posterior, posterior], θ, Z; estimator_names = ["A", "B"], N = 100, use_gpu = false, verbose = false)
    @test plot(merged) isa Figure
    @test plot(merged; grid = true) isa Figure

    # Layout: by default one row of panels per plot (3 × d); transposed, one row per parameter (d × 3)
    fig = plot(assessment)
    width, height = size(fig.scene)
    @test count(block -> block isa Axis, fig.content) == 3d
    @test width < height
    fig = plot(assessment; transpose = true)
    width, height = size(fig.scene)
    @test count(block -> block isa Axis, fig.content) == 3d
    @test width > height
    @test plot(assessment; transpose = false) isa Figure
    @test plot(merged; transpose = true, grid = true) isa Figure

    # Learned distributions (PriorEstimator), with and without test samples from the true distribution. The
    # default layout depends on the number of parameters: a histogram (d = 1), a row of two histograms and a
    # scatter plot (d = 2), and a pairs plot (d ≥ 3), whose upper triangle is empty without test samples or
    # with `overlay = true`
    naxes(fig) = count(block -> block isa Axis, fig.content)
    for (dd, npanels) in ((1, 1), (2, 3), (3, 9))
        θ_prior = randn(Float32, dd, 100)
        prior = train(PriorEstimator(dd; depth = 1, width = 16), θ_prior, θ_prior; epochs = 1, verbose = false, use_gpu = false, savepath = nothing)
        prior_assessment = assess(prior, θ_prior; N = 200, use_gpu = false)
        @test isnothing(prior_assessment.estimates)
        @test isfinite(risk(prior_assessment))
        @test nrow(prior_assessment.samples) == dd * (200 + 100)
        @test Set(prior_assessment.samples.distribution) == Set(["learned", "true"])
        learned = assess(prior; N = 200, use_gpu = false)
        @test isnothing(risk(learned))
        @test all(==("learned"), learned.samples.distribution)
        corner = dd >= 3 ? dd * (dd + 1) ÷ 2 : npanels
        @test naxes(plot(prior_assessment)) == npanels
        @test naxes(plot(prior_assessment; overlay = true)) == corner
        @test naxes(plot(learned)) == corner
        @test naxes(plot(prior_assessment; plots = :marginals)) == dd
        @test_throws ArgumentError plot(prior_assessment; plots = :ecdf)
        @test_throws ArgumentError bias(prior_assessment)
    end
    @test_throws ArgumentError assess(PriorEstimator(2; depth = 1, width = 16), θ, Z)

    # Intervals are drawn behind the point estimates of the recovery plot only on request
    rangebars(fig) = sum(count(p -> p isa Rangebars, block.scene.plots) for block in fig.content if block isa Axis)
    @test rangebars(plot(assessment; plots = :recovery)) == 0
    @test rangebars(plot(assessment; plots = :recovery, intervals = true)) == d
    @test plot(merged; intervals = true) isa Figure

    # Simultaneous confidence band for the ECDF: its coverage probability against an enumeration of all outcomes,
    # and the band attaining the nominal coverage without exceeding it by much
    ext = Base.get_extension(NeuralEstimators, :NeuralEstimatorsPlottingMakieExt)
    n, p = 5, [0.1, 0.4, 0.75, 1.0]
    lower, upper = [0, 0, 2, n], [1, 4, 4, n]
    cells = diff([0; p])
    enumerated = sum(Iterators.product(ntuple(_ -> eachindex(p), n)...)) do assignment
        counts = cumsum([count(==(i), assignment) for i in eachindex(p)])
        all(lower .<= counts .<= upper) ? prod(cells[collect(assignment)]) : 0.0
    end
    @test ext._coverage(n, p, lower, upper) ≈ enumerated
    n, p = 100, (1:100) ./ 100
    lower, upper = ext._ecdfband(n, p, 0.95)
    @test all(lower .<= upper) && lower[end] == upper[end] == n
    @test 0.95 <= ext._coverage(n, p, lower, upper) < 0.96
end

@testset "assess(θ, θ̂)" begin
    # Point estimates: the methods for estimators call assess(θ, θ̂) once the estimator has been applied to the data
    point = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)))
    θ̂ = estimate(point, Z; use_gpu = false)
    @test assess(point, θ, Z; use_gpu = false).estimates == assess(θ, θ̂).estimates
    assessment = assess(θ, θ̂; estimator_name = "other", m = fill(m, K))
    @test ismissing(only(assessment.runtime.runtime))
    @test all(==(m), assessment.estimates.m)
    @test nrow(rmse(assessment; average_over_sample_sizes = false)) == d

    # θ is recycled when there are several data sets for each parameter vector
    assessment = assess(θ, hcat(θ̂, θ̂))
    @test sort(unique(assessment.estimates.j)) == [1, 2]
    @test filter(:j => ==(1), assessment.estimates).truth == filter(:j => ==(2), assessment.estimates).truth
    @test_throws AssertionError assess(θ, θ̂[:, 1:(K - 1)])

    # Posterior samples, compared with those of a neural posterior estimator
    N = 100
    samples = randn(Float32, d, N, K)
    assessment = assess(θ, samples; pointsummary = median, estimator_name = "other", runtime = 1.0)
    @test nrow(assessment.samples) == d * N * K
    @test assessment.estimates.estimate ≈ vec(permutedims(dropdims(median(samples; dims = 2); dims = 2)))
    @test plot(assessment) isa Figure
    posterior = PosteriorEstimator(Chain(Dense(m, 16, gelu), Dense(16, 3d)), d; num_summaries = 3d, q = Gaussian)
    merged = merge(assessment, assess(posterior, θ, Z; N = N, estimator_name = "NPE", use_gpu = false))
    @test Set(merged.samples.estimator) == Set(["other", "NPE"])
    @test plot(merged) isa Figure

    # Functions of the parameters are assessed as further parameters
    total = "μ + σ" => x -> x[1] + x[2]
    assessment = assess(θ, samples; functions = [total])
    @test unique(assessment.estimates.parameter) == ["μ", "σ", "μ + σ"]
    @test filter(:parameter => ==("μ + σ"), assessment.estimates).truth ≈ vec(sum(convert(Matrix, θ); dims = 1))
    @test filter(:parameter => ==("μ + σ"), assessment.samples).value ≈ vec(sum(samples; dims = 1))
    @test plot(assessment) isa Figure
    @test "μ + σ" ∈ assess(posterior, θ, Z; N = N, functions = [total], use_gpu = false).estimates.parameter
    @test assess(θ, samples; functions = total).samples == assess(θ, samples; functions = [total]).samples   # a single function need not be wrapped
    @test assess(θ, samples; functions = "μσ²").samples == assess(θ, samples; functions = ["μσ²"]).samples
    @test "μσ²" ∈ assess(posterior, θ, Z; N = N, functions = "μσ²", use_gpu = false).estimates.parameter

    # ... indexed by name, or given as strings in the parameter names, in which superscripts denote powers and adjacent names products
    assessment = assess(θ, samples; functions = ["μσ" => x -> x["μ"] * x["σ"], "μ·σ²", "μσ²", "log(σ²)"])
    @test filter(:parameter => ==("μσ"), assessment.samples).value ≈ vec(samples[1, :, :] .* samples[2, :, :])
    @test filter(:parameter => ==("μ·σ²"), assessment.samples).value ≈ vec(samples[1, :, :] .* samples[2, :, :] .^ 2)
    @test filter(:parameter => ==("μσ²"), assessment.samples).value == filter(:parameter => ==("μ·σ²"), assessment.samples).value
    @test filter(:parameter => ==("log(σ²)"), assessment.samples).value ≈ vec(log.(samples[2, :, :] .^ 2))
    @test filter(:parameter => ==("μ·σ²"), assessment.estimates).truth ≈ θ["μ", :] .* θ["σ", :] .^ 2
    squared = assess(abs2.(convert(Matrix, θ)), abs2.(samples); parameter_names = ["μ²", "σ²"], functions = ["σ"])
    @test filter(:parameter => ==("σ"), squared.samples).value ≈ vec(abs.(samples[2, :, :]))
    @test_throws ArgumentError assess(θ, samples; functions = ["μ + ν"])

    # An estimator cannot be assessed without data
    @test_throws MethodError assess(point, θ)
end

@testset "PosteriorEstimator: SpikeAndSlab" begin
    d1 = 1
    num_summaries = 6

    # Univariate spike-and-slab prior: spike at 0 with prob 0.5, else continuous
    sampler1(K, d = nothing) = NamedMatrix(θ = Float32.(rand(K) .< 0.5) .* randn(Float32, K))
    simulator1(θ::AbstractVector, m::Integer) = θ["θ"] .+ sort(randn(Float32, m))
    simulator1(θ::AbstractMatrix, m::Integer) = reduce(hcat, simulator1.(eachcol(θ), m))
    θ1 = sampler1(K)
    Z1 = simulator1(θ1, m)

    summary_network = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    q = SpikeAndSlab(d1, num_summaries) # convenience constructor (default slab = GaussianMixture)
    @test numdistributionalparams(q) == 1 + numdistributionalparams(q.slab)

    # Density evaluation, including spike (θ == 0) and slab (θ != 0) entries
    θ_plain = reshape(Float32[0, 0.5, -0.3, 0, 1.2], 1, :)
    tz = randn(Float32, num_summaries, size(θ_plain, 2))
    dens = _logdensity(q, θ_plain, tz)
    @test size(dens) == (1, size(θ_plain, 2))
    @test all(isfinite, dens)

    estimator = PosteriorEstimator(summary_network, d1; num_summaries = num_summaries, q = SpikeAndSlab)
    estimator = PosteriorEstimator(summary_network, q)
    @test numdistributionalparams(estimator) == numdistributionalparams(q)
    estimator = train(estimator, sampler1, simulator1, simulator_args = m, epochs = 1, verbose = false)
    samples = sampleposterior(estimator, Z1) # posterior draws
    @test size(samples) == (d1, 1000, K)
    posteriormean(estimator, Z1)

    # spikeprobability: vector for multiple data sets, scalar for a single data set
    sp = spikeprobability(estimator, Z1)
    @test length(sp) == K
    @test all(0 .<= sp .<= 1)
    sp1 = spikeprobability(estimator, simulator1(sampler1(1), m))
    @test sp1 isa Real
    @test 0 <= sp1 <= 1

    # Custom slab type with non-identity transform/invtransform (positive-support slab)
    sampler2(K, d = nothing) = NamedMatrix(θ = Float32.(rand(K) .< 0.5) .* (rand(Float32, K) .+ 0.5f0))
    simulator2(θ::AbstractVector, m::Integer) = θ["θ"] .+ sort(randn(Float32, m))
    simulator2(θ::AbstractMatrix, m::Integer) = reduce(hcat, simulator2.(eachcol(θ), m))
    Z2 = simulator2(sampler2(K), m)

    q2 = SpikeAndSlab(d1, num_summaries; slab = NormalisingFlow, transform = log, invtransform = exp)
    @test numdistributionalparams(q2) == 1 + numdistributionalparams(q2.slab)
    estimator2 = PosteriorEstimator(summary_network, q2)
    estimator2 = train(estimator2, sampler2, simulator2, simulator_args = m, epochs = 1, verbose = false)
    samples2 = sampleposterior(estimator2, Z2)
    @test size(samples2) == (d1, 1000, K)
    @test all(samples2 .>= 0) # spike (0) or positive slab draws
end

@testset "PosteriorEstimator: parameter bounds" begin
    # scaledlogit and scaledlogistic with finite, one-sided, and infinite bounds
    x = collect(-10.0:0.5:10.0)
    for (a, b) in ((0.0, 1.0), (2.0, 7.0), (0.0, Inf), (-Inf, 1.0), (-Inf, Inf))
        y = scaledlogistic.(x, a, b)
        @test all(a .< y .< b)
        @test scaledlogit.(y, a, b) ≈ x
        @test @inferred(scaledlogit(y[1], a, b)) isa Float64
        @test @inferred(scaledlogistic(0.5, a, b)) isa Float64
        @test @inferred(scaledlogit(Float32(y[1]), Float32(a), Float32(b))) isa Float32
        @test @inferred(scaledlogistic(0.5f0, a, b)) isa Float64 # computed in the precision of the bounds
        # A value on a finite bound is mapped to a finite value
        isfinite(a) && @test isfinite(scaledlogit(a, a, b))
        isfinite(b) && @test isfinite(scaledlogit(b, a, b))
    end
    L = log(1 / eps(Float32) - 1)
    @test scaledlogit(0.0, 0.0, 1.0) ≈ -L
    @test scaledlogit(1, 0, 1) ≈ L
    @test scaledlogit(prevfloat(2.0), 2.0, 7.0) ≈ -L # beyond a bound by a rounding error
    @test scaledlogit(1.0000003f-5, 1e-6, 1e-5) ≈ L # e.g., a Float32 log-uniform draw beyond the upper bound through rounding
    @test scaledlogit(1e-10, 0.0, 1.0) ≈ log(1e-10 / (1 - 1e-10)) # values inside the bounds are mapped exactly, even close to a bound
    @test_throws DomainError scaledlogit(1.9, 2.0, 7.0)
    @test_throws DomainError scaledlogit(-0.1, 0.0, Inf)
    @test isnan(scaledlogit(NaN, 0.0, 1.0))
    @test Flux.gradient(t -> scaledlogit(t, 0.0, 1.0), 0.0)[1] == 0 # finite at a bound
    @test scaledlogistic(40.0, -0.1, 0.2) <= 0.2 # without clamping, rounding gives 0.20000000000000004

    # μ ∈ ℝ and σ ∈ (0, 1), the support of the prior given by sampler()
    lower, upper = [-Inf, 0], [Inf, 1]
    num_summaries = 3d
    network() = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    for approxdist in [NormalisingFlow, GaussianMixture, Gaussian]
        estimator = PosteriorEstimator(network(), d; num_summaries = num_summaries, q = approxdist, lower = lower, upper = upper)
        @test PosteriorEstimator(network(), approxdist(d, num_summaries); lower = lower, upper = upper) isa PosteriorEstimator
        @test PosteriorEstimator(approxdist(d, num_summaries), network(); lower = lower, upper = upper) isa PosteriorEstimator

        # The parameters are transformed before training
        θ_plain = θ.array
        input, _ = NeuralEstimators._inputoutput(estimator, Z, θ_plain)
        @test input[2] ≈ scaledlogit.(θ_plain, lower, upper)
        @test estimator.invtransform(input[2]) ≈ θ_plain

        # The posterior draws, and summaries thereof, lie within the bounds
        estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, K = 100, verbose = false)
        samples = sampleposterior(estimator, Z)
        @test size(samples) == (d, 1000, K)
        @test all(lower .<= samples .<= upper)
        @test all(lower .<= posteriormean(estimator, Z) .<= upper)
        @test all(lower .<= posteriorquantile(estimator, Z, [0.1, 0.9]) .<= upper)
        @test size(sampleposterior(estimator, Z; N = 0)) == (d, 0, K)
        @test assess(estimator, θ, Z) isa Assessment
    end

    # A parameter beyond a bound by more than a rounding error is an error
    estimator = PosteriorEstimator(network(), d; num_summaries = num_summaries, lower = lower, upper = upper)
    @test_throws DomainError NeuralEstimators._inputoutput(estimator, Z, [0.0 0.0; 0.5 1.5])

    # Training parameters on a bound (e.g., rand(Float32) draws equal to zero) give finite risks
    estimator = PosteriorEstimator(network(), d; num_summaries = num_summaries, q = GaussianMixture, lower = lower, upper = upper)
    θ_train, θ_val = sampler(100), sampler(50)
    θ_train[2, 1:2] = [0, 1]
    θ_val[2, 1] = 0
    savepath = mktempdir()
    trained = train(estimator, θ_train, θ_val, simulator, simulator_args = m, epochs = 2, savepath = savepath, verbose = false)
    @test all(isfinite, loadrisk(savepath))

    # The bounds are saved with the estimator, so it is loaded into an estimator constructed without them
    @test keys(Flux.state(estimator)) == (:summary_network, :q)
    loaded = loadestimator(PosteriorEstimator(network(), d; num_summaries = num_summaries, q = GaussianMixture), savepath)
    @test loaded.transform isa NeuralEstimators._ScaledLogit && loaded.transform.lower == lower && loaded.transform.upper == upper
    seed!(1)
    samples1 = sampleposterior(trained, Z; N = 50, use_gpu = false)
    seed!(1)
    samples2 = sampleposterior(loaded, Z; N = 50, use_gpu = false)
    @test samples1 == samples2

    # A custom transformation: the log of σ + 0.1, so that σ > -0.1
    transform(θ) = vcat(θ[1:1, :], log.(θ[2:2, :] .+ 0.1f0))
    invtransform(θ) = vcat(θ[1:1, :], exp.(θ[2:2, :]) .- 0.1f0)
    estimator = PosteriorEstimator(network(), d; num_summaries = num_summaries, q = GaussianMixture, transform = transform, invtransform = invtransform)
    estimator = train(estimator, sampler, simulator, simulator_args = m, epochs = 1, K = 100, verbose = false)
    @test all(sampleposterior(estimator, Z)[2, :, :] .> -0.1f0)

    # Invalid specifications
    @test_throws ArgumentError PosteriorEstimator(network(), d; num_summaries = num_summaries, lower = lower, transform = transform, invtransform = invtransform)
    @test_throws ArgumentError PosteriorEstimator(network(), d; num_summaries = num_summaries, transform = transform)
    @test_throws AssertionError PosteriorEstimator(network(), d; num_summaries = num_summaries, lower = [0, 1], upper = [1, 1])
    @test_throws ArgumentError PosteriorEstimator(network(), 1; num_summaries = num_summaries, q = SpikeAndSlab, lower = 0)
    # NB a transformation of the slab of a SpikeAndSlab is given to the SpikeAndSlab itself
    @test_throws ArgumentError PosteriorEstimator(network(), 1; num_summaries = num_summaries, q = SpikeAndSlab, transform = θ -> log.(θ), invtransform = θ -> exp.(θ))
end

@testset "Neural Bayes estimators: parameter bounds" begin
    using NeuralEstimators: _ScaledLogit, _ScaledLogistic, _applybounds, _homogeneity, _fitunfitted, _mapstandardisers, _mark, _inputoutput, _unstandardise, _repeatrows, _rows, MetalDevice
    lower, upper = [-Inf, 0], [Inf, 1] # μ ∈ ℝ and σ ∈ (0, 1), the support of the prior given by sampler()
    num_summaries = 3d
    network() = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))
    θ_train, θ_val = sampler(100), sampler(50)
    Z_train, Z_val = simulator(θ_train, m), simulator(θ_val, m)

    # The bounds give the maps to and from the real line, which are not part of the functors
    for est in (PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)); lower, upper), PointEstimator(network(), d; num_summaries, lower, upper), IntervalEstimator(network(), d; num_summaries, lower, upper), QuantileEstimator(network(), d; num_summaries, lower, upper))
        @test est.transform isa _ScaledLogit && est.invtransform isa _ScaledLogistic
        @test est.transform.lower == lower && est.transform.upper == upper
    end
    @test PointEstimator(network(), d; num_summaries).transform === identity
    @test PointEstimator(network(), d; num_summaries, lower = [0]).transform.lower === 0.0 # a single bound applies to every parameter
    @test keys(Flux.state(PointEstimator(network(), d; num_summaries, lower, upper))) == (:summary_network, :inference_network)
    @test_throws AssertionError PointEstimator(network(), d; num_summaries, lower = [0, 1], upper = [1, 1])
    @test_throws ArgumentError IntervalEstimator(network(), d; num_summaries, c = Compress([0, 0], [1, 1])) # removed: use lower and upper

    # On another device (e.g., a GPU, or traced by Reactant), the bounds of each row are fixed in a closure, with the same result as on the host
    x = randn(2, 7)
    for (a, b) in ((lower, upper), (0.0, 1.0), (0.0, Inf), (-Inf, 1.0), ([-1.0, -Inf], [1.0, 2.0]))
        @test _applybounds(scaledlogistic, x, a, b, Any) == _applybounds(scaledlogistic, x, a, b)
        y = _applybounds(scaledlogistic, x, a, b)
        @test _applybounds(scaledlogit, y, a, b, Any) == _applybounds(scaledlogit, y, a, b)
        # On a GPU that may not support double precision (e.g., Metal), the bounds are applied in single precision
        y32 = _applybounds(scaledlogistic, Float32.(x), a, b, MetalDevice)
        @test y32 isa Matrix{Float32} && y32 ≈ y
    end

    # ---- Conformance: each estimator equals the same networks without bounds or standardisation, applied to standardised
    # data, with the outputs unstandardised and mapped back by scaledlogistic (for each probability level) ----
    sZ = Standardiser(Z_train)
    sθ = Standardiser(scaledlogit.(θ_train.array, lower, upper)) # the transformed parameters are standardised
    Zs = (Z .- sZ.mean) ./ sZ.std
    back(y) = scaledlogistic.(_unstandardise(sθ, y), lower, upper)
    unmapped = (standardise_data = false, standardise_parameters = false)
    est = PointEstimator(network(), d; num_summaries, lower, upper, standardise_data = sZ, standardise_parameters = sθ)
    ref = PointEstimator(est.summary_network, est.inference_network; unmapped...)
    @test est(Z) ≈ back(ref(Zs))
    # NB the estimates are mapped back in double precision, but computed by networks in single precision, and estimate()
    # applies them in batches, which can change the rounding (e.g., on Linux with OpenBLAS); hence the tolerance of single precision
    @test estimate(est, Z; use_gpu = false) ≈ est(Z) rtol = 1e-5
    @test _mapstandardisers(_mark, est)(Zs) ≈ ref(Zs) # the estimator being trained outputs the transformed and standardised scale
    @test _inputoutput(est, Z, θ.array)[2] ≈ (scaledlogit.(θ.array, lower, upper) .- sθ.mean) ./ sθ.std
    est = IntervalEstimator(network(), d; num_summaries, lower, upper, standardise_data = sZ, standardise_parameters = sθ)
    ref = IntervalEstimator(est.summary_network, est.u, est.v, est.probs, est.g; unmapped...)
    out = ref(Zs)
    @test est(Z) ≈ vcat(back(out[1:d, :]), back(out[(d + 1):end, :]))
    @test _mapstandardisers(_mark, est)(Zs) ≈ out
    est = QuantileEstimator(network(), d; num_summaries, lower, upper, standardise_data = sZ, standardise_parameters = sθ)
    ref = QuantileEstimator(est.summary_network, est.summary_network_θ, est.v, est.probs, est.g, est.i; unmapped...)
    out = ref(Zs)
    @test est(Z) ≈ reduce(vcat, [back(out[(1:d) .+ (t - 1) * d, :]) for t in 1:3])
    @test _mapstandardisers(_mark, est)(Zs) ≈ out
    θs = (scaledlogit.(θ.array, lower, upper) .- sθ.mean) ./ sθ.std # the parameters on the scale of training
    @test _inputoutput(est, Z, θ.array)[2] ≈ θs
    input, output = _inputoutput(QuantileEstimator(network(), d; num_summaries, i = 1, lower, upper, standardise_data = sZ, standardise_parameters = sθ), Z, θ.array)
    @test input[2] ≈ θs[2:2, :] && output ≈ θs[1:1, :]
    # Full conditionals: θ₋ᵢ is mapped to the real line and standardised, and the quantiles of θᵢ are mapped back
    est = QuantileEstimator(network(), d; num_summaries, i = 1, lower, upper, standardise_data = sZ, standardise_parameters = sθ)
    ref = QuantileEstimator(est.summary_network, est.summary_network_θ, est.v, est.probs, est.g, est.i; unmapped...)
    σ = θ.array[2:2, :]
    σs = (scaledlogit.(σ, 0, 1) .- sθ.mean[2:2, :]) ./ sθ.std[2:2, :]
    @test est(Z, σ) ≈ _unstandardise(_repeatrows(_rows(sθ, 1:1), 3), ref(Zs, σs)) # μ is unbounded
    est = QuantileEstimator(network(), d; num_summaries, i = 2, lower, upper, standardise_data = sZ, standardise_parameters = sθ)
    ref = QuantileEstimator(est.summary_network, est.summary_network_θ, est.v, est.probs, est.g, est.i; unmapped...)
    μ = θ.array[1:1, :]
    μs = (μ .- sθ.mean[1:1, :]) ./ sθ.std[1:1, :]
    @test est(Z, μ) ≈ scaledlogistic.(_unstandardise(_repeatrows(_rows(sθ, 2:2), 3), ref(Zs, μs)), 0, 1)
    est = QuantileEstimator(network(), d; num_summaries, i = 1, lower = [-Inf, 0], upper = [Inf, 10])
    @test size(est(Z, rand(1:9, 1, K))) == (3, K) # integer θ₋ᵢ
    # Without standardisation, the bounds are still applied (the forward passes that apply no map are those of training)
    for est in (PointEstimator(network(), d; num_summaries, lower, upper, unmapped...), IntervalEstimator(network(), d; num_summaries, lower, upper, unmapped...), QuantileEstimator(network(), d; num_summaries, lower, upper, unmapped...))
        y = est(Z)
        @test all(0 .<= y[2:d:end, :] .<= 1)
        @test _mapstandardisers(_mark, est)(Z) ≈ scaledlogit.(y, repeat(lower, size(y, 1) ÷ d), repeat(upper, size(y, 1) ÷ d)) rtol = 1e-4
    end

    # ---- train() warns if the loss is one for which the bounds of a PointEstimator change the Bayes estimator: any loss
    # whose Bayes estimators are not posterior quantiles. The check is that of _invariantloss, with homogeneity of degree one ----
    s = Standardiser(θ_train.array)
    @test _homogeneity(NeuralEstimators.mae, s, θ_train.array) ≈ 2 rtol = 1e-3
    @test _homogeneity((x, y) -> quantileloss(x, y, 0.3), s, θ_train.array) ≈ 2 rtol = 1e-3
    @test _homogeneity(NeuralEstimators.mse, s, θ_train.array) ≈ 4 rtol = 1e-3
    @test isnothing(_homogeneity((x, y) -> tanhloss(x, y, 0.1), s, θ_train.array))
    @test isnothing(_homogeneity((x, y) -> sum(sqrt.(mean(abs2, x .- y; dims = 2))), s, θ_train.array)) # not a mean over the parameter vectors
    fit(est, loss; verbose = false) = _fitunfitted(FluxTrainState(est, Adam(), Optimisers.setup(Adam(), est)), θ_train, Z_train, loss, verbose)
    bounded = PointEstimator(network(), d; num_summaries, lower, upper, standardise_parameters = true)
    @test_logs fit(bounded, NeuralEstimators.mae; verbose = true)
    @test_logs fit(bounded, (x, y) -> quantileloss(x, y, 0.1); verbose = true)
    @test_logs (:warn, r"within the neural network itself") fit(bounded, NeuralEstimators.mse)
    @test_logs (:warn, r"within the neural network itself") fit(bounded, (x, y) -> tanhloss(x, y, 0.1); verbose = true)
    unbounded = PointEstimator(network(), d; num_summaries, standardise_parameters = true) # NB constructed outside @test_logs, which would capture its @info
    @test_logs fit(unbounded, NeuralEstimators.mse; verbose = true)
    fitted = PointEstimator(network(), d; num_summaries, lower, upper, standardise_data = sZ, standardise_parameters = sθ) # nothing to fit: the warning is given at every call to train()
    @test_logs (:warn, r"within the neural network itself") fit(fitted, NeuralEstimators.mse)
    @test_logs (:warn, r"within the neural network itself") fit(bounded, (x, y) -> sum(sqrt.(mean(abs2, x .- y; dims = 2)))) # not a mean over the parameter vectors

    # ---- Training: the parameters are transformed before they are standardised, and the estimates lie within the bounds.
    # The bounds are saved with the estimator, so it is loaded into an estimator constructed without them ----
    for (est, arch) in ((PointEstimator(network(), d; num_summaries, lower, upper), () -> PointEstimator(network(), d; num_summaries)),
                        (IntervalEstimator(network(), d; num_summaries, lower, upper), () -> IntervalEstimator(network(), d; num_summaries)))
        savepath = mktempdir()
        trained = train(est, θ_train, θ_val, Z_train, Z_val; epochs = 2, savepath = savepath, verbose = false)
        @test trained.parameter_standardiser.mean ≈ Standardiser(scaledlogit.(θ_train.array, lower, upper)).mean
        @test trained.invtransform isa _ScaledLogistic # no longer marked
        θ̂ = estimate(trained, Z_val; use_gpu = false)
        @test all(0 .<= θ̂[2:d:end, :] .<= 1)
        loaded = loadestimator(arch(), savepath)
        @test loaded.transform isa _ScaledLogit && loaded.transform.lower == lower && loaded.transform.upper == upper
        @test estimate(loaded, Z_val; use_gpu = false) ≈ θ̂
    end
    trained = train(QuantileEstimator(network(), d; num_summaries, lower, upper), θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false)
    @test trained.parameter_standardiser.mean ≈ Standardiser(scaledlogit.(θ_train.array, lower, upper)).mean
    @test all(0 .<= estimate(trained, Z; use_gpu = false)[2:d:end, :] .<= 1)
end

# A type of data defined by the user, standardised through the hook of the package (see Standardiser): the values Z are
# standardised, and the indicators W are not. NB types and methods must be defined at the top level, not within a testset
struct _UserData{A, B}
    Z::A
    W::B
end
MLUtils.numobs(d::_UserData) = MLUtils.numobs(d.Z)
MLUtils.getobs(d::_UserData, i) = _UserData(MLUtils.getobs(d.Z, i), MLUtils.getobs(d.W, i))
(s::Standardiser)(d::_UserData) = _UserData(s(d.Z), d.W)
NeuralEstimators.fitstandardiser(d::_UserData) = NeuralEstimators.fitstandardiser(d.Z)
# A type whose method of fitstandardiser returns neither a Standardiser nor nothing
struct _BadData
    Z::Matrix{Float64}
end
NeuralEstimators.fitstandardiser(::_BadData) = 1

# A RatioEstimator pairs each data set with two parameter vectors without merging the data, so that data stored in a type
# defined by the user need only support numobs and getobs
@testset "RatioEstimator: data stored in a type defined by the user" begin
    θ_train, θ_val = sampler(64), sampler(32)
    userdata(θ) = _UserData(simulator(θ, m), rand(Bool, m, size(θ, 2)))
    Z_train, Z_val = userdata(θ_train), userdata(θ_val)
    network = Chain(z -> vcat(z.Z, Float32.(z.W)), Dense(2m, 16, gelu), Dense(16, 3d))
    trained = train(RatioEstimator(network, d; num_summaries = 3d), θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false)
    @test all(isfinite, trained(Z_val, θ_val.array))
end

@testset "Standardisation" begin
    using NeuralEstimators: _standardise, _unstandardise, _standardisedata, _Prestandardised, _inputoutput, _transformations, _repeatrows, _rows, _mapstandardisers, _mark
    using NeuralEstimators: _AutoStandardiser, _invariantloss, _linearoutputs, _fitunfitted, fitstandardiser, _withoutstandardisers
    num_summaries = 3d
    network() = Chain(Dense(m, 16, gelu), Dense(16, num_summaries))

    # ---- Standardiser ----
    θm = Float64.(θ.array)
    s = Standardiser(θ)
    @test s isa Standardiser{Matrix{Float32}} && size(s.mean) == (d, 1)
    @test vec(s.mean) ≈ vec(mean(θm; dims = 2))
    @test vec(s.std) ≈ vec(std(θm; dims = 2, corrected = false))
    @test _standardise(s, θm) ≈ (θm .- s.mean) ./ s.std
    @test _unstandardise(s, _standardise(s, θm)) ≈ θm
    @test Standardiser(1, 2) isa Standardiser{Float32}
    @test Standardiser([1, 2], 3).std == Float32[3; 3;;] # broadcast to a common shape
    @test_throws ArgumentError Standardiser(0, 0)
    @test_throws ArgumentError Standardiser([0, 1], [1, Inf])
    @test_throws ArgumentError Standardiser("a", 1)

    # The default rule for each container
    @test Standardiser(randn(Float32, 3)).mean isa Float32                    # vector: a single statistic
    @test size(Standardiser(randn(4, 10)).mean) == (4, 1)                    # matrix: one per row
    @test Standardiser(randn(Float32, 4, 4, 1, 10)).mean isa Float32          # three or more dimensions: a single statistic
    @test Standardiser(rand(1:3, 4, 4, 1, 10)).mean isa Float32               # integer data
    Zv = [randn(Float32, 3, mᵢ) .+ Float32[0, 10, 100] for mᵢ in (5, 7)]       # vector of data sets: per row, pooled over replicates
    sv = Standardiser(Zv)
    @test vec(sv.mean) ≈ vec(mean(reduce(hcat, Zv); dims = 2))
    @test vec(sv.std) ≈ vec(std(reduce(hcat, Zv); dims = 2, corrected = false))
    @test Standardiser(PackedReplicates(Zv)).mean ≈ sv.mean
    sP = Standardiser(PackedReplicates(Zv; max_sample_size = 7))             # the padding is excluded
    @test sP.mean ≈ sv.mean && sP.std ≈ sv.std
    @test Standardiser(fill(5.0f0, 2, 10)).std == ones(Float32, 2, 1)        # zero variance: σ = 1
    @test Standardiser([1f7, 1f7 + 1]).std == 1f0                           # variation too small to resolve in single precision: σ = 1
    @test_throws ArgumentError Standardiser(fill(NaN, 2, 3))                 # non-finite values
    @test_throws ArgumentError Standardiser((Z, Z))                          # unsupported container
    S = randn(Float32, 2, K)
    dS = DataAndSummaries(Z, S)
    @test_throws ArgumentError Standardiser(dS)

    # Standardisation of the data, in single precision, for each container
    Zs = _standardisedata(sv, Zv)
    @test eltype(Zs) == Matrix{Float32} && Zs[2] ≈ (Zv[2] .- sv.mean) ./ sv.std
    P = PackedReplicates(Zv; max_sample_size = 7)
    Ps = _standardisedata(sv, P)
    @test Ps isa PackedReplicates && Ps.data ≈ (P.data .- sv.mean) ./ sv.std && Ps.mask == P.mask
    sZ, sS = Standardiser(Z), Standardiser(S)
    t = _standardisedata((Z = sZ, S = sS), dS)
    @test t.Z ≈ (Z .- sZ.mean) ./ sZ.std && t.S ≈ (S .- sS.mean) ./ sS.std
    @test _standardisedata(sZ, dS).S === S                                 # a single standardiser applies to the data only
    @test _standardisedata(sZ, Summaries(S)).S === S                       # precomputed summary statistics are not standardised
    @test _standardisedata(Standardiser(), Z) === Z                         # unfitted
    @test _standardisedata(_Prestandardised(sZ), Z) === Z                   # standardised on the host during training
    @test eltype(_standardisedata(Standardiser(1, 2), Float64.(Z))) == Float32
    # The public call operator applies a standardiser in the same way, and an unfitted standardiser returns the data unchanged
    @test sZ(Z) ≈ (Z .- sZ.mean) ./ sZ.std && sv(Zv) == Zs && sv(P).data == Ps.data && sZ(dS).Z ≈ t.Z && sZ(dS).S === S
    @test sZ(Summaries(S)).S === S && Standardiser()(Z) === Z && Standardiser()(Zv) === Zv

    # ---- Defaults ----
    @test PointEstimator(network(), d; num_summaries).data_standardiser isa Standardiser{Nothing}
    @test PointEstimator(network(), d; num_summaries).parameter_standardiser isa _AutoStandardiser # standardised if this leaves the estimator unchanged
    @test_throws ArgumentError PointEstimator(network(), d; num_summaries, standardise_data = :auto) # :auto applies to the parameters of a PointEstimator only
    @test_throws ArgumentError PosteriorEstimator(network(), d; num_summaries, standardise_parameters = :auto)
    @test isnothing(PointEstimator(network(), d; num_summaries, standardise_data = false).data_standardiser)
    @test PointEstimator(network(), d; num_summaries, standardise_data = (Z = false, S = true)).data_standardiser == (Z = nothing, S = Standardiser())
    @test_throws ArgumentError PointEstimator(network(), d; num_summaries, standardise_data = 1)
    @test IntervalEstimator(network(), d; num_summaries).parameter_standardiser isa Standardiser{Nothing}
    @test QuantileEstimator(network(), d; num_summaries).parameter_standardiser isa Standardiser{Nothing}
    @test PosteriorEstimator(network(), d; num_summaries).parameter_standardiser isa Standardiser{Nothing}
    @test isnothing(PosteriorEstimator(network(), 1; num_summaries, q = SpikeAndSlab).parameter_standardiser)
    @test_throws ArgumentError PosteriorEstimator(network(), 1; num_summaries, q = SpikeAndSlab, standardise_parameters = true)
    @test RatioEstimator(network(), d; num_summaries).parameter_standardiser isa Standardiser{Nothing}
    @test TelescopingRatioEstimator(network(), d; num_summaries, sampler).parameter_standardiser isa Standardiser{Nothing}
    @test PriorEstimator(d).parameter_standardiser isa Standardiser{Nothing}
    @test isnothing(PriorEstimator(1; q = SpikeAndSlab).parameter_standardiser)
    @test_throws ArgumentError PriorEstimator(SpikeAndSlab(1, 0); standardise_parameters = true)

    # The standardisers are not part of the functors, so the Flux state (and hence the loading of files saved previously) is unchanged
    @test keys(Flux.state(PointEstimator(network(), d; num_summaries))) == (:summary_network, :inference_network)
    @test keys(Flux.state(IntervalEstimator(network(), d; num_summaries))) == (:summary_network, :u, :v, :probs, :g)
    @test keys(Flux.state(QuantileEstimator(network(), d; num_summaries))) == (:summary_network, :summary_network_θ, :v, :probs, :g, :i)
    @test keys(Flux.state(PosteriorEstimator(network(), d; num_summaries))) == (:summary_network, :q)
    @test keys(Flux.state(RatioEstimator(network(), d; num_summaries))) == (:summary_network, :summary_network_θ, :inference_network)
    @test keys(Flux.state(TelescopingRatioEstimator(network(), d; num_summaries, sampler))) == (:summary_network, :heads)
    @test keys(Flux.state(PriorEstimator(d))) == (:q,)

    # Unfitted standardisers are the identity, so that an untrained estimator can be applied
    e1 = PointEstimator(network(), d; num_summaries, standardise_parameters = true)
    e0 = PointEstimator(e1.summary_network, e1.inference_network; standardise_data = false)
    @test e1(Z) == e0(Z)

    # ---- Conformance: a standardised estimator equals the unstandardised estimator with the same networks, applied to
    # standardised inputs and with its outputs mapped back ----
    sθ = Standardiser(θ)
    Zs = (Z .- sZ.mean) ./ sZ.std
    θs = (θ.array .- sθ.mean) ./ sθ.std
    grid = expandgrid(-1:0.5:1, 0:0.25:1)'
    grids = (grid .- sθ.mean) ./ sθ.std

    # PointEstimator
    est = PointEstimator(network(), d; num_summaries, standardise_data = sZ, standardise_parameters = sθ)
    ref = PointEstimator(est.summary_network, est.inference_network; standardise_data = false)
    @test est(Z) ≈ _unstandardise(sθ, ref(Zs))
    @test estimate(est, Z; use_gpu = false) ≈ est(Z)
    @test summarystatistics(est, Z; use_gpu = false) ≈ est.summary_network(Zs)

    # IntervalEstimator and QuantileEstimator
    est = IntervalEstimator(network(), d; num_summaries, standardise_data = sZ, standardise_parameters = sθ)
    ref = IntervalEstimator(est.summary_network, est.u, est.v, est.probs, est.g; standardise_data = false)
    out = ref(Zs)
    @test est(Z) ≈ vcat(_unstandardise(sθ, out[1:d, :]), _unstandardise(sθ, out[(d + 1):end, :]))
    est = QuantileEstimator(network(), d; num_summaries, standardise_data = sZ, standardise_parameters = sθ)
    ref = QuantileEstimator(est.summary_network, est.summary_network_θ, est.v, est.probs, est.g, est.i; standardise_data = false)
    @test est(Z) ≈ _unstandardise(_repeatrows(sθ, 3), ref(Zs))
    est = QuantileEstimator(network(), d; num_summaries, i = 2, standardise_data = sZ, standardise_parameters = sθ)
    ref = QuantileEstimator(est.summary_network, est.summary_network_θ, est.v, est.probs, est.g, est.i; standardise_data = false)
    θ₁ = θ.array[1:1, :]
    @test est(Z, θ₁) ≈ _unstandardise(_repeatrows(_rows(sθ, 2:2), 3), ref(Zs, θs[1:1, :]))

    # PosteriorEstimator
    est = PosteriorEstimator(network(), d; num_summaries, q = Gaussian, standardise_data = sZ, standardise_parameters = sθ)
    ref = PosteriorEstimator(est.summary_network, est.q; standardise_data = false, standardise_parameters = false)
    seed!(1)
    a = sampleposterior(est, Z; N = 10, use_gpu = false)
    seed!(1)
    b = sampleposterior(ref, Zs; N = 10, use_gpu = false)
    @test reshape(a, d, :) ≈ _unstandardise(sθ, reshape(b, d, :))

    # RatioEstimator
    est = RatioEstimator(network(), d; num_summaries, standardise_data = sZ, standardise_parameters = sθ)
    ref = RatioEstimator(est.summary_network, est.summary_network_θ, est.inference_network; standardise_data = false, standardise_parameters = false)
    @test est(Z, θ.array) ≈ ref(Zs, θs)
    @test logratio(est, Z[:, 1:2]; grid = grid, use_gpu = false) ≈ logratio(ref, Zs[:, 1:2]; grid = grids, use_gpu = false)
    samples = sampleposterior(est, Z[:, 1:1]; lower = [-3.0, 0.0], upper = [3.0, 1.0], N = 10, warmup = 10, use_gpu = false)
    @test size(samples) == (d, 10, 1) && all([-3.0, 0.0] .<= samples[:, :, 1] .<= [3.0, 1.0])

    # TelescopingRatioEstimator
    est = TelescopingRatioEstimator(network(), d; num_summaries, sampler, standardise_data = sZ, standardise_parameters = sθ)
    ref = TelescopingRatioEstimator(est.summary_network, est.heads, sampler; standardise_data = false, standardise_parameters = false)
    @test est(Z, θ.array) ≈ ref(Zs, θs)
    @test logratio(est, Z[:, 1:2]; grid = grid, use_gpu = false) ≈ logratio(ref, Zs[:, 1:2]; grid = grids, use_gpu = false)
    lower, upper = [-3.0, 0.0], [3.0, 1.0]
    lowers, uppers = vec(_standardise(sθ, reshape(lower, :, 1))), vec(_standardise(sθ, reshape(upper, :, 1)))
    lq = logposterior(est, θ.array[:, 1:5], Z[:, 1:1]; lower, upper, method = :chebyshev, use_gpu = false)
    lqs = logposterior(ref, θs[:, 1:5], Zs[:, 1:1]; lower = lowers, upper = uppers, method = :chebyshev, use_gpu = false)
    @test lq ≈ lqs .- sum(log, sθ.std) rtol = 1e-3 # the densities of the standardised parameters, with the Jacobian
    seed!(1)
    a = sampleposterior(est, Z[:, 1:1]; lower, upper, N = 20, use_gpu = false)
    seed!(1)
    b = sampleposterior(ref, Zs[:, 1:1]; lower = lowers, upper = uppers, N = 20, use_gpu = false)
    @test reshape(a, d, :) ≈ _unstandardise(sθ, reshape(b, d, :)) rtol = 1e-3

    # PriorEstimator
    est = PriorEstimator(GaussianMixture(d, 0); standardise_parameters = sθ)
    ref = PriorEstimator(est.q; standardise_parameters = false)
    @test logdensity(est, θ; use_gpu = false) ≈ logdensity(ref, θs; use_gpu = false) .- sum(log, sθ.std)
    @test logdensity(est, θ.array[:, 1]; use_gpu = false) ≈ logdensity(est, θ; use_gpu = false)[1]
    @test all(isfinite, ForwardDiff.gradient(x -> logdensity(est, x; use_gpu = false), Float64.(θ.array[:, 1]))) # e.g., as the log prior of NUTS
    seed!(1)
    a = sample(est, 10; use_gpu = false)
    seed!(1)
    b = sample(ref, 10; use_gpu = false)
    @test a ≈ _unstandardise(sθ, b)
    flow = PriorEstimator(NormalisingFlow(d, 0); standardise_parameters = sθ)
    flowref = PriorEstimator(flow.q; standardise_parameters = false)
    @test forward(flow, θ; use_gpu = false) ≈ forward(flowref, θs; use_gpu = false)
    U = randn(Float32, d, 5)
    @test inverse(flow, U; use_gpu = false) ≈ _unstandardise(sθ, inverse(flowref, U; use_gpu = false))

    # Expert summary statistics (DataAndSummaries)
    est = PointEstimator(network(), d; num_summaries = num_summaries + 2, standardise_data = (Z = sZ, S = sS))
    ref = PointEstimator(est.summary_network, est.inference_network; standardise_data = false)
    Ss = (S .- sS.mean) ./ sS.std
    @test est(dS) ≈ ref(DataAndSummaries(Zs, Ss))
    @test estimate(est, dS; use_gpu = false) ≈ est(dS)
    @test summarystatistics(est, dS; use_gpu = false) ≈ vcat(est.summary_network(Zs), Ss)

    # ---- Training: the standardisers are fitted once, and the data and parameters are standardised on the host ----
    θ_train, θ_val = sampler(100), sampler(50)
    Z_train, Z_val = simulator(θ_train, m), simulator(θ_val, m)
    est = PointEstimator(network(), d; num_summaries, standardise_parameters = true)
    trained = train(est, θ_train, θ_val, Z_train, Z_val; epochs = 2, verbose = false)
    @test trained.data_standardiser.mean ≈ Standardiser(Z_train).mean # fitted to the training data
    @test trained.parameter_standardiser.mean ≈ Standardiser(θ_train).mean
    @test est.parameter_standardiser isa Standardiser{Nothing}       # the given estimator keeps its unfitted standardisers
    input, output = _inputoutput(trained, Z_val, θ_val.array)
    @test output ≈ _standardise(trained.parameter_standardiser, θ_val.array)
    marked = _mapstandardisers(_mark, trained)                       # the estimator being trained skips standardisation
    @test marked(input) ≈ _standardise(trained.parameter_standardiser, trained(Z_val)) rtol = 1e-4
    trained2 = train(trained, θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false) # never refitted
    @test trained2.parameter_standardiser.mean == trained.parameter_standardiser.mean
    @test trained2.data_standardiser.std == trained.data_standardiser.std
    reused = PointEstimator(summarynetwork(trained), d; num_summaries, standardise_data = datastandardiser(trained)) # e.g., transfer learning
    @test datastandardiser(reused) === datastandardiser(trained)
    target = PointEstimator(network(), d; num_summaries)
    transferred = setsummarynetwork(target, trained)
    @test summarynetwork(transferred) === summarynetwork(trained)
    @test datastandardiser(transferred) === datastandardiser(trained)
    @test transferred.inference_network === target.inference_network
    @test datastandardiser(setdatastandardiser(target, datastandardiser(trained))) === datastandardiser(trained)
    @test datastandardiser(setdatastandardiser(target, marked)) === datastandardiser(trained) # a standardiser marked during training
    @test isnothing(datastandardiser(setdatastandardiser(target, PointEstimator(network(), d; num_summaries, standardise_data = false))))
    @test isnothing(datastandardiser(setdatastandardiser(target, nothing)))
    withexpert = PointEstimator(network(), d; num_summaries, standardise_data = (Z = sZ, S = sS))
    @test datastandardiser(setdatastandardiser(target, withexpert)) == (Z = sZ, S = sS)
    @test datastandardiser(setsummarynetwork(target, withexpert)) == (Z = sZ, S = sS)
    @test_throws ArgumentError setdatastandardiser(PriorEstimator(GaussianMixture(d, 0)), trained)
    trained = train(est, sampler, simulator; simulator_args = m, K = 100, epochs = 1, verbose = false) # fitted to the validation data
    @test trained.parameter_standardiser isa Standardiser{Matrix{Float32}}
    trained = train(est, θ_train, θ_val, simulator; simulator_args = m, epochs = 1, verbose = false)
    @test trained.parameter_standardiser.mean ≈ Standardiser(θ_train).mean
    trained = train(est, θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false, freeze_summary_network = true)
    @test trained.data_standardiser isa Standardiser{Matrix{Float32}}
    est = IntervalEstimator(network(), d; num_summaries, lower = [-Inf, 0], upper = [Inf, 1], standardise_parameters = true)
    trained = train(est, θ_train, θ_val, Z_train, Z_val; epochs = 2, verbose = false)
    intervals = estimate(trained, Z_val; use_gpu = false)
    @test all(0 .<= intervals[[2, 4], :] .<= 1)
    @test trained.parameter_standardiser.mean ≈ Standardiser(scaledlogit.(θ_train.array, [-Inf, 0], [Inf, 1])).mean # fitted to the transformed parameters
    est = PointEstimator(network(), d; num_summaries, standardise_parameters = false)
    trainstate = FluxTrainState(est, Adam(), Optimisers.setup(Adam(), est))
    trainstate = @test_logs (:info, r"data are not standardised") _fitunfitted(trainstate, θ_train, (Z_train, Z_train), NeuralEstimators.mae, true) # unsupported data
    @test isnothing(trainstate.model.data_standardiser)

    # ---- Automatic standardisation of the parameters of a PointEstimator: only if this leaves the estimator unchanged ----
    # The loss: a sum over the parameters of positively homogeneous functions of the errors, on both scales
    θl = θ_train.array
    sl = Standardiser(θl)
    for loss in (NeuralEstimators.mae, NeuralEstimators.mse, Flux.mae, (x, y) -> quantileloss(x, y, 0.3), (x, y) -> kpowerloss(x, y, 2; joint = false, safeorigin = false))
        @test _invariantloss(loss, sl, θl)
    end
    for loss in ((x, y) -> tanhloss(x, y, 0.1), (x, y) -> tanhloss(x, y, 0.1; joint = false), (x, y) -> tanhloss(x, y, 1e-4), # incl. saturated
                 (x, y) -> kpowerloss(x, y, 0.5), (x, y) -> kpowerloss(x, y, 2; safeorigin = false), (x, y) -> Flux.huber_loss(x, y),
                 (x, y) -> mean(abs.(x .- y) ./ abs.(y)), (x, y) -> mean(abs.(log.(x) .- log.(y))), (x, y) -> error("a loss that throws"))
        @test !_invariantloss(loss, sl, θl)
    end
    # Constants chosen for the original scale are detected, also when the loss is nearly linear on the standardised scale
    θ289 = 500 .+ 289 .* randn(2, 200)
    @test !_invariantloss((x, y) -> tanhloss(x, y, 50), Standardiser(θ289), θ289)
    @test !_invariantloss((x, y) -> Flux.huber_loss(x, y; delta = 6), Standardiser(θ289), θ289)
    for θs in (1f-3 .* randn(Float32, 2, 100) .+ 0.5f0, 1f3 .* randn(Float32, 2, 100), 1f5 .+ randn(Float32, 2, 100), randn(Float32, 1, 100), randn(Float32, 1000, 100))
        @test _invariantloss(NeuralEstimators.mae, Standardiser(θs), θs)
        @test !_invariantloss((x, y) -> tanhloss(x, y, 0.1), Standardiser(θs), θs)
    end
    θ1000 = randn(1000, 100)
    @test !_invariantloss((x, y) -> kpowerloss(x, y, 0.5; safeorigin = false), Standardiser(θ1000), θ1000) # a power of a sum over many parameters
    seed!(3); u = rand(); seed!(3); _invariantloss(NeuralEstimators.mae, sl, θl); @test rand() == u # the global random stream is unchanged
    # The neural network: a final linear layer with one output per parameter
    @test _linearoutputs(PointEstimator(network(), d; num_summaries)) == d                       # the MLP inference network
    @test _linearoutputs(PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)))) == d             # a single network
    @test _linearoutputs(Chain(Dense(m, 16, gelu), Dense(16, d), identity)) == d                   # final identity layers are skipped
    @test isnothing(_linearoutputs(Chain(Dense(m, 16, gelu), Dense(16, d, exp))))
    @test isnothing(_linearoutputs(Chain(Dense(m, d), Compress(zeros(d), ones(d)))))
    @test isnothing(_linearoutputs(Chain(Dense(m, d, softplus), identity)))
    @test isnothing(_linearoutputs(Chain(Dense(m, 1), Parallel(vcat, Dense(1, 1), Dense(1, 1, softplus)))))
    @test isnothing(_linearoutputs(Chain()))
    @test _linearoutputs(DeepSet(Chain(Dense(1, 8, relu)), Chain(Dense(8, d)))) == d
    @test isnothing(_linearoutputs(DeepSet(Chain(Dense(1, 8, relu)), Chain(Dense(8, d, exp)))))
    @test isnothing(_linearoutputs(DeepSet(Chain(Dense(1, 8, relu)), identity)))
    # The decision, made when the standardisers are fitted (with a message when verbose)
    decide(est, loss; verbose = false) = _fitunfitted(FluxTrainState(est, Adam(), Optimisers.setup(Adam(), est)), θ_train, Z_train, loss, verbose).model.parameter_standardiser
    tanh01 = (x, y) -> tanhloss(x, y, 0.1)
    auto = PointEstimator(network(), d; num_summaries)
    constrained = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d, exp)))
    wide = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, 2d)))      # an output that is not one per parameter
    forced = PointEstimator(network(), d; num_summaries, standardise_parameters = true)
    off = PointEstimator(network(), d; num_summaries, standardise_parameters = false)
    @test decide(auto, NeuralEstimators.mae) isa Standardiser{Matrix{Float32}}
    @test (@test_logs (:info, r"parameters are standardised") decide(auto, NeuralEstimators.mae; verbose = true)) isa Standardiser
    @test isnothing(@test_logs (:info, r"would change the Bayes estimator") decide(auto, tanh01; verbose = true))
    @test isnothing(@test_logs (:info, r"may constrain the estimates") decide(constrained, NeuralEstimators.mae; verbose = true))
    @test isnothing(decide(wide, (x, y) -> NeuralEstimators.mae(x[1:size(y, 1), :], y)))
    @test (@test_logs decide(forced, tanh01; verbose = true)) isa Standardiser # explicit: no message
    @test isnothing(@test_logs decide(off, NeuralEstimators.mae; verbose = true))
    # Training: the decision is made once, by the first call to train()
    trained = train(PointEstimator(network(), d; num_summaries), θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false)
    @test trained.parameter_standardiser.mean ≈ Standardiser(θ_train).mean
    @test isnothing(train(PointEstimator(network(), d; num_summaries), θ_train, θ_val, Z_train, Z_val; loss = tanh01, epochs = 1, verbose = false).parameter_standardiser)
    @test train(trained, θ_train, θ_val, Z_train, Z_val; loss = tanh01, epochs = 1, verbose = false).parameter_standardiser.mean == trained.parameter_standardiser.mean
    @test isnothing(train(PointEstimator(network(), d; num_summaries), θ_train, θ_val, simulator; simulator_args = m, loss = tanh01, epochs = 1, verbose = false).parameter_standardiser)
    @test isnothing(train(PointEstimator(network(), d; num_summaries), sampler, simulator; simulator_args = m, K = 50, loss = tanh01, epochs = 1, verbose = false).parameter_standardiser)
    # Parameters stored as an array with three or more dimensions (e.g., p parameters at each of L locations), with a single statistic
    p, L = 2, 5
    θ3 = 100 .+ 10 .* randn(Float32, p, L, 64)
    Z3 = randn(Float32, m, 64)
    e3 = PointEstimator(Chain(Dense(m, 4L, gelu), x -> reshape(x, 4, L, :), Dense(4, p)))
    @test _invariantloss(NeuralEstimators.mae, Standardiser(θ3), θ3) && !_invariantloss(tanh01, Standardiser(θ3), θ3)
    trained3 = train(e3, θ3, θ3, Z3, Z3; epochs = 1, verbose = false)
    @test trained3.parameter_standardiser isa Standardiser{Float32}
    @test size(trained3(Z3)) == size(θ3)
    # Interval and quantile estimators standardise the parameters by default
    @test train(QuantileEstimator(network(), d; num_summaries), θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false).parameter_standardiser isa Standardiser{Matrix{Float32}}
    # An untrained estimator keeps its default through saving and loading
    savepath5 = mktempdir()
    saveestimator(PointEstimator(network(), d; num_summaries), savepath5)
    @test loadestimator(PointEstimator(network(), d; num_summaries, standardise_parameters = false), savepath5).parameter_standardiser isa _AutoStandardiser

    # ---- Data stored in a type defined by the user (see _UserData) ----
    ud_train, ud_val = _UserData(Z_train, rand(Bool, size(Z_train))), _UserData(Z_val, rand(Bool, size(Z_val)))
    @test fitstandardiser(ud_train).mean == fitstandardiser(Z_train).mean # the statistics of the values only
    @test Standardiser(ud_train).std == Standardiser(Z_train).std
    @test_throws ArgumentError Standardiser(_BadData(randn(2, 3)))         # fitstandardiser must return a Standardiser or nothing
    userest = PointEstimator(Chain(u -> vcat(u.Z, Float32.(u.W)), Dense(2m, 16, gelu), Dense(16, d)))
    trained = train(userest, θ_train, θ_val, ud_train, ud_val; epochs = 1, verbose = false)
    su = trained.data_standardiser
    @test su.mean ≈ Standardiser(Z_train).mean
    ud_std = _UserData(su(ud_val.Z), ud_val.W)                             # the values standardised, the indicators unchanged
    @test trained(ud_val) ≈ _unstandardise(trained.parameter_standardiser, _withoutstandardisers(trained)(ud_std))
    @test estimate(trained, ud_val; use_gpu = false) ≈ trained(ud_val)    # mini-batches of the type
    @test summarystatistics(trained, ud_val; use_gpu = false) ≈ trained.summary_network(ud_std)
    @test su(DataAndSummaries(ud_val, randn(Float32, 2, size(Z_val, 2)))).Z.Z ≈ ud_std.Z # within a container of the package
    @test Standardiser()(ud_val).Z === ud_val.Z
    usimulator(θ, m) = _UserData(simulator(θ, m), rand(Bool, m, size(θ, 2)))
    trained = train(userest, θ_train, θ_val, usimulator; simulator_args = m, epochs = 1, verbose = false) # simulated data
    @test trained.data_standardiser isa Standardiser{Matrix{Float32}}

    # ---- Persistence ----
    savepath = mktempdir()
    est = PosteriorEstimator(network(), d; num_summaries, q = Gaussian, lower = [-Inf, 0], upper = [Inf, 1])
    trained = train(est, θ_train, θ_val, Z_train, Z_val; epochs = 2, savepath = savepath, verbose = false)
    @test _transformations(trained).bounds == (lower = [-Inf, 0.0], upper = [Inf, 1.0])
    loaded = loadestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian), savepath) # the architecture only
    @test loaded.transform isa NeuralEstimators._ScaledLogit
    @test loaded.parameter_standardiser.mean == trained.parameter_standardiser.mean
    @test loaded.data_standardiser.std == trained.data_standardiser.std
    seed!(1)
    a = sampleposterior(trained, Z; N = 20, use_gpu = false)
    seed!(1)
    b = sampleposterior(loaded, Z; N = 20, use_gpu = false)
    @test a == b
    # The saved estimator is authoritative, with a warning if the given estimator has different statistics
    given = PosteriorEstimator(network(), d; num_summaries, q = Gaussian, standardise_parameters = Standardiser([0, 0], [1, 1]))
    loaded = @test_logs (:warn, r"differs") loadestimator(given, savepath)
    @test loaded.parameter_standardiser.mean == trained.parameter_standardiser.mean
    # saveestimator() saves the same file
    savepath2 = mktempdir()
    saveestimator(trained, savepath2)
    loaded = loadestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian), savepath2)
    seed!(1)
    @test sampleposterior(loaded, Z; N = 20, use_gpu = false) == a
    # Custom transformations cannot be saved, so they must be given again
    transform(x) = vcat(x[1:1, :], log.(x[2:2, :] .+ 0.1))
    invtransform(x) = vcat(x[1:1, :], exp.(x[2:2, :]) .- 0.1)
    saveestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian, transform, invtransform), savepath2)
    @test_throws ArgumentError loadestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian), savepath2)
    @test loadestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian, transform, invtransform), savepath2).transform === transform
    # A file saved without the standardisers and bounds (e.g., by an earlier version) is loaded without them
    model_state = Flux.state(trained)
    savepath3 = mktempdir()
    NeuralEstimators.@save joinpath(savepath3, "best_estimator.bson") model_state
    loaded = loadestimator(PosteriorEstimator(network(), d; num_summaries, q = Gaussian), savepath3)
    @test isnothing(loaded.data_standardiser) && isnothing(loaded.parameter_standardiser) && loaded.transform === identity
    # Ensemble: the standardisers of each component
    savepath4 = mktempdir()
    ensemble = Ensemble([PointEstimator(network(), d; num_summaries, standardise_parameters = true) for _ = 1:2])
    ensemble = train(ensemble, θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false, savepath = savepath4)
    loaded = loadestimator(Ensemble([PointEstimator(network(), d; num_summaries) for _ = 1:2]), savepath4)
    @test estimate(loaded, Z; use_gpu = false) ≈ estimate(ensemble, Z; use_gpu = false)
    @test_throws ArgumentError loadestimator(PointEstimator(network(), d; num_summaries), savepath4) # an Ensemble file

    # ---- Containers, shapes, and precision ----
    est = PointEstimator(network(), d; num_summaries, standardise_data = sZ, standardise_parameters = sθ)
    # Containers whose element type is wider than their contents are standardised like concretely typed data
    @test est(Matrix{Union{Missing, Float32}}(Z)) ≈ est(Z)
    Zvec = [randn(Float32, 1, mᵢ) .+ 5 for mᵢ in (3, 5)]
    deepset = PointEstimator(DeepSet(Chain(Dense(1, 16, gelu)), Chain(Dense(16, num_summaries))), d; num_summaries, standardise_data = Standardiser([z .- 5 for z in Zvec]))
    @test deepset(Any[z for z in Zvec]) ≈ deepset(Zvec)                                         # Vector{Any}
    @test deepset([Float64.(Zvec[1]), Zvec[2]]) ≈ deepset(Zvec)                                # mixed precision (Vector{Matrix})
    posterior = PosteriorEstimator(network(), d; num_summaries, q = Gaussian, standardise_data = sZ, standardise_parameters = sθ)
    @test summarystatistics(posterior, Matrix{Union{Missing, Float64}}(Z); use_gpu = false) ≈ summarystatistics(posterior, Z; use_gpu = false)
    # A single data set given as a vector keeps its shape, and a mis-shaped input is an error rather than broadcast
    @test est(Z[:, 1]) isa AbstractVector && est(Z[:, 1]) ≈ vec(est(Z[:, 1:1]))
    @test_throws DimensionMismatch est(Z[1:1, :])
    q2 = QuantileEstimator(network(), 3; num_summaries, i = 1, standardise_data = sZ, standardise_parameters = Standardiser([0, 1, 2], [1, 2, 3]))
    @test_throws DimensionMismatch q2(Z[:, 1:2], [0.2f0, 0.7f0]) # a single conditioning vector whose length equals the number of data sets
    # Data in double precision are rounded to single precision before they are standardised, as for estimate()
    Z64 = 1f6 .+ Float64.(Z)
    est64 = PointEstimator(network(), d; num_summaries, standardise_data = Standardiser(Z64))
    @test est64(Z64) ≈ estimate(est64, Z64; use_gpu = false)

    # ---- Statistics ----
    @test Standardiser(fill(1, 1, 1), fill(3, 1, 1)).std === 3f0 # a single statistic is stored as a number
    single = PriorEstimator(GaussianMixture(d, 0); standardise_parameters = Standardiser(fill(1, 1, 1), fill(3, 1, 1)))
    scalar = PriorEstimator(single.q; standardise_parameters = Standardiser(1, 3))
    @test logdensity(single, θ; use_gpu = false) ≈ logdensity(scalar, θ; use_gpu = false)
    tiny = Standardiser(1e-50 .* randn(2, 100)) # a standard deviation that underflows in single precision
    @test tiny.std == ones(Float32, 2, 1) && all(isfinite, _standardise(tiny, 1e-50 .* randn(2, 3)))
    @test_throws ArgumentError Standardiser(fill(1e40, 2, 3) .* randn(2, 3)) # overflows in single precision
    @test tanhloss(randn(Float32, d, 5), randn(Float32, d, 5), 0.1f0) isa Real # NB tanh_fast

    # ---- freeze_summary_network with an identity inference network returns the estimator unchanged ----
    e = PointEstimator(Chain(Dense(m, 16, gelu), Dense(16, d)))
    ret = @test_logs (:warn, r"returning the estimator unchanged") match_mode = :any train(e, θ_train, θ_val, Z_train, Z_val; epochs = 1, verbose = false, freeze_summary_network = true)
    @test ret.data_standardiser isa Standardiser{Nothing}
    @test ret(Z) == e(Z)
    for args in ((θ_train, θ_val, simulator), (sampler, simulator))
        ret = @test_logs (:warn, r"returning the estimator unchanged") match_mode = :any train(e, args...; simulator_args = m, epochs = 1, verbose = false, freeze_summary_network = true)
        @test ret(Z) == e(Z)
    end
end

@testset "Expert summaries only (no summary network)" begin
    num_summaries = 4
    S = randn(Float32, num_summaries, K)
    mlp_kwargs = (depth = 1, width = 16)

    point = PointEstimator(d; num_summaries = num_summaries, mlp_kwargs...)
    @test summarynetwork(point) === identity
    @test size(estimate(point, S; use_gpu = false)) == (d, K)

    interval = IntervalEstimator(d; num_summaries = num_summaries, mlp_kwargs...)
    @test size(interval(S)) == (2d, K)

    quantile = QuantileEstimator(d; num_summaries = num_summaries, mlp_kwargs...)
    @test size(quantile(S)) == (3d, K)

    posterior = PosteriorEstimator(d; num_summaries = num_summaries, q = Gaussian, mlp_kwargs...)
    samples = sampleposterior(posterior, S; N = 10, use_gpu = false)
    @test size(samples) == (d, 10, K)

    ratio = RatioEstimator(d; num_summaries = num_summaries, mlp_kwargs...)
    @test size(ratio(S, θ)) == (1, K)

    telescoping = TelescopingRatioEstimator(d; num_summaries = num_summaries, sampler = sampler, mlp_kwargs...)
    @test size(telescoping(S, θ)) == (d, K)

    # Both argument orders construct the same way
    ψ = Chain(Dense(m, num_summaries))
    e_new = PointEstimator(d, ψ; num_summaries = num_summaries, mlp_kwargs...)
    e_old = PointEstimator(ψ, d; num_summaries = num_summaries, mlp_kwargs...)
    @test summarynetwork(e_new) === ψ
    @test summarynetwork(e_old) === ψ
    p_new = PosteriorEstimator(d, ψ; num_summaries = num_summaries, q = Gaussian, mlp_kwargs...)
    p_old = PosteriorEstimator(ψ, d; num_summaries = num_summaries, q = Gaussian, mlp_kwargs...)
    @test summarynetwork(p_new) === ψ
    @test summarynetwork(p_old) === ψ
    @test_throws ArgumentError PointEstimator(d, num_summaries; num_summaries = num_summaries)
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

    # Training (on-the-fly simulation)
    ensemble = train(ensemble, sampler, simulator, simulator_args = m, epochs = 1, verbose = verbose, use_gpu = dvc == gpu)

    # Training (fixed parameters and data) — exercises the Ensemble vs
    # AbstractNeuralEstimator method that was previously ambiguous
    θ_train = sampler(16)
    θ_val = sampler(8)
    Z_train = simulator(θ_train, m)
    Z_val = simulator(θ_val, m)
    ensemble = Ensemble([initestimator() for _ = 1:J])
    ensemble = train(ensemble, θ_train, θ_val, Z_train, Z_val, epochs = 1, verbose = verbose, use_gpu = dvc == gpu)

    # Assessment
    assessment = assess(ensemble, θ, Z)
    rmse(assessment)

    # Apply to data
    estimate(ensemble, Z)
end

@testset "EM" begin
    d = 2    # number of parameters in the statistical model

    # Set the (gridded) spatial domain
    points = range(0.0, 1.0, 16)
    S = expandgrid(points, points)

    # Model information that is constant (and which will be passed into later functions)
    ξ = (
        S = S,
        D = pairwise(Euclidean(), S, S, dims = 1),
        d = d
    )

    struct GPParameters <: AbstractParameterSet
        θ
        cholesky_factors
    end

    function GPParameters(K::Integer, ξ)

        # Sample parameters from the prior
        τ = 0.3 * rand(K)
        ρ = 0.3 * rand(K)

        # Compute Cholesky factors
        cholesky_factors = map(1:K) do k
            C = exp.(-ξ.D ./ ρ[k])
            L = cholesky(Symmetric(C)).L
            convert(Array, L)
        end
        cholesky_factors = Base.stack(cholesky_factors)

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
        D = ξ.D
        D₂₂ = D[I₂, I₂]
        D₁₁ = D[I₁, I₁]
        D₁₂ = D[I₁, I₂]

        # Extract the parameters from θ
        τ = θ[1]
        ρ = θ[2]

        # Compute covariance matrices
        Σ₂₂ = exp.(-D₂₂ ./ ρ)
        Σ₂₂[diagind(Σ₂₂)] .+= τ^2
        Σ₁₁ = exp.(-D₁₁ ./ ρ)
        Σ₁₁[diagind(Σ₁₁)] .+= τ^2
        Σ₁₂ = exp.(-D₁₂ ./ ρ)

        # Compute the Cholesky factor of Σ₁₁ and solve the lower triangular system
        L₁₁ = cholesky(Symmetric(Σ₁₁)).L
        x = L₁₁ \ Σ₁₂

        # Conditional covariance matrix, cov(Z₂ ∣ Z₁, θ), and its Cholesky factor
        Σ = Σ₂₂ - x'x
        L = cholesky(Symmetric(Σ)).L

        # Conditional mean, E(Z₂ ∣ Z₁, θ)
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
    Z = simulate(θ, 1)[1][:, :]
    Z = removedata(Z, 0.25)

    # Construct neural MAP estimator
    ψ = Chain(
        Conv((10, 10), 1 => 16, relu),
        Conv((5, 5), 16 => 32, relu),
        Conv((3, 3), 32 => 64, relu),
        Flux.flatten
    )
    ϕ = Chain(Dense(64, 32, relu), Dense(32, d, exp))
    network = DeepSet(ψ, ϕ)
    neuralMAPestimator = PointEstimator(network)

    # EM object
    neuralem = EM(simulateconditional, neuralMAPestimator)
    θ₀ = [0.15, 0.15]
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
    Z = simulate(θ, 1)[1]
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
        ρ = 0.6f0
        Σ = Symmetric(exp.(-D / ρ))
        L = cholesky(Σ).L
        m = 5

        @test eltype(simulateschlather(L, m)) == Float32
        @test eltype(simulategaussian(L, m)) == Float32

        ## Potts model
        β = 0.7
        complete_grid = simulatepotts(n, n, 2, 1.15)      # simulate marginally from the Ising model
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
