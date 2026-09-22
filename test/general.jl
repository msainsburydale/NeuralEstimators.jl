using NeuralEstimators
using NeuralEstimators: _check_sizes, _extractθ, rowwisenorm, triangularnumber, forward, inverse, _logdensity
using NeuralEstimators: _TrainDisplay, _status!, _finishline!, _epoch_status, _bar!, _refresh_status!, _clear_refresh!, _fit_line
using NeuralEstimators: ActNorm, Permutation, AffineCouplingBlock, CouplingLayer
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

            # Without expert summaries
            ds_no_s = DataAndSummaries(Z)
            @test ds_no_s.Z === Z
            @test isnothing(ds_no_s.S)

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
            ds = DataAndSummaries(Z)
            ds32 = f32(ds)
            @test eltype(ds32.Z[1]) == Float32
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

            # Without expert summaries: output size unchanged
            ds_no_s = DataAndSummaries(Z) |> f32
            t_no_s = estimator(ds_no_s)
            @test size(t_no_s, 1) == num_summaries
            @test size(t_no_s, 2) == K
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
            @test size(samples) == (d, N, K)
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
            tgt = randn(Float32, d, length(ms))
            g1 = trainables(Flux.gradient(m -> mae(m(G), tgt), ds)[1])
            g2 = trainables(Flux.gradient(m -> mae(m(P), tgt), ds)[1])
            @test all(isapprox.(g1, g2; rtol = 1e-3, atol = 1e-6))
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

            p = plot(assessment)
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
    p = plot(assessment)

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
    p = plot(assessment)

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
    @test size(samples) == (d, 1000, 1)
    seed!(1)
    samples1 = sampleposterior(estimator, z; grid = grid, N = 50)
    seed!(1)
    samples2 = infer(estimator, z; grid = grid, N = 50)
    @test samples1 == samples2

    # Assessment (grid-based)
    assessment = assess(estimator, θ, Z; grid = grid)
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

    assessment = assess(estimator, θ, Z; lower = lower, upper = upper)
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
        p = plot(assessment)
    end
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
