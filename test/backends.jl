using Test
using CUDA, cuDNN
using NeuralEstimators, ADTypes, Enzyme, Zygote, Reactant
using Optimisers
using Lux
using Flux
using AdvancedHMC, ForwardDiff, LogDensityProblems # loads the AdvancedHMC extension (NUTS sampling for RatioEstimator)
using Random
using Statistics: mean

d = 2
n = 100

sampler(K) = NamedMatrix(μ = rand(Float32, K), σ = rand(Float32, K))
simulator(θ::AbstractVector) = θ["μ"] .+ θ["σ"] .* sort(randn(Float32, n))
simulator(θ::AbstractMatrix) = reduce(hcat, map(simulator, eachcol(θ)))

K = 1000
θ_train = sampler(K)
θ_val = sampler(K)
Z_train = simulator(θ_train);
Z_val = simulator(θ_val);
θ_test = sampler(500)
Z_test = simulator(θ_test);
θ_single = sampler(1)
Z_single = simulator(θ_single);

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
Return (devices, adtypes) appropriate for the given backend, skipping GPU entries when CUDA is not functional.
"""
function backend_config(backend)
    devices = Any[cpu_device()]
    adtypes = Any[AutoZygote()]
    if backend === Lux
        adtypes = push!(adtypes, AutoEnzyme())
        if CUDA.functional()
            push!(devices, gpu_device())
            # Reactant GPU only when available
            try
                Reactant.set_default_backend("gpu")
                push!(devices, reactant_device())
                push!(adtypes, AutoReactant())
            catch
                @warn "Reactant GPU backend unavailable, skipping"
            end
        end
        return devices, adtypes
    elseif backend === Flux
        # adtypes = push!(adtypes, AutoEnzyme())
        CUDA.functional() && push!(devices, gpu_device())
        return devices, adtypes
    else
        error("Unknown backend: $backend")
    end
end

"""Build a fresh estimator for the given backend and estimator type."""
function make_estimator(backend, estimator_type::Symbol)
    network = MLP(n, d; depth = 1, width = 16, backend = backend)

    est = if estimator_type === :point
        PointEstimator(network, d; num_summaries = d, depth = 1)
    elseif estimator_type === :ratio
        RatioEstimator(network, d; num_summaries = d, depth = 1)
    elseif estimator_type === :posterior_mixture
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = GaussianMixture)
    elseif estimator_type === :posterior_mixture_diagonal
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = GaussianMixture, diagonal = true)
    elseif estimator_type === :posterior_gaussian
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = Gaussian)
    else
        error("Unknown estimator type: $estimator_type")
    end

    backend === Lux ? (est |> LuxEstimator) : est
end

# ──────────────────────────────────────────────────────────────────────────────
# DeepSet (Lux)
# ──────────────────────────────────────────────────────────────────────────────

function _leafarrays(x, acc = Any[])
    if x isa AbstractArray
        push!(acc, x)
    elseif x isa Union{NamedTuple, Tuple}
        foreach(v -> _leafarrays(v, acc), x)
    end
    return acc
end

@testset "DeepSet Lux" begin
    rng = Random.default_rng()
    n_ds, w, dₜ, out_dim = 10, 32, 16, 5
    makeψ() = Lux.Chain(Lux.Dense(n_ds => w, Lux.relu), Lux.Dense(w => dₜ, Lux.relu))
    makeϕ(dₛ = 0) = Lux.Chain(Lux.Dense(dₜ + dₛ => w, Lux.relu), Lux.Dense(w => out_dim))

    @testset "forward and gradients" begin
        for M in ((3, 3, 3), (3, 4, 7))
            for condition_on_sample_size in (false, true)
                dₛ = Int(condition_on_sample_size)
                ds = DeepSet(makeψ(), makeϕ(dₛ); condition_on_sample_size)
                ps, st = Lux.setup(rng, ds)
                @test haskey(ps, :ψ) && haskey(ps, :ϕ)
                @test haskey(st, :ψ) && haskey(st, :ϕ)

                Z = [rand(Float32, n_ds, m) for m in M]
                y, st_new = ds(Z, ps, st)
                @test size(y) == (out_dim, length(M))
                @test haskey(st_new, :ψ) && haskey(st_new, :ϕ)

                P = PackedReplicates(Z)
                yP, _ = ds(P, ps, st)
                @test yP ≈ y

                Ppad = PackedReplicates(Z; max_sample_size = maximum(M))
                yPad, _ = ds(Ppad, ps, st)
                @test yPad ≈ y

                y1, _ = ds(Z[1], ps, st)
                @test size(y1, 1) == out_dim

                gs = Zygote.gradient(ps -> sum(abs2, first(ds(Z, ps, st))), ps)[1]
                gsP = Zygote.gradient(ps -> sum(abs2, first(ds(P, ps, st))), ps)[1]
                gsPad = Zygote.gradient(ps -> sum(abs2, first(ds(Ppad, ps, st))), ps)[1]
                @test !isempty(_leafarrays(gs))
                @test all(isapprox.(_leafarrays(gs), _leafarrays(gsP); rtol = 1.0f-3))
                @test all(isapprox.(_leafarrays(gs), _leafarrays(gsPad); rtol = 1.0f-3))
            end
        end
    end

    @testset "convenience constructor infers Lux backend" begin
        ds = DeepSet(makeψ(); latent_dim = dₜ, output_dim = out_dim)
        @test ds.ϕ isa Lux.Chain
        ps, st = Lux.setup(rng, ds)
        Z = [rand(Float32, n_ds, m) for m in (3, 4)]
        y, _ = ds(Z, ps, st)
        @test size(y) == (out_dim, 2)

        ds = DeepSet(makeψ(); latent_dim = dₜ, output_dim = out_dim, condition_on_sample_size = true)
        @test ds.ϕ isa Lux.Chain
        ps, st = Lux.setup(rng, ds)
        y, _ = ds(Z, ps, st)
        @test size(y) == (out_dim, 2)
    end

    @testset "PointEstimator smoke test" begin
        num_summaries = 8
        ds = DeepSet(
            Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => num_summaries, Lux.relu)),
            Lux.Chain(Lux.Dense(num_summaries => 16, Lux.relu), Lux.Dense(16 => num_summaries))
        )
        est = LuxEstimator(PointEstimator(ds, d; num_summaries = num_summaries, depth = 1, width = 8))
        K_small = 16
        θ_tr = sampler(K_small)
        θ_va = sampler(K_small)
        Z_tr = [randn(Float32, 1, 10) for _ = 1:K_small]
        Z_va = [randn(Float32, 1, 10) for _ = 1:K_small]
        est = train(est, θ_tr, θ_va, Z_tr, Z_va; epochs = 1, verbose = false, device = cpu_device(), adtype = AutoZygote())
        out = estimate(est, Z_tr; use_gpu = false)
        @test size(out) == (d, K_small)
    end
end

@testset "DeepSet Lux Reactant padded PackedReplicates" begin
    if CUDA.functional()
        reactant_ok = try
            Reactant.set_default_backend("gpu")
            true
        catch
            @warn "Reactant GPU backend unavailable, skipping DeepSet Reactant test"
            false
        end
        if reactant_ok
            num_summaries = 8
            ds = DeepSet(
                Lux.Chain(Lux.Dense(1 => 16, Lux.relu), Lux.Dense(16 => num_summaries, Lux.relu)),
                Lux.Chain(Lux.Dense(num_summaries => 16, Lux.relu), Lux.Dense(16 => num_summaries))
            )
            est = LuxEstimator(PointEstimator(ds, d; num_summaries = num_summaries, depth = 1, width = 8))
            K_small = 16
            θ_tr = sampler(K_small)
            θ_va = sampler(K_small)
            Z_tr = PackedReplicates([randn(Float32, 1, m) for m in rand(5:10, K_small)]; max_sample_size = 10)
            Z_va = PackedReplicates([randn(Float32, 1, m) for m in rand(5:10, K_small)]; max_sample_size = 10)
            est = train(est, θ_tr, θ_va, Z_tr, Z_va; epochs = 1, verbose = false, device = reactant_device())
            out = estimate(est, Z_tr; use_gpu = false)
            @test size(out) == (d, K_small)
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Training scenarios
# ──────────────────────────────────────────────────────────────────────────────

TRAINING_SCENARIOS = [
# (args = (θ_train, θ_val, simulator),            label = "on-the-fly simulator"),
    (args = (θ_train, θ_val, Z_train, Z_val), label = "fixed parameters and data"),
# (args = (sampler, simulator),                   label = "on-the-fly sampler+simulator"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────────────────────────────────────

@testset "Backends, devices, and AD types" begin
    for backend in (Flux, Lux)
        backend_name = string(backend)
        devices, adtypes = backend_config(backend)

        @testset "$backend_name backend" begin
            for estimator_type in (:point, :ratio, :posterior_mixture, :posterior_mixture_diagonal, :posterior_gaussian)
                est_label = string(estimator_type)

                @testset "$est_label estimator" begin
                    @testset "Forward pass (summarystatistics)" begin
                        est = make_estimator(backend, estimator_type)
                        @test begin
                            out = summarystatistics(est, Z_test; device = first(devices))
                            out !== nothing
                        end
                    end

                    @testset "Training" begin
                        for device in devices
                            device_name = nameof(typeof(device))
                            for adtype in adtypes
                                adtype_name = nameof(typeof(adtype))

                                for scenario in TRAINING_SCENARIOS
                                    for freeze in (true, false)
                                        test_label = "$device_name | $adtype_name | $(scenario.label) | freeze=$freeze"
                                        @testset "$test_label" begin
                                            est = make_estimator(backend, estimator_type)
                                            @test begin
                                                train(
                                                    est, scenario.args...;
                                                    adtype = adtype,
                                                    device = device,
                                                    freeze_summary_network = freeze,
                                                    savepath = mktempdir(),
                                                    epochs = 1,
                                                    verbose = false
                                                )
                                                true
                                            end broken=false
                                        end
                                    end
                                end
                            end
                        end
                    end

                    @testset "Inference" begin
                        est = make_estimator(backend, estimator_type)

                        if estimator_type === :point
                            @testset "estimate" begin
                                out = estimate(est, Z_single; device = first(devices))
                                @test out isa AbstractArray
                                @test size(out, 1) == d
                            end

                            @testset "assess (point)" begin
                                result = assess(est, θ_test, Z_test; device = first(devices))
                                @test result !== nothing
                            end

                        elseif estimator_type in (:posterior_gaussian, :posterior_mixture, :posterior_mixture_diagonal)
                            @testset "sampleposterior" begin
                                samples = sampleposterior(est, Z_single; device = first(devices))
                                @test samples isa AbstractArray
                            end

                            @testset "posteriormean" begin
                                pm = posteriormean(est, Z_single; device = first(devices))
                                @test pm isa AbstractArray
                                @test size(pm, 1) == d
                            end

                            @testset "assess (posterior)" begin
                                result = assess(est, θ_test, Z_test; device = first(devices))
                                @test result !== nothing
                            end

                        elseif estimator_type === :ratio
                            grid = expandgrid(0:0.01:1, 0:0.01:1)'

                            @testset "logratio" begin
                                lr = logratio(est, Z_single; grid = grid, device = first(devices))
                                @test lr isa AbstractArray
                            end

                            @testset "sampleposterior (ratio)" begin
                                samples = sampleposterior(est, Z_single; grid = grid, device = first(devices))
                                @test samples isa AbstractArray
                            end

                            @testset "posteriormean (ratio)" begin
                                pm = posteriormean(est, Z_single; grid = grid, device = first(devices))
                                @test pm isa AbstractArray
                                @test size(pm, 1) == d
                            end

                            @testset "assess (ratio)" begin
                                result = assess(est, θ_test, Z_test; grid = grid, device = first(devices))
                                @test result !== nothing
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset "No summary network" begin
    S = randn(Float32, d, 8)
    for backend in (Flux, Lux)
        est = PointEstimator(d; num_summaries = d, depth = 1, width = 8, backend = backend)
        est = backend === Lux ? LuxEstimator(est) : est
        out = estimate(est, S; use_gpu = false)
        @test size(out) == (d, 8)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Lux versions of estimators and distributions not covered by make_estimator()
# ──────────────────────────────────────────────────────────────────────────────

make_network(backend) = MLP(n, d; depth = 1, width = 16, backend = backend)

# NormalisingFlow and TelescopingRatioEstimator require runtime activity with Enzyme
const ADTYPES_RUNTIME_ACTIVITY = [AutoZygote(), AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))]

# Train on the CPU with each AD type supported by Lux
function train_lux_adtypes(est, θ_tr, θ_va, Z_tr, Z_va; adtypes = last(backend_config(Lux)))
    for adtype in adtypes
        @testset "$(nameof(typeof(adtype)))" begin
            est = train(est, θ_tr, θ_va, Z_tr, Z_va; adtype = adtype, device = cpu_device(), epochs = 1, verbose = false)
            @test est isa LuxEstimator
        end
    end
    return est
end

@testset "Lux NormalisingFlow" begin
    num_summaries = 3
    rng = Random.default_rng()
    flow = NormalisingFlow(d, num_summaries; backend = Lux)
    ps, st = Lux.setup(rng, flow)
    θ = randn(Float32, d, 16)
    tz = randn(Float32, num_summaries, 16)

    U, log_det_J, _ = NeuralEstimators.forward(flow, θ, tz, ps, st)
    @test size(U) == (d, 16)
    @test size(log_det_J) == (1, 16)
    # Invertibility is checked in Float64: with Lux's default initialisation the latent values can
    # be large across the coupling layers, and the Float32 round trip then loses precision
    θ64, tz64 = Float64.(θ), Float64.(tz)
    for use_act_norm in (true, false)
        f = NormalisingFlow(d, num_summaries; backend = Lux, use_act_norm = use_act_norm)
        ps64, st64 = Lux.setup(rng, f)
        ps64 = Lux.f64(ps64)
        U64, _, _ = NeuralEstimators.forward(f, θ64, tz64, ps64, st64)
        X, _ = NeuralEstimators.inverse(f, U64, tz64, ps64, st64)
        @test maximum(abs.(X - θ64)) < 1e-6
    end
    dens, _ = NeuralEstimators._logdensity(flow, θ, tz, ps, st)
    @test size(dens) == (1, 16)
    @test all(isfinite, dens)
    @test size(sampleposterior(flow, tz, 10, ps, st)) == (d, 10, 16)

    est = LuxEstimator(PosteriorEstimator(make_network(Lux), d; num_summaries = d, q = NormalisingFlow, depth = 1))
    est = train_lux_adtypes(est, θ_train, θ_val, Z_train, Z_val; adtypes = ADTYPES_RUNTIME_ACTIVITY)
    @test size(sampleposterior(est, Z_single; N = 50)) == (d, 50, 1)
    @test size(posteriormean(est, Z_single)) == (d, 1)
    @test assess(est, θ_test, Z_test; N = 50) isa Assessment
end

@testset "Lux SpikeAndSlab" begin
    num_summaries = 4
    sampler1(K) = NamedMatrix(θ = Float32.(rand(K) .< 0.5) .* randn(Float32, K))
    simulator1(θ::AbstractMatrix) = reduce(hcat, [θₖ[1] .+ sort(randn(Float32, n)) for θₖ in eachcol(θ)])
    θ1_train, θ1_val = sampler1(K), sampler1(K)
    Z1_train, Z1_val = simulator1(θ1_train), simulator1(θ1_val)
    Z1 = simulator1(sampler1(5))
    network() = MLP(n, num_summaries; depth = 1, width = 16, backend = Lux)

    q = SpikeAndSlab(1, num_summaries; backend = Lux)
    ps, st = Lux.setup(Random.default_rng(), q)
    θ_plain = reshape(Float32[0, 0.5, -0.3, 0, 1.2], 1, :) # spike (θ = 0) and slab entries
    tz = randn(Float32, num_summaries, 5)
    dens, _ = NeuralEstimators._logdensity(q, θ_plain, tz, ps, st)
    @test size(dens) == (1, 5)
    @test all(isfinite, dens)

    @test SpikeAndSlab(q.classifier, q.slab; spike = 1) isa SpikeAndSlab # constructor from a pre-built classifier and slab

    est = LuxEstimator(PosteriorEstimator(network(), q))
    est = train_lux_adtypes(est, θ1_train, θ1_val, Z1_train, Z1_val)
    @test size(sampleposterior(est, Z1; N = 50)) == (1, 50, 5)
    sp = spikeprobability(est, Z1)
    @test length(sp) == 5
    @test all(0 .<= sp .<= 1)

    # Positive-support slab via a NormalisingFlow with non-identity transform/invtransform
    q2 = SpikeAndSlab(1, num_summaries; slab = NormalisingFlow, transform = log, invtransform = exp, backend = Lux)
    est2 = LuxEstimator(PosteriorEstimator(network(), q2))
    θ2 = abs.(θ1_train)
    est2 = train(est2, θ2, θ2, simulator1(θ2), simulator1(θ2); epochs = 1, verbose = false)
    samples = sampleposterior(est2, Z1; N = 50)
    @test size(samples) == (1, 50, 5)
    @test all(samples .>= 0) # spike (0) or positive slab draws
end

# The test matrix above includes a ReactantDevice only when CUDA is functional. Here, Reactant falls
# back to the XLA CPU backend, so that the approximate distributions with dense covariance matrices
# (GaussianMixture and Gaussian, by default) are also tested with Reactant on CPU-only machines
# (e.g., the CI runners).
@testset "Lux $estimator_type: Reactant" for estimator_type in (:posterior_mixture, :posterior_gaussian)
    device = try
        Reactant.set_default_backend(CUDA.functional() ? "gpu" : "cpu")
        reactant_device()
    catch err
        @warn "Reactant backend unavailable, skipping the Reactant test of $estimator_type" err
        nothing
    end
    if !isnothing(device)
        # NB K divisible by the batchsize, so that `partial = false` (the default under a
        # ReactantDevice) does not drop validation samples
        K_r = 128
        θ_tr, θ_va = sampler(K_r), sampler(K_r)
        Z_tr, Z_va = simulator(θ_tr), simulator(θ_va)
        savepath = mktempdir()
        est0 = make_estimator(Lux, estimator_type)
        est = train(est0, θ_tr, θ_va, Z_tr, Z_va; device = device, epochs = 2, stopping_epochs = 3, savepath = savepath, verbose = false)
        @test est isa LuxEstimator
        history = loadrisk(savepath)
        @test all(isfinite, history)

        # The initial validation risk is computed by the compiled graph from the starting weights,
        # so it should match the risk of the same weights computed eagerly on the CPU
        # NB train() does not mutate Lux estimators
        # NB on the XLA GPU backend, single-precision results differ from the eager ones even when
        # everything is correct (see the pooled CNN test below; in double precision, the compiled
        # and eager log-densities agree to machine precision). Over 600 random initialisations of
        # the Gaussian estimator, the two risks differed by up to 2e-2 in relative terms on the XLA
        # GPU backend (99% quantile 9e-3), against 5e-7 on the XLA CPU backend. Hence, the check is
        # strict on the CPU, and only guards against gross errors on the GPU.
        rtol = CUDA.functional() ? 1.0f-1 : 1.0f-4
        @test -mean(est0((Z_va, θ_va.array))) ≈ history[1, 2] rtol = rtol
        @test size(sampleposterior(est, Z_single; N = 50)) == (d, 50, 1)
    end
end

@testset "Lux TelescopingRatioEstimator" begin
    lower, upper = [0.0f0, 0.0f0], [1.0f0, 1.0f0]
    grid = expandgrid(0:0.1:1, 0:0.1:1)'
    est = LuxEstimator(TelescopingRatioEstimator(make_network(Lux), d; num_summaries = d, sampler = sampler, depth = 1, width = 16))
    @test size(est(Z_test, θ_test)) == (d, size(θ_test, 2))
    est = train_lux_adtypes(est, θ_train, θ_val, Z_train, Z_val; adtypes = ADTYPES_RUNTIME_ACTIVITY)

    @test size(logratio(est, Z_single; grid = grid)) == (1, size(grid, 2))
    samples = sampleposterior(est, Z_single; lower = lower, upper = upper, N = 50)
    @test size(samples) == (d, 50, 1)
    @test all(lower .<= minimum(samples; dims = (2, 3))) && all(maximum(samples; dims = (2, 3)) .<= upper)
    @test size(logposterior(est, grid, Z_single; lower = lower, upper = upper)) == (size(grid, 2),)
    lp = logposterior(est, grid, Z_single; lower = lower, upper = upper, method = :chebyshev, degree = 16)
    @test size(lp) == (size(grid, 2),)
    @test all(isfinite, lp)
    @test_throws AssertionError logposterior(est, grid, Z_single; lower = lower, upper = upper, method = :other)
    @test assess(est, θ_test[:, 1:10], Z_test[:, 1:10]; lower = lower, upper = upper, N = 50) isa Assessment
end

@testset "Lux RatioEstimator: NUTS sampling" begin
    est = make_estimator(Lux, :ratio)
    lower, upper = [0.0f0, 0.0f0], [1.0f0, 1.0f0]
    samples = sampleposterior(est, Z_single; lower = lower, upper = upper, N = 30, warmup = 30)
    @test size(samples) == (d, 30, 1)
    @test all(lower .<= minimum(samples; dims = (2, 3))) && all(maximum(samples; dims = (2, 3)) .<= upper)
    samples = sampleposterior(est, Z_test[:, 1:2]; lower = lower, upper = upper, N = 30, warmup = 30)
    @test size(samples) == (d, 30, 2)
end

@testset "Lux early stopping" begin
    # Gradient ascent makes the validation risk increase (see the Flux test in general.jl)
    savepath = mktempdir()
    epochs = 20
    train(make_estimator(Lux, :point), θ_train, θ_val, Z_train, Z_val;
        optimiser = Descent(-1.0f-1), lr_schedule = nothing,
        epochs = epochs, stopping_epochs = 2, savepath = savepath, verbose = false)
    @test size(loadrisk(savepath), 1) < epochs + 1
end

# ──────────────────────────────────────────────────────────────────────────────
# Saving and loading the neural network
# ──────────────────────────────────────────────────────────────────────────────

"""
Return the (backend, device, adtype) combinations used to test saving and loading.

NB an explicit list is used rather than the cross product of `backend_config()`, since
`_resolve_adtype()` silently rewrites mismatched device/adtype combinations (e.g., AutoZygote()
is rewritten to AutoReactant() under a ReactantDevice), which would duplicate cases.
"""
function saveload_cases()
    cases = Any[
        (Flux, cpu_device(), AutoZygote()),
        (Lux, cpu_device(), AutoZygote()),
        (Lux, cpu_device(), AutoEnzyme())
    ]
    if CUDA.functional()
        push!(cases, (Flux, gpu_device(), AutoZygote()))
        push!(cases, (Lux, gpu_device(), AutoZygote()))
    end
    # Reactant: XLA GPU backend when CUDA is available, otherwise the XLA CPU backend
    try
        Reactant.set_default_backend(CUDA.functional() ? "gpu" : "cpu")
        device = reactant_device()
        nameof(typeof(device)) === :ReactantDevice || error("reactant_device() returned $(typeof(device))")
        push!(cases, (Lux, device, AutoReactant()))
    catch err
        @warn "Reactant backend unavailable, skipping Reactant save/load case" err
    end
    return cases
end

# Validation risk of an estimator, computed in the same way as the default loss (mae)
_saveload_risk(estimator, Z, θ) = mean(abs.(estimate(estimator, Z; device = cpu_device()) .- θ.array))

@testset "Saving and loading the neural network" begin
    # NB K divisible by the batchsize, so that `partial = false` (the default under a
    # ReactantDevice) does not drop validation samples, which would make the validation risk
    # recorded in loss_per_epoch.csv differ from the risk computed here
    K_sl = 128
    θ_tr, θ_va = sampler(K_sl), sampler(K_sl)
    Z_tr, Z_va = simulator(θ_tr), simulator(θ_va)
    epochs = 3

    for (backend, device, adtype) in saveload_cases()
        test_label = "$(nameof(backend)) | $(nameof(typeof(device))) | $(nameof(typeof(adtype)))"
        @testset "$test_label" begin
            savepath = mktempdir()
            trained = train(
                make_estimator(backend, :point), θ_tr, θ_va, Z_tr, Z_va;
                device = device, adtype = adtype, epochs = epochs,
                stopping_epochs = epochs + 1, savepath = savepath, verbose = false
            )
            # NB snapshot the trained estimates immediately: train() mutates Flux estimators in place
            out_trained = estimate(trained, Z_va; device = cpu_device())

            @testset "the neural network is saved" begin
                for prefix in ("best", "final")
                    @test isfile(joinpath(savepath, "$(prefix)_estimator.bson"))
                    @test isfile(joinpath(savepath, "$(prefix)_optimizer.bson"))
                    # the optimiser is also saved to tempdir(), so that it can be loaded without arguments
                    @test isfile(joinpath(tempdir(), "$(prefix)_optimizer.bson"))
                    # the combined checkpoint of previous versions is no longer saved
                    @test !isfile(joinpath(savepath, "$(prefix)_trainstate.bson"))
                end
                @test isfile(joinpath(savepath, "loss_per_epoch.csv"))
                @test isfile(joinpath(savepath, "train_time.csv"))
                @test size(loadrisk(savepath)) == (epochs + 1, 2)
                @test loadoptimiser(savepath) isa Optimisers.AbstractRule
                @test loadoptimiser(savepath; best = false) isa Optimisers.AbstractRule
            end

            @testset "the weights can be loaded after saving" begin
                fresh = make_estimator(backend, :point)
                out_fresh = estimate(fresh, Z_va; device = cpu_device())
                @test !isapprox(out_fresh, out_trained) # sanity check: an untrained network differs

                loaded = loadestimator(fresh, savepath)
                @test nameof(typeof(loaded)) === nameof(typeof(fresh))
                @test estimate(loaded, Z_va; device = cpu_device()) ≈ out_trained
                # loadestimator() does not modify the estimator it is given
                @test estimate(fresh, Z_va; device = cpu_device()) ≈ out_fresh

                # the parameters from the final epoch can also be loaded
                final = loadestimator(fresh, savepath; best = false)
                @test !isapprox(estimate(final, Z_va; device = cpu_device()), out_fresh)
            end

            @testset "the loaded weights are the optimised weights" begin
                fresh = make_estimator(backend, :point)
                loaded = loadestimator(fresh, savepath)
                final = loadestimator(fresh, savepath; best = false)
                risk_best = _saveload_risk(loaded, Z_va, θ_va)
                risk_final = _saveload_risk(final, Z_va, θ_va)
                # the risk of the loaded network is the best validation risk of the training run
                history = loadrisk(savepath)
                @test risk_best ≈ minimum(history[:, 2]) rtol = 1.0f-2
                @test risk_best <= risk_final + 1.0f-4

                # When the validation risk never improves, the best network is the one we started
                # from, not the final (diverged) one. NB train() mutates Flux estimators in place,
                # hence the deepcopy.
                savepath2 = mktempdir()
                diverged = train(
                    deepcopy(trained), θ_tr, θ_va, Z_tr, Z_va;
                    device = device, adtype = adtype, optimiser = Optimisers.Adam(10.0),
                    lr_schedule = nothing, epochs = 2, stopping_epochs = 3,
                    savepath = savepath2, verbose = false
                )
                history2 = loadrisk(savepath2)
                @test all(risk -> isnan(risk) || risk >= history2[1, 2], history2[:, 2])
                @test estimate(diverged, Z_va; device = cpu_device()) ≈ out_trained
                @test estimate(loadestimator(fresh, savepath2), Z_va; device = cpu_device()) ≈ out_trained
                @test !isapprox(estimate(loadestimator(fresh, savepath2; best = false), Z_va; device = cpu_device()), out_trained)
            end
        end
    end

    @testset "estimators containing Lux networks need not be wrapped" begin
        # NB make_estimator() wraps Lux estimators in a LuxEstimator, but users are not obliged
        # to do so: train() wraps them itself, and so must loadestimator()
        unwrapped() = PointEstimator(MLP(n, d; depth = 1, width = 16, backend = Lux), d; num_summaries = d, depth = 1)
        savepath = mktempdir()
        trained = train(
            unwrapped(), θ_tr, θ_va, Z_tr, Z_va;
            device = cpu_device(), adtype = AutoZygote(), epochs = 1, savepath = savepath, verbose = false
        )
        @test trained isa LuxEstimator
        out_trained = estimate(trained, Z_va; device = cpu_device())

        loaded = loadestimator(unwrapped(), savepath)
        @test loaded isa LuxEstimator
        @test estimate(loaded, Z_va; device = cpu_device()) ≈ out_trained
    end

    @testset "informative errors" begin
        savepath_lux = mktempdir()
        train(make_estimator(Lux, :point), θ_tr, θ_va, Z_tr, Z_va;
            device = cpu_device(), adtype = AutoZygote(), epochs = 1, savepath = savepath_lux, verbose = false)
        savepath_flux = mktempdir()
        train(make_estimator(Flux, :point), θ_tr, θ_va, Z_tr, Z_va;
            device = cpu_device(), adtype = AutoZygote(), epochs = 1, savepath = savepath_flux, verbose = false)

        # checkpoint saved with a different backend
        @test_throws ArgumentError loadestimator(make_estimator(Flux, :point), savepath_lux)
        @test_throws ArgumentError loadestimator(make_estimator(Lux, :point), savepath_flux)

        # same backend, different architecture
        wrong_flux = PointEstimator(MLP(n, d; depth = 1, width = 64, backend = Flux), d; num_summaries = d, depth = 1)
        wrong_lux = LuxEstimator(PointEstimator(MLP(n, d; depth = 1, width = 64, backend = Lux), d; num_summaries = d, depth = 1))
        @test_throws ArgumentError loadestimator(wrong_flux, savepath_flux)
        @test_throws ArgumentError loadestimator(wrong_lux, savepath_lux)

        # no saved estimator
        @test_throws AssertionError loadestimator(make_estimator(Lux, :point), mktempdir())
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Pooled CNN: the risk recorded during training must match the risk at inference
# ──────────────────────────────────────────────────────────────────────────────

# Regression test for a Reactant GPU miscompilation (present at Reactant 0.2.262, fixed in
# 0.2.290, which is now the compat floor). A `GlobalMeanPool` feeding a `Dense` layer was
# computed incorrectly -- 40-125% relative error -- whenever the layer's width equalled the
# batch size, i.e. whenever that layer's GEMM output was square, and the intermediate
# activation was not also returned as an output of the compiled graph. The effect was exact
# in the width/batchsize pairs tried: width 64 failed only at batchsize 64, width 128 only at
# 128, width 256 only at 256; batchsize 127 and 129 were both fine. CPU XLA was exact
# throughout. Hence `Dense(8, 128)` with `batchsize = 128` below -- change either and the test
# stops exercising the bug.
#
# Nothing in loss_per_epoch.csv gave this away, because `_risk` evaluates the same compiled
# graph that training used, whereas `estimate` and `sampleposterior` use the eager path. The
# only symptom was that the trained estimator scored far worse on the test set than its
# recorded validation risk implied.
#
# The MLP cases above cannot catch this: the architecture needs a pooling layer. On a CPU-only
# runner this test guards against regressions rather than reproducing the original failure.
@testset "pooled CNN: recorded risk matches the risk at inference" begin
    g = 16                 # grid side length
    K_cnn = 256            # divisible by the batchsize, so `partial = false` drops nothing
    batchsize = 128        # must equal the Dense width below; see the comment above
    epochs = 3

    # XLA uses TF32 for GEMMs on Ampere and later, while eager cuDNN does not, so a compiled
    # risk and an eager risk of the same weights differ by ~1e-3 in relative terms even when
    # everything is correct. The miscompilation this test guards against moved them apart by
    # ~1e-2, so the tolerances below sit between the two scales. Seeding keeps the TF32 noise
    # reproducible rather than redrawing it on every run.
    Random.seed!(2024)
    rtol_forward = 5.0f-3

    # NB rows of θ.array are (μ, σ), matching sampler() above
    function cnn_simulator(θ)
        A = θ.array
        Z = Array{Float32}(undef, g, g, 1, size(A, 2))
        for k in axes(A, 2)
            Z[:, :, 1, k] .= A[1, k] .+ A[2, k] .* randn(Float32, g, g)
        end
        return Z
    end

    make_cnn_estimator() = LuxEstimator(PointEstimator(
        Lux.Chain(
            Lux.Conv((3, 3), 1 => 8, Lux.relu, pad = 1),
            Lux.Conv((3, 3), 8 => 8, Lux.relu, pad = 1, stride = 2),
            Lux.GlobalMeanPool(),
            Lux.FlattenLayer(),
            Lux.Dense(8, 128, Lux.relu),
            Lux.Dense(128, d)
        ), d; num_summaries = d, depth = 1))

    θ_tr, θ_va = sampler(K_cnn), sampler(K_cnn)
    Z_tr, Z_va = cnn_simulator(θ_tr), cnn_simulator(θ_va)

    devices = Any[cpu_device()]
    CUDA.functional() && push!(devices, gpu_device())
    try
        Reactant.set_default_backend(CUDA.functional() ? "gpu" : "cpu")
        push!(devices, reactant_device())
    catch err
        @warn "Reactant backend unavailable, skipping the Reactant pooled-CNN case" err
    end

    for device in devices
        @testset "$(nameof(typeof(device)))" begin
            savepath = mktempdir()
            # The initial risk is the sharpest check available: loss_per_epoch.csv row 1 is the
            # validation risk of exactly these starting weights, computed by _risk on `device`,
            # so comparing it against the eager risk of the same estimator isolates the forward
            # pass with no training noise in the way. NB train() does not mutate Lux estimators.
            est0 = make_cnn_estimator()
            risk_initial_eager = _saveload_risk(est0, Z_va, θ_va)
            trained = train(
                est0, θ_tr, θ_va, Z_tr, Z_va;
                device = device, epochs = epochs, stopping_epochs = epochs + 1,
                batchsize = batchsize, savepath = savepath, verbose = false
            )
            history = loadrisk(savepath)

            @test risk_initial_eager ≈ history[1, 2] rtol = rtol_forward

            # The invariant: the estimator train() returns, evaluated eagerly, attains the best
            # validation risk that training recorded. Under the miscompilation the recorded risk
            # was far lower than the eager risk of the very same weights.
            @test _saveload_risk(trained, Z_va, θ_va) ≈ minimum(history[:, 2]) rtol = rtol_forward

            # ... and so does the checkpoint written for that epoch
            loaded = loadestimator(make_cnn_estimator(), savepath)
            @test _saveload_risk(loaded, Z_va, θ_va) ≈ minimum(history[:, 2]) rtol = rtol_forward
        end
    end
end
