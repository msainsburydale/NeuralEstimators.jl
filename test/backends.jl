using Test
using CUDA, cuDNN
using NeuralEstimators, ADTypes, Enzyme, Zygote, Reactant
using Lux
using Flux
using SimpleChains

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
        adtypes = push!(adtypes, AutoEnzyme())
        CUDA.functional() && push!(devices, gpu_device())
        return devices, adtypes
    elseif backend === SimpleChains
        return devices, adtypes
    else
        error("Unknown backend: $backend")
    end
end

"""Build a fresh estimator for the given backend and estimator type."""
function make_estimator(backend, estimator_type::Symbol)
    mod = backend === SimpleChains ? Lux : backend
    network = MLP(n, d; depth = 1, width = 16, backend = mod)

    if backend === SimpleChains
        network = ToSimpleChainsAdaptor(n)(network)
    end

    est = if estimator_type === :point
        PointEstimator(network)
    elseif estimator_type === :ratio
        RatioEstimator(network, d; num_summaries = d, depth = 1)
    elseif estimator_type === :posterior
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = GaussianMixture)
    else
        error("Unknown estimator type: $estimator_type")
    end

    mod === Lux ? (est |> LuxEstimator) : est
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

    # ── Backend × estimator type matrix ──────────────────────────────────────
    for backend in (Flux, Lux, SimpleChains)
        backend_name = string(backend)
        devices, adtypes = backend_config(backend)

        @testset "$backend_name backend" begin

            # ── Estimator types ───────────────────────────────────────────────
            for estimator_type in (:point, :ratio, :posterior)
                est_label = string(estimator_type)

                @testset "$est_label estimator" begin

                    # ── Forward pass ──────────────────────────────────────────
                    @testset "Forward pass (summarystatistics)" begin
                        est = make_estimator(backend, estimator_type)
                        @test begin
                            out = summarystatistics(est, Z_test; device = first(devices))
                            out !== nothing
                        end
                    end

                    # ── Training ──────────────────────────────────────────────
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
                                                true   # reached without exception
                                            end broken=false
                                        end
                                    end
                                end
                            end
                        end
                    end

                    # ── Inference ─────────────────────────────────────────────
                    @testset "Inference" begin
                        est = make_estimator(backend, estimator_type)
                        # Train for one epoch so weights are initialised properly
                        train(est, θ_train, θ_val, Z_train, Z_val;
                            adtype = first(adtypes),
                            device = first(devices),
                            epochs = 1,
                            verbose = false)

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

                        elseif estimator_type === :posterior
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
                    end  # Inference
                end  # estimator type
            end  # estimator_type loop
        end  # backend
    end  # backend loop
end  # NeuralEstimators
