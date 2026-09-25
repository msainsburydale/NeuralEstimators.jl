using Test
using CUDA, cuDNN
using NeuralEstimators, ADTypes, Enzyme, Zygote, Reactant
using Optimisers
using Lux
using Flux
using SimpleChains
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
        PointEstimator(network, d; num_summaries = d, depth = 1)
    elseif estimator_type === :ratio
        RatioEstimator(network, d; num_summaries = d, depth = 1)
    elseif estimator_type === :posterior_mixture
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = GaussianMixture)
    elseif estimator_type === :posterior_gaussian
        PosteriorEstimator(network, d; num_summaries = d, depth = 1, q = Gaussian)
    else
        error("Unknown estimator type: $estimator_type")
    end

    mod === Lux ? (est |> LuxEstimator) : est
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
    for backend in (Flux, Lux, SimpleChains)
        backend_name = string(backend)
        devices, adtypes = backend_config(backend)

        @testset "$backend_name backend" begin
            for estimator_type in (:point, :ratio, :posterior_mixture, :posterior_gaussian)
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

                                # Skip Reactant for posterior_gaussian (triangular solve causing issues with XLA)
                                if estimator_type === :posterior_gaussian && nameof(typeof(device)) === :ReactantDevice
                                    continue
                                end

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

                        elseif estimator_type === :posterior_gaussian || estimator_type === :posterior_mixture
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
