##############################################################################################################
#                                             load packages
##############################################################################################################
using BenchmarkTools
using Distributions: Normal
using Distributions: Uniform
using Flux 
using NeuralEstimators
using NeuralEstimators: ActNorm
using NeuralEstimators: CouplingLayer
using NeuralEstimators: forward
using NeuralEstimators: inverse
using NeuralEstimators: Permutation
##############################################################################################################
#                                                   setup
##############################################################################################################
n = 2        # dimension of each data replicate 
m = 5        # number of independent replicates 
d = 2        # dimension of the parameter vector θ
w = 128      # width of each hidden layer 

function sample(K)
    μ = rand(Normal(0, 10f0), K)
    σ = rand(Uniform(0, 10f0), K)
    θ = vcat(μ', σ')
    return θ
end

simulate(θ, m) = [ϑ[1] .+ ϑ[2] * randn(Float32, n, m) for ϑ ∈ eachcol(θ)]

SUITE = BenchmarkGroup()
##############################################################################################################
#                                             PointEstimator
##############################################################################################################
SUITE[:PointEstimator] = BenchmarkGroup()

final_layer = Parallel(
    vcat,
    Dense(w, 1, identity),     # μ ∈ ℝ
    Dense(w, 1, softplus)      # σ > 0
)

estimator = PointEstimator(DeepSet(
    Chain(Dense(n, w, relu), Dense(w, d, relu)), 
    Chain(Dense(d, w, relu), final_layer)
))

SUITE[:PointEstimator][:train] = @benchmarkable(
    train($estimator, $sample, simulate, epochs = 5, m = $m),
    seconds = 15,
)

SUITE[:PointEstimator][:estimate] = @benchmarkable(
    estimate($estimator, data),
    setup = (data = simulate([5, 0.3f0], $m))
)

for B ∈ [1000, 5000]
    SUITE[:PointEstimator][:bootstrap,B] = @benchmarkable(
        bootstrap($estimator, data; B = $B),
        setup = (data = simulate([5, 0.3f0], $m))
    )
end

SUITE[:PointEstimator][:assess] = @benchmarkable(
    assess($estimator, prior_samples, data),
    seconds = 10,
    setup = (
        prior_samples = sample(1000); 
        data = simulate(prior_samples, $m)
    )
)
##############################################################################################################
#                                             RatioEstimator
##############################################################################################################
SUITE[:RatioEstimator] = BenchmarkGroup()

estimator = RatioEstimator(DeepSet(
    Chain(Dense(n, w, relu), Dense(w, w, relu), Dense(w, w, relu)), 
    Chain(Dense(w + d, w, relu), Dense(w, w, relu), Dense(w, 1))
))

SUITE[:RatioEstimator][:train] = @benchmarkable(
    train($estimator, $sample, simulate, epochs = 5, m = $m),
    seconds = 30,
)

SUITE[:RatioEstimator][:sampleposterior] = @benchmarkable(
    sampleposterior($estimator, data),
    setup = (data = simulate([5, 0.3f0], $m))
)

SUITE[:RatioEstimator][:mlestimate] = @benchmarkable(
    mlestimate($estimator, data; θ_grid),
    setup = (
        data = simulate([5, 0.3f0], $m);
        θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'    
    )
)

SUITE[:RatioEstimator][:posteriormean] = @benchmarkable(
    posteriormean($estimator, data; θ_grid),
    setup = (
        data = simulate([5, 0.3f0], $m);
        θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'    
    )
)

SUITE[:RatioEstimator][:posteriormode] = @benchmarkable(
    posteriormode($estimator, data; θ_grid),
    setup = (
        data = simulate([5, 0.3f0], $m);
        θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'    
    )
)

SUITE[:RatioEstimator][:posteriormedian] = @benchmarkable(
    posteriormedian($estimator, data; θ_grid),
    setup = (
        data = simulate([5, 0.3f0], $m);
        θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'    
    )
)

SUITE[:RatioEstimator][:sampleposterior] = @benchmarkable(
    sampleposterior($estimator, data; θ_grid),
    setup = (
        data = simulate([5, 0.3f0], $m);
        θ_grid = f32(expandgrid(0:0.01:1, 0:0.01:1))'    
    )
)
##############################################################################################################
#                                             PosteriorEstimator
##############################################################################################################
SUITE[:PosteriorEstimator] = BenchmarkGroup()

estimator = PosteriorEstimator(
    NormalisingFlow(d, 2 * d), 
    DeepSet(
        Chain(Dense(n, w, relu), Dense(w, w, relu)),
        Chain(Dense(w, w, relu), Dense(w, 2 * d))
    )   
)

SUITE[:PosteriorEstimator][:train] = @benchmarkable(
    train($estimator, $sample, simulate, epochs = 5, m = $m),
    seconds = 30,
)

SUITE[:PosteriorEstimator][:sampleposterior] = @benchmarkable(
    sampleposterior($estimator, data),
    setup = (data = simulate([5, 0.3f0], $m))
)

SUITE[:PosteriorEstimator][:sampleposterior_multipledatasets] = @benchmarkable(
    sampleposterior($estimator, data),
    setup = (data = repeat(simulate([5, 0.3f0], $m), 500))
)

SUITE[:PosteriorEstimator][:posteriormean] = @benchmarkable(
    posteriormean($estimator, data),
    setup = (data = simulate([5, 0.3f0], $m))
)

SUITE[:PosteriorEstimator][:posteriormedian] = @benchmarkable(
    posteriormedian($estimator, data),
    setup = (data = simulate([5, 0.3f0], $m))
)

SUITE[:PosteriorEstimator][:posteriorquantile] = @benchmarkable(
    posteriorquantile($estimator, data, probs),
    setup = (data = simulate([5, 0.3f0], $m); probs = [.025, .975])
)

SUITE[:PosteriorEstimator][:assess] = @benchmarkable(
    assess($estimator, prior_samples, data),
    seconds = 10,
    setup = (
        prior_samples = sample(1000); 
        data = simulate(prior_samples, $m)
    )
)
##############################################################################################################
#                                             assessment functions
##############################################################################################################
# assement functions are performed once---I don't think assessment functions depend on neural estimation method
SUITE[:assessments] = BenchmarkGroup()
# create assessment; too slow for setup
prior_samples = sample(1000)
data = simulate(prior_samples, m)
assessment = assess(estimator, prior_samples, data)

SUITE[:assessments][:risk] = @benchmarkable(
    risk($assessment),
)

SUITE[:assessments][:bias] = @benchmarkable(
    bias($assessment),
)

SUITE[:assessments][:rmse] = @benchmarkable(
    rmse($assessment),
)
##############################################################################################################
#                                             Layers
##############################################################################################################
SUITE[:layers] = BenchmarkGroup()
SUITE[:layers][:compress] = @benchmarkable(
    l(θ),
    setup = (
        a = [25, 0.5, -pi/2];
        b = [500, 2.5, 0];
        p = length(a);
        K = 100;
        θ = randn(p, K);
        l = Compress(a, b);
    )
)

for m ∈ [10, 100]
    SUITE[:layers][:deepset,m] = @benchmarkable(
        deepset(data),
        setup = (
            d = 4;
            n = 100;
            w = 128;
            deepset = DeepSet(
                Chain(Dense(n, w, $relu), Dense(w, w, $relu)),
                Chain(Dense(w, w, $relu), Dense(w, 2 * d))
            );
            data = [rand32(n, $m)] 
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:AffineCouplingBlock][:foward,K] = @benchmarkable(
        forward(layer, θ2, θ1, TZ),
        setup = (
            d₁ = 100;
            dstar = 50; 
            d₂ = 100;
            θ1  = rand32(d₁, $K);
            θ2  = rand32(d₁, $K);
            TZ  = rand32(dstar, $K);
            layer = AffineCouplingBlock(d₁, dstar, d₂);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:AffineCouplingBlock][:inverse,K] = @benchmarkable(
        inverse(layer, θ1, U2[1], TZ),
        setup = (
            d₁ = 100;
            dstar = 50;
            d₂ = 100;
            θ1  = rand32(d₁, $K);
            θ2  = rand32(d₁, $K);
            TZ  = rand32(dstar, $K);
            layer = AffineCouplingBlock(d₁, dstar, d₂);
            U2 = forward(layer, θ2, θ1, TZ); 
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:CouplingLayer][:forward,K] = @benchmarkable(
        forward(layer, θ, TZ),
        setup = (
            d = 100;
            dstar = 50;
            θ = rand32(d, $K);
            TZ = rand32(dstar, $K);
            layer = CouplingLayer(d, dstar);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:CouplingLayer][:inverse,K] = @benchmarkable(
        inverse(layer, U[1], TZ),
        setup = (
            d = 100;
            dstar = 50;
            θ = rand32(d, $K);
            TZ = rand32(dstar, $K);
            layer = CouplingLayer(d, dstar);
            U = forward(layer, θ, TZ); 
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:NormalisingFlow][:forward,K] = @benchmarkable(
        forward(layer, θ, TZ),
        setup = (
            d = 100;
            dstar = 50;
            θ = rand32(d, $K);
            TZ = rand32(dstar, $K);
            layer = NormalisingFlow(d, dstar);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:NormalisingFlow][:inverse,K] = @benchmarkable(
        inverse(layer, U[1], TZ),
        setup = (
            d = 100;
            dstar = 50;
            θ = rand32(d, $K);
            TZ = rand32(dstar, $K);
            layer = NormalisingFlow(d, dstar);
            U = forward(layer, θ, TZ);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:NormalisingFlow][:logdensity,K] = @benchmarkable(
        logdensity(layer, θ, TZ),
        setup = (
            d = 100;
            dstar = 50;
            θ = rand32(d, $K);
            TZ = rand32(dstar, $K);
            layer = NormalisingFlow(d, dstar);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:Permutation][:forward,K] = @benchmarkable(
        forward(layer, θ),
        setup = (
            d = 100;
            θ = rand32(d, $K);
            layer = Permutation(d);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:Permutation][:inverse,K] = @benchmarkable(
        inverse(layer, U),
        setup = (
            d = 100;
            θ = rand32(d, $K);
            layer = Permutation(d);
            U = forward(layer, θ);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:ActNorm][:forward,K] = @benchmarkable(
        forward(layer, θ),
        setup = (
            d = 100;
            θ = rand32(d, $K);
            layer = ActNorm(d);
        )
    )
end

for K ∈ [5, 10]
    SUITE[:layers][:ActNorm][:inverse,K] = @benchmarkable(
        inverse(layer, U[1]),
        setup = (
            d = 100;
            θ = rand32(d, $K);
            layer = ActNorm(d);
            U = forward(layer, θ);
        )
    )
end

results = run(SUITE) 
