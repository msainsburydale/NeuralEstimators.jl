# More complicated example

In this example, we'll consider a standard spatial model, the linear Gaussian-Gaussian model,

```math
Z_i = Y(\mathbf{s}_i) + \epsilon_i, \quad  i = 1, \dots, n,
```

where ``\mathbf{Z} \equiv (Z_1, \dots, Z_n)^\top`` are data observed at locations ``\{\mathbf{s}_1, \dots, \mathbf{s}_n\} \subset \mathcal{D}``, ``Y(\cdot)`` is a spatially-correlated mean-zero Gaussian process, and ``\epsilon_i \sim \rm{N}(0, \sigma^2_\epsilon)`` is Gaussian white noise with ``\sigma^2_\epsilon`` the measurement-error variance parameter. An important component of the model is the covariance function, ``C(\mathbf{s}, \mathbf{u}) \equiv \rm{cov}(Y(\mathbf{s}), Y(\mathbf{u}))``, for ``\mathbf{s}, \mathbf{u} \in \mathcal{D}``, which is the primary mechanism for capturing spatial dependence. Here, we use the popular isotropic Matérn covariance function,

```math
 C(\mathbf{h}) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\frac{\|\mathbf{h}\|}{\rho}\right) K_\nu \left(\frac{\|\mathbf{h}\|}{\rho}\right),
```

 where ``\sigma`` is the marginal variance parameter, ``\Gamma(\cdot)`` is the gamma function, ``K_\nu(\cdot)`` is the Bessel function of the second kind of order ``\nu``, and ``\rho > 0`` and ``\nu > 0`` are the range and smoothness parameters, respectively. We follow the common practice decision to fix ``\sigma`` to 1, and we also fix ``\nu = 1`` for simplicity. This leaves two unknown parameters that need to be estimated: ``\mathbf{\theta} \equiv (\sigma_\epsilon, \rho)^\top``.

 The invariant model objects in this example are the prior distribution, the spatial locations, and the distance matrix. We'll assume prior independence between the parameters with log-uniform margins. We'll take our spatial domain of interest, ``\mathcal{D}``, to be ``[0, 9] \times [0, 9]``, and we'll simulate data on a regular grid with neighbours in each dimension separated by 1 unit.

```
using NeuralEstimators
using Distributions
Ω = (
 	σₑ = LogUniform(0.05, 0.5),
	ρ  = LogUniform(2, 6)
)
S = expandgrid(1:9, 1:9)
S = Float32.(S)
D = pairwise(Euclidean(), S, S, dims = 1)
ξ = (Ω = Ω, S = S, D = D)
```

 Next, we define a subtype of [`ParameterConfigurations`](@ref). For the current model, Cholesky factors of covariance matrices are a key intermediate object needed for data simulation, so they are included in addition to the compulsory field `θ`.

```
struct Parameters <: ParameterConfigurations
 	θ
	chols
end
```

We then define a `Parameters` constructor, returning a `Parameters` object with `K` parameters and corresponding Choleksy factors: Below, we employ the function [`maternchols(D, ρ, ν)`](@ref).

```
function Parameters(ξ, K::Integer)

 Ω  = ξ.Ω
 σₑ = rand(Ω.σₑ, 1, K)
 ρ  = rand(Ω.ρ, 1, K)
 θ  = vcat(σₑ, ρ)

 chols = maternchols(ξ.D, ρ, 1)

 Parameters(ξ, chols)
end
```

 Next, we implicitly define the statistical model by overloading [`simulate`](@ref):

```
import NeuralEstimators: simulate
function simulate(parameters::Parameters, ξ, m::Integer)

	L = parameters.chols
	n = size(L, 1)
	K = size(parameters, 2)

	# For the kth parameter configuration, extract the corresponding Cholesky
	# factor, initialise an array to store the m replicates, and simulate from
	# the model:
	Z = map(1:K) do k
		Lₖ = L[:, :, k]
		z = similar(Lₖ, n, m)
		for i ∈ 1:m
			z[:, i] = Lₖ *  randn(T, n) + σₑ * randn(T, n)
		end
		return z
	end

	# Convert fields to a square domain and add a singleton dimension for
	# compatibility with Flux:
	@assert isqrt(n) == sqrt(n)
	Z = reshape.(Z, isqrt(n), isqrt(n), 1, :)  

	return Z
end
```

 We then choose an architecture for modelling ψ(⋅) and ϕ(⋅) in the Deep Set framework, and initialise the neural estimator as a [`DeepSet`](@ref) object. Note that functions for defining architectures based on the number of parameters in the statistical model, p, can be useful when working with several models.

```
function architecture(p)

 ψ = Chain(
	 Conv((10, 10), 1 => 32,  relu),
	 Conv((5, 5),  32 => 64,  relu),
	 Conv((3, 3),  64 => 128, relu),
	 Flux.flatten,
	 Dense(128, 256, relu)
	 )

 ϕ = Chain(
	 Dense(256, 128, relu),
	 Dense(128, 64, relu),
	 Dense(64, 32, relu),
	 Dense(32, p),
	 x -> exp.(x)
 )

 return ψ, ϕ
end

p = 2
ψ, ϕ = architecture(p)
θ̂ = DeepSet(ψ, ϕ)
```

Next, we train the neural estimator using [`train`](@ref). For this model, generating `Parameters` is somewhat expensive due to computation of Choleksy factors. Hence, it can be computationally advantageous to keep the training and validation parameter sets fixed during training, and this is achieved by providing these sets to [`train`](@ref):

```
θ_train = Parameters(ξ, 5000)
θ_val   = Parameters(ξ, 500)
θ̂ = train(θ̂, ξ, Parameters, m = 10)
```

Note that the above set sizes may be too low to obtain an optimal estimator and, with the current implementation, we are somewhat restricted to using modest set sizes: For a general way to cope with this challenge, see [Sharing intermediate objects between parameter configurations](@ref).

Irrespective of the model, the functions [`estimate`](@ref) and [`merge`](@ref) are used as described previously.
