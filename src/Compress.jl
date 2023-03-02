"""
    Compress(a, b)

Uses the scaled logistic function to compress the output of a neural network to
be between `a` and `b`.

The elements of `a` should be less than the corresponding element of `b`.

# Examples
```
using NeuralEstimators
using Flux

p = 3
a = [0.1, 4, 2]
b = [0.9, 9, 3]
l = Compress(a, b)
K = 10
θ = rand(p, K)
l(θ)

n = 20
Z = rand(n, K)
θ̂ = Chain(Dense(n, 15), Dense(15, p), l)
θ̂(Z)

Z = Z |> gpu
θ̂ = θ̂ |> gpu
θ̂(Z)
```
"""
struct Compress{T}
  a::T
  b::T
  m::T
end
Compress(a, b) = Compress(a, b, (b + a) / 2)

(l::Compress)(θ) = l.a .+ (l.b - l.a) ./ (one(eltype(θ)) .+ exp.(-(θ .- l.m)))

Flux.@functor Compress
Flux.trainable(l::Compress) =  ()
