"""
	AbstractParameterSet

An abstract supertype for user-defined types that store parameters and any auxiliary objects needed for data simulation.

The user-defined type must have a field `θ` that stores the parameters. Typically, 
`θ` is a ``d`` × ``K`` matrix, where ``d`` is the dimension of the
parameter vector and ``K`` is the number of sampled parameter vectors, though
any batchable object compatible with `numobs`/`getobs` is supported. There are no other requirements.

The number of parameter instances can be retrieved with `numobs`, and the size of `θ` can be inspected with `size`. 

Subtypes of `AbstractParameterSet` support indexing via `Base.getindex`, 
with any batchable fields subsetted accordingly and all other fields left unchanged.
To modify this default behaviour, provide a specific `Base.getindex` method for your concrete subtype.

# Examples
```julia
struct Parameters <: AbstractParameterSet
	θ
	# auxiliary objects needed for data simulation
end

θ = randn(2, 100)
parameters = Parameters(θ)
numobs(parameters)   # 100
size(parameters)     # (2, 100)
parameters[1:10]     # subset of 10 parameter vectors
```
"""
abstract type AbstractParameterSet end

_extractθ(parameters::AbstractParameterSet) = parameters.θ
_extractθ(parameters) = parameters
numobs(parameters::AbstractParameterSet) = numobs(_extractθ(parameters))

Base.getindex(parameters::AbstractParameterSet, i::Integer) = Base.getindex(parameters, i:i)
function Base.getindex(parameters::P, idx) where {P <: AbstractParameterSet}

    maximum(idx) <= numobs(parameters) || throw(BoundsError(parameters, idx))

    fields = map(fieldnames(P)) do name
        field = getfield(parameters, name)
        try
            getobs(field, idx)
        catch
            field
        end
    end

    return P(fields...)
end

size(parameters::AbstractParameterSet) = size(_extractθ(parameters))
size(parameters::AbstractParameterSet, d) = size(_extractθ(parameters), d)

Base.show(io::IO, parameters::P) where {P <: AbstractParameterSet} = print(io, "\nA subtype of `AbstractParameterSet` with $(numobs(parameters)) parameter instances")
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: AbstractParameterSet} = print(io, parameters)

# Backwards compatability
const ParameterConfigurations = AbstractParameterSet
export ParameterConfigurations
