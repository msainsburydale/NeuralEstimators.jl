"""
	AbstractParameterSet

An abstract supertype for user-defined types that store parameters and any
intermediate objects needed for data simulation.

The user-defined type must have a field `θ` that stores the ``d`` × ``K`` matrix
of parameters, where ``d`` is the dimension of the parameter vector to make 
inference on and ``K`` is the number of sampled parameter vectors. There are no
other requirements.

The user-defined type must have a field `θ` that stores the parameters. Typically, 
`θ` is a ``d`` × ``K`` matrix, where ``d`` is the dimension of the
parameter vector and ``K`` is the number of sampled parameter vectors, though
any batchable object compatible with `numobs`/`getobs` from MLUtils.jl is
supported. There are no other requirements.

Objects of type `P <: AbstractParameterSet` are indexed using `getindex`/`getobs`, with any batchable fields indexed accordingly (all other fields are left unchanged). 
To modify this default behaviour, provide a specific method Base.getindex(parameters::P, idx) for your concrete type `P <: AbstractParameterSet`.

# Examples
```julia
struct P <: AbstractParameterSet
	θ
	# other expensive intermediate objects...
end
```
"""
abstract type AbstractParameterSet end

_extractθ(parameters::AbstractParameterSet) = parameters.θ
_extractθ(parameters) = parameters
numobs(parameters::AbstractParameterSet) = numobs(_extractθ(parameters))

Base.getindex(parameters::AbstractParameterSet, i::Integer) = Base.getindex(parameters, i:i) 
function Base.getindex(parameters::P, idx) where {P <: AbstractParameterSet}

    @assert maximum(idx) <= numobs(parameters) "Index out of bounds: attempted to access observation $(maximum(idx)) from a parameter set with $(numobs(parameters)) observations."

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

Base.show(io::IO, parameters::P) where {P <: AbstractParameterSet} = print(io, "\nA subtype of `AbstractParameterSet` with K = $(size(parameters, 2)) instances of the $(size(parameters, 1))-dimensional parameter vector")
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: AbstractParameterSet} = print(io, parameters)

# Backwards compatability
const ParameterConfigurations = AbstractParameterSet
export ParameterConfigurations