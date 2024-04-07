"""
	ParameterConfigurations

An abstract supertype for user-defined types that store parameters and any
intermediate objects needed for data simulation.

The user-defined type must have a field `θ` that stores the ``p`` × ``K`` matrix
of parameters, where ``p`` is the number of parameters in the model and ``K`` is the
number of parameter vectors sampled from the prior distribution. There are no
other restrictions.

See [`subsetparameters`](@ref) for the generic function for subsetting these objects.

# Examples

```
struct P <: ParameterConfigurations
	θ
	# other expensive intermediate objects...
end
```
"""
abstract type ParameterConfigurations end

Base.show(io::IO, parameters::P) where {P <: ParameterConfigurations} = print(io, "\nA subtype of `ParameterConfigurations` with K = $(size(parameters, 2)) instances of the $(size(parameters, 1))-dimensional parameter vector θ.")
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: ParameterConfigurations} = print(io, parameters)

size(parameters::P) where {P <: ParameterConfigurations} = size(_extractθ(parameters))
size(parameters::P, d::Integer) where {P <: ParameterConfigurations} = size(_extractθ(parameters), d)

_extractθ(params::P) where {P <: ParameterConfigurations} = params.θ
_extractθ(params::P) where {P <: AbstractMatrix} = params

"""
	subsetparameters(parameters::M, indices) where {M <: AbstractMatrix}
	subsetparameters(parameters::P, indices) where {P <: ParameterConfigurations}

Subset `parameters` using a collection of `indices`.

Arrays in `parameters::P` with last dimension equal in size to the
number of parameter configurations, K, are also subsetted (over their last dimension)
using `indices`. All other fields are left unchanged. To modify this default
behaviour, overload `subsetparameters`.
"""
function subsetparameters(parameters::P, indices) where {P <: ParameterConfigurations}

	K = size(parameters, 2)
	@assert maximum(indices) <= K

	fields = [getfield(parameters, name) for name ∈ fieldnames(P)]
	fields = map(fields) do field

		try
			N = ndims(field)
			if size(field, N) == K
				colons  = ntuple(_ -> (:), N - 1)
				field[colons..., indices]
			else
				field
			end
		catch
			field
		end

	end
	return P(fields...)
end

function subsetparameters(parameters::M, indices) where {M <: AbstractMatrix}

	K = size(parameters, 2)
	@assert maximum(indices) <= K

	return parameters[:, indices]
end

# wrapper that allows for indices to be a single Integer
subsetparameters(θ::P, indices::Integer) where {P <: ParameterConfigurations} = subsetparameters(θ, indices:indices)
subsetparameters(θ::M, indices::Integer) where {M <: AbstractMatrix} = subsetparameters(θ, indices:indices)


# ---- _ParameterLoader: Analogous to DataLoader for ParameterConfigurations objects ----

struct _ParameterLoader{P <: Union{AbstractMatrix, ParameterConfigurations}, I <: Integer}
    parameters::P
    batchsize::Integer
    nobs::Integer
    partial::Bool
    imax::Integer
    indices::Vector{I}
    shuffle::Bool
end

function _ParameterLoader(parameters::P; batchsize::Integer = 1, shuffle::Bool = false, partial::Bool = true) where {P <: ParameterConfigurations}
    @assert batchsize > 0
    K = size(parameters, 2)
    if K <= batchsize batchsize = K end
    imax = partial ? K : K - batchsize + 1 # imax ≡ the largest index that we go to
    _ParameterLoader(parameters, batchsize, K, partial, imax, [1:K;], shuffle)
end

# returns parameters in d.indices[i+1:i+batchsize]
@propagate_inbounds function Base.iterate(d::_ParameterLoader, i = 0)
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    indices = d.indices[i+1:nexti]
	batch = subsetparameters(d.parameters, indices)

	try
		batch = subsetparameters(d.parameters, indices)
	catch
		error("The default method for `subsetparameters` has failed; please see `?subsetparameters` for details.")
	end

    return (batch, nexti)
end
