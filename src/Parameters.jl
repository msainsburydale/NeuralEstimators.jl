"""
	ParameterConfigurations

An abstract supertype for storing parameters `θ` and any intermediate objects
needed for data simulation with `simulate`.
"""
abstract type ParameterConfigurations end

import Base: show
# this is used to handle a call to `print`
Base.show(io::IO, parameters::P) where {P <: ParameterConfigurations} = print(io, "\nA subtype of `ParameterConfigurations` with K = $(size(parameters, 2)) instances of the $(size(parameters, 1))-dimensional parameter vector θ.")
# this is used to show values in the REPL and when using IJulia
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: ParameterConfigurations} = print(io, parameters)

import Base: size
size(parameters::P) where {P <: ParameterConfigurations} = size(parameters.θ)
size(parameters::P, d::Integer) where {P <: ParameterConfigurations} = size(parameters.θ, d)

"""
	subsetparameters(parameters::Parameters, indices) where {Parameters <: ParameterConfigurations}
Subset `parameters` using a collection of `indices`.

The default method assumes that each field of `parameters` is an array with the
last dimension corresponding to the parameter configurations (i.e., it subsets
over the last dimension of each array). If this is not the case, define an
appropriate subsetting method by overloading `subsetparameters` after running
`import NeuralEstimators: subsetparameters`.
"""
function subsetparameters(parameters::Parameters, indices) where {Parameters <: ParameterConfigurations}

	K = size(parameters, 2)
	@assert maximum(indices) <= K

	fields = [getfield(parameters, name) for name ∈ fieldnames(Parameters)]
	fields = map(fields) do field
		colons  = ntuple(_ -> (:), ndims(field) - 1)
		field[colons..., indices]
	end
	return Parameters(fields...)
end

# ---- _ParameterLoader: Analogous to DataLoader for ParameterConfigurations objects ----

struct _ParameterLoader{P <: ParameterConfigurations, I <: Integer}
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
    if K < batchsize batchsize = K end
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

function Base.length(d::_ParameterLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int, n) : floor(Int, n)
end
