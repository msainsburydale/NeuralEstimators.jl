"""
	ParameterConfigurations

An abstract supertype for storing parameters `θ` and any intermediate objects
needed for data simulation with `simulate`.
"""
abstract type ParameterConfigurations end

import Base: show
# this is used to handle a call to `print`
Base.show(io::IO, parameters::P) where {P <: ParameterConfigurations} = print(io, "\nA sub-type of `ParameterConfigurations` K = $(size(parameters, 2)) instances of the $(size(parameters, 1))-dimensional parameter vector θ.")
# this is used to show values in the REPL and when using IJulia
Base.show(io::IO, m::MIME"text/plain", parameters::P) where {P <: ParameterConfigurations} = print(io, parameters)

import Base: size
size(parameters::P) where {P <: ParameterConfigurations} = size(parameters.θ)
size(parameters::P, d::Integer) where {P <: ParameterConfigurations} = size(parameters.θ, d)

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


# FIXME This is a hack that will work for now, but I need to think of a general solution.
# ParameterLoader is only used when we're simulating just in time with fixed parameters;
# since I may end up only allowing full on-the-fly simulation (both the data
# and the parameters), I may not need this code... Only change it if I do.

function _getparameters(parameters::P, θ_idx) where {P <: ParameterConfigurations}

	K = size(parameters, 2)
	@assert maximum(θ_idx) <= K
	@assert length(θ_idx) == length(unique(θ_idx)) # all elements of idx are unique

	try
	    return _getparametersθ(parameters, θ_idx)
	catch
		try
			return _getparameterschols(parameters, θ_idx)
		catch
			error("The default indexing function _getparameters() has failed.")
		end
	end
end

function _getparametersθ(parameters::P, θ_idx) where {P <: ParameterConfigurations}

	θ = parameters.θ[:, θ_idx]

	return P(parameters, θ)
end




function _getparameterschols(parameters::P, θ_idx) where {P <: ParameterConfigurations}

	θ        = parameters.θ[:, θ_idx]
    chol_idx = parameters.chol_idx[θ_idx]
	chols    = parameters.chols[chol_idx]

	return P(parameters, θ, chols)
end


# returns parameters in d.indices[i+1:i+batchsize]
@propagate_inbounds function Base.iterate(d::_ParameterLoader, i = 0)
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
	batch = _getparameters(d.parameters, ids)
    return (batch, nexti)
end

function Base.length(d::_ParameterLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end
