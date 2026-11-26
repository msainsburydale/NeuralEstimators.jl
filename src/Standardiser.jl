@doc raw"""
	Standardiser(mean, std)
	Standardiser(x)
The location ``\mu`` (`mean`) and scale ``\sigma`` (`std`) used to standardise the data or the parameters of an estimator.

Every estimator that takes data standardises them before they enter the neural networks (keyword argument
`standardise_data`), and every estimator also standardises the parameters
(keyword argument `standardise_parameters`), except a [`PointEstimator`](@ref), which by default
(`standardise_parameters = :auto`) standardises the parameters only if this leaves the estimator unchanged. Each of these keyword arguments may be:
- `true`: the statistics are computed by [`train`](@ref) from the training data;
- `false`: no standardisation;
- a `Standardiser`, with statistics given by the user;
- for data stored as [`DataAndSummaries`](@ref) objects, a named tuple `(Z = …, S = …)` of these values, one for the data `Z` and one for the expert summary statistics `S` (e.g., `(Z = Standardiser(d.Z), S = true)`). With `true`, both are standardised, while a single `Standardiser` applies to `Z` only.

`Standardiser(mean, std)` stores statistics given by the user. Each is a number, a vector with one element per row of the
data or parameters, or an array that broadcasts against a batch of data (e.g., with size ``1 × 1 × C × 1`` for one
statistic per channel of images stored as a ``W × H × C × K`` array).

A standard deviation that is zero, or too small relative to the mean for the standardised values to be resolved in single
precision, is set to one. Data that are not supported (e.g., graphs) or that contain non-finite values are not standardised.
Data stored in a type defined by the user can be supported by adding methods to `s(Z)`, which applies a `Standardiser`
`s` to data `Z`, and to [`NeuralEstimators.fitstandardiser`](@ref) (see [Standardisation](@ref advanced-standardisation)).

Standardisation is a fixed bijective map, so it does not change the quantity being estimated. The exception is a neural Bayes
estimator ([`PointEstimator`](@ref)): the Bayes estimator is unchanged if the loss is a
sum over the parameters of terms that each depend on the error of an estimate only, through a positively homogeneous
function (e.g., `mae`, `mse`, and the quantile loss), but not otherwise (e.g., for [`tanhloss`](@ref)).

The statistics are saved with the estimator (see [`train`](@ref) and
[`saveestimator`](@ref)) and restored by [`loadestimator`](@ref), and they are never recomputed once training has begun.

See also [`datastandardiser`](@ref).

# Examples
```julia
using NeuralEstimators

θ = rand(2, 1000)
s = Standardiser(θ)                         # one mean and standard deviation per parameter
s = Standardiser([0.5, 0.5], [0.29, 0.29])  # statistics given by the user
s.mean, s.std
```
"""
struct Standardiser{M}
    mean::M
    std::M
    Standardiser{M}(mean::M, std::M) where {M} = new{M}(mean, std)
end

# An unfitted standardiser (specified by `true`), which train() replaces by one fitted to the training data. It is the
# identity until then, so that an untrained estimator can still be applied (e.g., to check the dimensions of its output)
Standardiser() = Standardiser{Nothing}(nothing, nothing)

# The default parameter standardiser of a PointEstimator (`standardise_parameters = :auto`), which train() replaces by one
# fitted to the training parameters if this leaves the estimator unchanged, and by `nothing` otherwise (see
# _autostandardiser). Like an unfitted standardiser, it is the identity until then
struct _AutoStandardiser end

function Standardiser(mean, std)
    μ, σ = _statistic(mean), _statistic(std)
    μ, σ = μ .+ zero.(σ), σ .+ zero.(μ) # same type and shape, e.g., given a common mean and a standard deviation per row
    length(μ) == 1 && ((μ, σ) = (only(μ), only(σ))) # a single statistic (e.g., 1 × 1) applies to every row, as a number does
    all(isfinite, μ) && all(isfinite, σ) || throw(ArgumentError("The statistics of a Standardiser must be finite"))
    all(>(0), σ) || throw(ArgumentError("The standard deviations of a Standardiser must be positive"))
    return Standardiser{typeof(μ)}(μ, σ)
end
_statistic(x::Real) = Float32(x)
_statistic(x::AbstractVector{<:Real}) = reshape(Float32.(cpu_device()(_stripnames(x))), :, 1) # one statistic per row, on the host
_statistic(x::AbstractArray{<:Real}) = Float32.(cpu_device()(_stripnames(x)))
_statistic(x) = throw(ArgumentError("The statistics of a Standardiser must be numbers or arrays of numbers; got $(typeof(x))"))

function Standardiser(x)
    x isa DataAndSummaries && throw(ArgumentError("Construct the standardisers of the data and of the expert summary statistics of a DataAndSummaries object `d` separately, as `(Z = Standardiser(d.Z), S = Standardiser(d.S))`"))
    s = _checkedfit(_stripnames(_extractθ(x)))
    isnothing(s) && throw(ArgumentError("Cannot compute standardisation statistics from an object of type $(typeof(x)), or it contains non-finite values: give the statistics with `Standardiser(mean, std)` instead (for data stored in a type defined by the user, see `NeuralEstimators.fitstandardiser`)"))
    return s
end

# The statistics are never moved by the functors of the estimators (see _ondevice)
Functors.@leaf Standardiser

# Normalises the keyword arguments standardise_data and standardise_parameters
_standardiser(x::Bool) = x ? Standardiser() : nothing
_standardiser(::Nothing) = nothing
_standardiser(s::Standardiser) = s
_standardiser(s::_AutoStandardiser) = s
_standardiser(t::NamedTuple{(:Z, :S)}) = map(_standardiser, t)
_standardiser(x) = throw(ArgumentError("Standardisation is specified by `true`, `false`, or a `Standardiser` (or, for data stored as DataAndSummaries objects, by a named tuple `(Z = …, S = …)` of these); got $(typeof(x))"))

# Standardisers of an estimator; estimators defined by the user need not have them
_datastandardiser(e) = hasfield(typeof(e), :data_standardiser) ? e.data_standardiser : nothing
_parameterstandardiser(e) = hasfield(typeof(e), :parameter_standardiser) ? e.parameter_standardiser : nothing

# ---- Training: standardisation on the host ----

# During training, _inputoutput() standardises the data and parameters once per simulated data set, on the host, and the
# estimator being trained holds its standardisers wrapped in a _Prestandardised, so that its forward pass does not apply
# them again. Hence, training does no standardisation per batch, and the networks output values on the standardised scale,
# so that the loss is evaluated on that scale. The wrapper is removed when train() returns.
struct _Prestandardised{S}
    s::S
end
Functors.@leaf _Prestandardised
_mark(::Nothing) = nothing
_mark(s::_Prestandardised) = s
_mark(s) = _Prestandardised(s)
_unmark(s::_Prestandardised) = s.s
_unmark(s) = s

function _mapstandardisers(f, e)
    hasfield(typeof(e), :data_standardiser) && (e = @set e.data_standardiser = f(e.data_standardiser))
    hasfield(typeof(e), :parameter_standardiser) && (e = @set e.parameter_standardiser = f(e.parameter_standardiser))
    # The inverse transform of a neural Bayes estimator with parameter bounds, which its forward pass applies to the
    # estimates, is mapped too: during training, the parameters are transformed on the host (see _toqscale), so that the
    # forward pass skips it, also when the parameters are not standardised
    e isa AbstractBayesEstimator && _hasbounds(e) && (e = @set e.invtransform = f(e.invtransform))
    return e
end

# Standardisers that apply no map in a forward pass: none, unfitted (incl. the default of a PointEstimator), or applied on
# the host during training. The forward pass of an estimator whose standardisers are all of this type is the original one,
# written verbatim, so that training (where the estimator is marked) costs exactly what it did without standardisation.
# Otherwise, the forward pass maps the inputs, applies the estimator without its standardisers (i.e., the original forward
# pass), and maps the outputs back
const _NoMap = Union{Nothing, Standardiser{Nothing}, _AutoStandardiser, _Prestandardised}
_withoutstandardisers(e) = _mapstandardisers(s -> nothing, e)
_setmodel(trainstate, model) = @set trainstate.model = model
_markstandardisers(trainstate) = _setmodel(trainstate, _mapstandardisers(_mark, trainstate.model))
_unmarkstandardisers(trainstate) = _setmodel(trainstate, _mapstandardisers(_unmark, trainstate.model))

# ---- Application ----

# The statistics are stored on the host. They are converted to the device of the data where they are used, rather than
# moved with the neural networks, so that they are not part of the functor, the Flux state, or the Lux parameters and states.
# During training they are not used on the device (see _Prestandardised); under Reactant they become constants
_ondevice(c::Number, x) = c
_ondevice(c::AbstractArray, x) = @ignore_derivatives _ondevice(c, x, get_device_type(x))
_ondevice(c, x, ::Type{<:Union{Nothing, CPUDevice, ReactantDevice}}) = c
_ondevice(c, x, ::Type) = get_device(x)(c)

# Maps to and from the standardised scale (differentiable, e.g., for the outputs of a neural Bayes estimator). The result
# has the shape of the input: a vector stays a vector, and statistics that would broadcast an input to a larger size (e.g.,
# per-row statistics against a single row) give a DimensionMismatch rather than a silently wrong result
_standardise(::Nothing, x) = x
_standardise(::Standardiser{Nothing}, x) = x
_standardise(::_AutoStandardiser, x) = x
_standardise(::_Prestandardised, x) = x
_standardise(s::Standardiser, x) = reshape((x .- _ondevice(s.mean, x)) ./ _ondevice(s.std, x), size(x))
_unstandardise(::Nothing, y) = y
_unstandardise(::Standardiser{Nothing}, y) = y
_unstandardise(::_AutoStandardiser, y) = y
_unstandardise(::_Prestandardised, y) = y
_unstandardise(s::Standardiser, y) = reshape(y .* _ondevice(s.std, y) .+ _ondevice(s.mean, y), size(y))

# Statistics of a subset of the rows of the parameters, and of the rows repeated (e.g., for estimators that stack the
# estimates for several probability levels)
_rows(s::Standardiser{<:AbstractArray}, idx) = Standardiser{typeof(s.mean)}(s.mean[idx, :], s.std[idx, :])
_rows(s, idx) = s
_repeatrows(s::Standardiser{<:AbstractArray}, n) = Standardiser{typeof(s.mean)}(repeat(s.mean, n), repeat(s.std, n))
_repeatrows(s, n) = s

# Standardises data in single precision (the precision of the neural networks), for each supported container. The data
# are rounded to single precision first, so that the result is the same whether they are standardised during training
# (on the host, in double precision if given so) or at inference (after the conversion to single precision for the device)
_standardisedata(::Nothing, Z) = Z
_standardisedata(::Standardiser{Nothing}, Z) = Z
_standardisedata(::_Prestandardised, Z) = Z
_standardisedata(s::Standardiser, Z) = @ignore_derivatives s(Z) # incl. the methods for data types defined by the user
_standardisedata(s, Z) = @ignore_derivatives _applydata(s, Z)   # e.g., a named tuple of standardisers for DataAndSummaries

"""
	(s::Standardiser)(Z)
Applies the standardiser `s` to data `Z`, returning data of the same type, standardised in single precision. An unfitted
standardiser (specified by `true`; see [`Standardiser`](@ref)) returns `Z` unchanged.

Every estimator standardises its data in this way, so data stored in a type defined by the user (e.g., `MyData`) are
standardised once a method is added for that type, typically by applying `s` to the arrays that it stores, together with a
method of [`NeuralEstimators.fitstandardiser`](@ref) if [`train`](@ref) is to compute the statistics (see
[Standardisation](@ref advanced-standardisation)):
```julia
(s::Standardiser)(d::MyData) = MyData(s(d.Z), d.W) # standardise the values Z, but not the indicators W
```
"""
(s::Standardiser)(Z) = s isa Standardiser{Nothing} ? Z : _applydata(s, Z)
_applydata(s::Standardiser, Z::AbstractArray{<:Number}) = _applystandardiser(_ondevice(s.mean, Z), _ondevice(s.std, Z), Z) # incl. arrays traced by Reactant
function _applydata(s::Standardiser, Z::AbstractVector{<:AbstractArray}) # a vector of data sets
    isempty(Z) && return Z
    μ, σ = _ondevice(s.mean, first(Z)), _ondevice(s.std, first(Z))
    return map(z -> _applystandardiser(μ, σ, z), Z)
end
_applystandardiser(μ, σ, z::AbstractArray{<:Real}) = reshape((Float32.(z) .- μ) ./ σ, size(z))
_applystandardiser(μ, σ, z) = reshape((z .- μ) ./ σ, size(z)) # e.g., arrays traced by Reactant, whose elements are not Real
# A container whose element type is wider than its contents (e.g., a Vector{Any} built with push!, or an array with element
# type Union{Missing, T} but no missing values) is narrowed, as the conversion to single precision for the device does, so
# that it is standardised like concretely typed data; otherwise (e.g., a vector of graphs) it is returned unchanged
function _applydata(s::Standardiser, Z::AbstractArray)
    isconcretetype(eltype(Z)) && return Z
    Zn = map(identity, Z)
    return eltype(Zn) == eltype(Z) ? Z : s(Zn)
end
# NB the contents of containers are standardised with s(...), so that the methods for data types defined by the user apply
_applydata(s::Standardiser, P::PackedReplicates) = PackedReplicates(s(P.data), P.sample_sizes, P.mask) # the padding stays masked
_applydata(s::Standardiser, d::DataAndSummaries) = DataAndSummaries(s(d.Z), d.S) # a single standardiser applies to the data only
_applydata(s::NamedTuple, d::DataAndSummaries) = DataAndSummaries(_standardisedata(s.Z, d.Z), _standardisedata(s.S, d.S))
_applydata(s::NamedTuple, Z) = _standardisedata(s.Z, Z)
_applydata(s, Z) = Z # other containers (e.g., graphs, or precomputed Summaries) are not standardised

# ---- Fitting ----

"""
	NeuralEstimators.fitstandardiser(x)
Returns a [`Standardiser`](@ref) with the statistics of data or parameters `x`, computed by the default rule (see
[`Standardiser`](@ref)), or `nothing` if `x` is not supported (e.g., graphs) or contains non-finite values. [`train`](@ref)
calls it to fit the standardisers of an estimator before training begins, and leaves data for which it returns `nothing`
unstandardised; `Standardiser(x)` calls it too, and throws an error instead of returning `nothing`.

It is not exported. To standardise data stored in a type defined by the user (e.g., `MyData`), add a method for that type,
which returns a `Standardiser` or `nothing`, typically by calling `fitstandardiser` on the arrays that it stores, together
with a method of the call operator `(s::Standardiser)(Z)` (see [Standardisation](@ref advanced-standardisation)):
```julia
NeuralEstimators.fitstandardiser(d::MyData) = NeuralEstimators.fitstandardiser(d.Z) # statistics of the values Z only
```
"""
fitstandardiser(x) = nothing
fitstandardiser(x::AbstractArray{<:Real}) = _fromsums(_sums(x)...) # NB the sums are accumulated in double precision without copying x
function fitstandardiser(Z::AbstractVector{<:AbstractArray})
    # Pooled over the replicates of all data sets, by the rule for the data sets; per-row statistics need a common number of rows
    isempty(Z) && return nothing
    all(z -> eltype(z) <: Real, Z) || return nothing
    allequal(map(ndims, Z)) && (ndims(first(Z)) != 2 || allequal(map(z -> size(z, 1), Z))) || return nothing
    sums = map(_sums, Z)
    return _fromsums(sum(s -> s[1], sums), sum(s -> s[2], sums), sum(s -> s[3], sums))
end
# A container whose element type is wider than its contents is narrowed first (see _applydata)
function fitstandardiser(Z::AbstractArray)
    isconcretetype(eltype(Z)) && return nothing
    Zn = map(identity, Z)
    return eltype(Zn) == eltype(Z) ? nothing : fitstandardiser(Zn)
end
function fitstandardiser(P::PackedReplicates)
    isnothing(P.mask) && return fitstandardiser(P.data)
    # Exclude the padding, by weighting the replicates (along the last dimension) by the mask
    x = P.data
    w = reshape(vec(P.mask), ntuple(_ -> 1, ndims(x) - 1)..., :)
    s1, s2, _ = _sums(x .* w)
    n = round(Int, sum(P.mask)) * (ndims(x) == 2 ? 1 : prod(size(x)[1:(end - 1)]))
    return _fromsums(s1, s2, n)
end

# Sums of the values and of their squares, per row (as n × 1 matrices) of a matrix and overall (as numbers) otherwise, and
# the number of values in each sum
_sums(x::AbstractMatrix{<:Real}) = (sum(Float64, x; dims = 2), sum(abs2 ∘ Float64, x; dims = 2), size(x, 2))
_sums(x::AbstractArray{<:Real}) = (sum(Float64, x), sum(abs2 ∘ Float64, x), length(x))

function _fromsums(s1, s2, n::Integer)
    n > 0 || return nothing
    μ = cpu_device()(s1 ./ n) # the statistics are stored on the host, also when fitted to data on a GPU
    σ = sqrt.(max.(cpu_device()(s2 ./ n) .- μ .^ 2, 0))
    μ, σ = Float32.(μ), Float32.(σ)
    all(isfinite, μ) && all(isfinite, σ) || return nothing # incl. statistics that overflow in single precision
    # A standard deviation that is zero (incl. by underflow in single precision), or below the resolution of single
    # precision relative to the mean, is set to one
    σ = ifelse.((σ .>= floatmin(Float32)) .& (σ .> 4eps(Float32) .* abs.(μ)), σ, one.(σ))
    return Standardiser{typeof(μ)}(μ, σ)
end

# fitstandardiser(x), which must be a Standardiser with statistics or nothing: a method for a data type defined by the user
# that returns anything else is reported here, rather than failing later (e.g., when the estimator is saved)
function _checkedfit(x)
    s = fitstandardiser(x)
    s isa Union{Nothing, Standardiser} && !(s isa Standardiser{Nothing}) && return s
    throw(ArgumentError("`NeuralEstimators.fitstandardiser(::$(typeof(x)))` must return a Standardiser with statistics, or nothing if the statistics cannot be computed; got an object of type $(typeof(s))"))
end

# Replaces the unfitted standardisers in `s` by standardisers fitted to x, keeping those given by the user or fitted before
_fit(s, x) = s
_fit(::Standardiser{Nothing}, x) = _checkedfit(x)
_fit(::_AutoStandardiser, x) = _checkedfit(x)
_fit(::Standardiser{Nothing}, d::DataAndSummaries) = (Z = _checkedfit(d.Z), S = _checkedfit(d.S))
_fit(t::NamedTuple{(:Z, :S)}, d::DataAndSummaries) = (Z = _fit(t.Z, d.Z), S = _fit(t.S, d.S))
_fit(t::NamedTuple{(:Z, :S)}, x) = (Z = _fit(t.Z, x), S = t.S)

_isunfitted(::Standardiser{Nothing}) = true
_isunfitted(::_AutoStandardiser) = true
_isunfitted(t::NamedTuple) = any(_isunfitted, values(t))
_isunfitted(::Any) = false

# Whether fitting left an unfitted standardiser without statistics
_notfitted(::Standardiser{Nothing}, after) = isnothing(after)
_notfitted(::_AutoStandardiser, after) = isnothing(after)
_notfitted(before::NamedTuple{(:Z, :S)}, after::NamedTuple{(:Z, :S)}) = _notfitted(before.Z, after.Z) || _notfitted(before.S, after.S)
_notfitted(before, after) = false

"""
	_fitunfitted(trainstate, θ, Z, loss, verbose)
Fits the unfitted standardisers of the estimator being trained to parameters `θ` and data `Z`. This is done once per call
to [`train`](@ref), before the summary statistics of a frozen summary network are computed, and before the data loaders,
the initial risk, and the first checkpoint. Standardisers that are given by the user or were fitted before (e.g., those of
an estimator that was trained or loaded) are never refitted, since the trained networks depend on them. The `loss` given
to [`train`](@ref) decides whether the parameters of a [`PointEstimator`](@ref) are standardised by default (see
`_autostandardiser`). At every call, a warning is given if the loss is one for which the parameter bounds of a
[`PointEstimator`](@ref) change the Bayes estimator.
"""
function _fitunfitted(trainstate, θ, Z, loss, verbose::Bool)
    e = trainstate.model
    # The bounds of a PointEstimator are imposed by evaluating the loss on an unbounded scale, which leaves the Bayes
    # estimator unchanged if it is a vector of posterior quantiles: so if the loss is, in addition to the properties checked
    # by _invariantloss, positively homogeneous of degree one (that is, a sum over the parameters of check losses of the
    # errors), since quantiles commute with increasing transformations
    if e isa PointEstimator && _transform(e) !== identity
        θ̃ = _transform(e)(_stripnames(_extractθ(θ)))
        s = fitstandardiser(θ̃)
        if !isnothing(s) && !isapprox(something(_homogeneity(_loss(e, loss), s, θ̃), NaN), 2; rtol = 1e-3)
            @warn "The bounds `lower` and `upper` are imposed by evaluating the loss function on an unbounded scale, which leaves the Bayes estimator unchanged only for `mae` and quantile losses, whose Bayes estimators are posterior medians and quantiles. With this loss function, it changes the estimator. Impose the constraints within the neural network itself instead (e.g., with an output layer such as `Compress`, or an activation function such as `softplus`), and construct the PointEstimator without `lower` and `upper` (see PointEstimator)."
        end
    end
    sZ, sθ = _datastandardiser(e), _parameterstandardiser(e)
    _isunfitted(sZ) || _isunfitted(sθ) || return trainstate
    if _isunfitted(sZ)
        s = _fit(sZ, Z)
        verbose && _notfitted(sZ, s) && @info "The data are not standardised, since standardisation is not supported for data of type $(typeof(Z)) or the data contain non-finite values (data stored in a type defined by the user can be supported; see `NeuralEstimators.fitstandardiser`). Set `standardise_data = false` when constructing the estimator to silence this message."
        e = @set e.data_standardiser = s
    end
    if _isunfitted(sθ)
        θ = _transform(e)(_stripnames(_extractθ(θ)))
        s = _fit(sθ, θ)
        verbose && _notfitted(sθ, s) && @info "The parameters are not standardised, since they contain non-finite values."
        sθ isa _AutoStandardiser && !isnothing(s) && (s = _autostandardiser(e, s, θ, _loss(e, loss), verbose))
        e = @set e.parameter_standardiser = s
    end
    return _setmodel(trainstate, e)
end

# The default parameter standardiser of a PointEstimator: the standardiser `s` fitted to the parameters θ if standardising
# them leaves the estimator unchanged, and `nothing` otherwise. This requires that the neural network does not constrain
# its output, which would otherwise be constrained on the standardised scale, and that the Bayes estimator under the loss
# is unchanged
function _autostandardiser(e, s, θ, loss, verbose::Bool)
    silence = "Set `standardise_parameters` to `true` or `false` when constructing the estimator to silence this message."
    if _linearoutputs(e) != size(θ, 1)
        verbose && @info "The parameters are not standardised, since the neural network may constrain the estimates (it does not end in a Dense layer without activation function, with one output per parameter), and it would then constrain them on the standardised scale (see PointEstimator). $silence"
        return nothing
    elseif !_invariantloss(loss, s, θ)
        verbose && @info "The parameters are not standardised, since this would change the Bayes estimator under the loss function (see PointEstimator). $silence"
        return nothing
    end
    verbose && @info "The parameters are standardised, which leaves the Bayes estimator under the loss function unchanged, so the training and validation risks are evaluated on the standardised scale (see PointEstimator). $silence"
    return s
end

# The number of outputs of a neural network whose final layer is linear (a Dense layer without activation function), or
# `nothing` if its final layer may constrain its output (e.g., an activation function such as exp or softplus, or an
# output layer such as Compress). Layers that are not known to be linear are assumed to constrain their output. Methods
# for the neural networks of each backend are defined in the extensions
_linearoutputs(network) = nothing

"""
	_invariantloss(loss, s::Standardiser, θ)
Whether standardising the parameters `θ` (a d × K matrix, or an array whose last dimension indexes the K parameter
vectors) with `s` leaves the Bayes estimator under `loss` unchanged.
This is so if the loss is a sum (or mean) over the parameter vectors, and over the parameters, of terms that each depend
on an estimate and the true value through their difference (the error) only, as a positively homogeneous function of it
(e.g., `mae`, `mse`, and [`quantileloss`](@ref)), since standardisation then rescales each term by a constant. These properties are checked
numerically, with errors of several sizes relative to the spread of the parameters, both on the standardised scale (on
which the loss would be evaluated during training if the parameters were standardised) and on the original scale up to
its location (on which it would be evaluated otherwise), so that a loss with a constant that acts on the scale of the
parameters (e.g., the constant κ of [`tanhloss`](@ref)) is detected whichever scale the constant was chosen for:
- separability: swapping the values of a subset of the parameters between two pairs of arguments leaves the sum of their losses unchanged, for each subset of parameters whose index (minus one) has a given bit set, so that every pair of parameters is split by one of the subsets. The errors of the two pairs differ in size, since sums over many parameters would otherwise be nearly equal for both pairs, which would mask a loss that depends on such a sum (e.g., the joint version of [`kpowerloss`](@ref));
- additivity over the parameter vectors: the loss of the two pairs of arguments together is the sum, or the mean, of their losses (e.g., not the root of a mean of squares);
- dependence on the error only: shifting the estimates and the true values by the same amount leaves the loss unchanged;
- positive homogeneity: doubling the errors multiplies the loss, relative to its value at zero error, by a factor greater than one that is the same for every size of the errors.
The check uses a local random number generator, so that the global random number stream is unchanged. A loss that throws
an error or returns a non-finite value fails the check.
"""
_invariantloss(loss, s::Standardiser, θ::AbstractArray) = !isnothing(_homogeneity(loss, s, θ))

# The factor by which doubling the errors multiplies the loss, relative to its value at zero error, if the loss has the
# properties checked by _invariantloss, and nothing otherwise. The factor is 2^k for a loss that is positively homogeneous
# of degree k: e.g., 2 for `mae` and `quantileloss`, whose Bayes estimators are posterior quantiles, and 4 for `mse`
function _homogeneity(loss, s::Standardiser, θ::AbstractArray)
    rng = Xoshiro(1)
    θ = cpu_device()(θ)
    N, d, K = ndims(θ), size(θ, 1), size(θ, ndims(θ))
    n = min(K, 16)
    vectors(r) = Float32.(_standardise(s, collect(selectdim(θ, N, r)))) # standardised parameter vectors
    Θ₁, Θ₂ = vectors(1:n), vectors((K - n + 1):K)
    rows(v) = reshape(v, d, ntuple(_ -> 1, N - 1)...) # one value per row, the same along the other dimensions
    σ = rows(s.std isa Number ? fill(s.std, d) : vec(s.std))
    subsets = [rows(isodd.((0:(d - 1)) .>> b)) for b in 0:(ceil(Int, log2(d)) - 1)]
    L(θ̂, θ) = Float64(loss(θ̂, θ))
    same(a, b) = isfinite(a) && isfinite(b) && isapprox(a, b; rtol = 1e-3)
    factor = nothing
    try
        for δ in (1.0f0, 0.1f0, 0.01f0), λ in (rows(ones(Float32, d)), σ) # the standardised scale, and the original scale up to its location
            E₁, E₂ = δ .* randn(rng, Float32, size(Θ₁)), 4δ .* randn(rng, Float32, size(Θ₂))
            x = (λ .* (Θ₁ .+ E₁), λ .* Θ₁)
            y = (λ .* (Θ₂ .+ E₂), λ .* Θ₂)
            Lx, Ly = L(x...), L(y...)
            # Additivity over the parameter vectors (the two pairs have the same number of them)
            Lxy = L(cat(x[1], y[1]; dims = N), cat(x[2], y[2]; dims = N))
            same(Lxy, Lx + Ly) || same(Lxy, (Lx + Ly) / 2) || return nothing
            # Separability
            for S in subsets
                x′ = (ifelse.(S, y[1], x[1]), ifelse.(S, y[2], x[2]))
                y′ = (ifelse.(S, x[1], y[1]), ifelse.(S, x[2], y[2]))
                same(Lx + Ly, L(x′...) + L(y′...)) || return nothing
            end
            # Dependence on the error only
            c = λ .* rows(randn(rng, Float32, d))
            same(Lx, L(x[1] .+ c, x[2] .+ c)) || return nothing
            # Positive homogeneity
            L₀ = L(x[2], x[2])
            r = (L(λ .* (Θ₁ .+ 2 .* E₁), x[2]) - L₀) / (Lx - L₀)
            isfinite(r) && r >= 1.01 || return nothing
            isnothing(factor) ? (factor = r) : (same(r, factor) || return nothing)
        end
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    return factor
end

# ---- Parameter constraints ----

# Maps the parameters to and from the unbounded scale given lower and upper bounds (see scaledlogit and scaledlogistic).
# They are callable structs rather than closures, so that the bounds can be saved with the estimator (see _transformations)
struct _ScaledLogit{A, B}
    lower::A
    upper::B
end
(t::_ScaledLogit)(θ) = _applybounds(scaledlogit, θ, t.lower, t.upper)
struct _ScaledLogistic{A, B}
    lower::A
    upper::B
end
(t::_ScaledLogistic)(θ) = _applybounds(scaledlogistic, θ, t.lower, t.upper)

# Applies f(θ, lower, upper) elementwise, with the bounds of each row of θ: on the host, in one broadcast; on another device
# (e.g., the estimates of a neural Bayes estimator on a GPU, or traced by Reactant), with the bounds of each row fixed in a
# closure, so that the methods for Real bounds, which support infinite bounds, apply (see scaledlogistic)
_applybounds(f, θ, a, b) = _applybounds(f, θ, a, b, @ignore_derivatives(get_device_type(θ)))
_applybounds(f, θ, a, b, ::Type{<:Union{Nothing, CPUDevice}}) = f.(θ, a, b)
function _applybounds(f, θ, a, b, ::Type{D}) where {D}
    T = D <: Union{MetalDevice, oneAPIDevice} ? Float32 : Float64 # GPUs that may not support double precision
    rows = Any[]
    for i in axes(θ, 1) # NB a loop rather than map, which Reactant would trace (with a traced index) since the function captures θ
        aᵢ, bᵢ = T(a isa Real ? a : a[i]), T(b isa Real ? b : b[i])
        push!(rows, (x -> f(x, aᵢ, bᵢ)).(selectdim(θ, 1, i:i)))
    end
    return vcat(rows...)
end

# The bounds of a subset of the parameters, and repeated (e.g., for the estimates of several probability levels), as for a
# Standardiser (see _rows and _repeatrows); a bound given as a number applies to every parameter
_rows(t::T, idx) where {T <: Union{_ScaledLogit, _ScaledLogistic}} = T(map(a -> a isa Real ? a : a[idx], (t.lower, t.upper))...)
_repeatrows(t::T, n) where {T <: Union{_ScaledLogit, _ScaledLogistic}} = T(map(a -> a isa Real ? a : repeat(a, n), (t.lower, t.upper))...)

# The map of the parameters to an unbounded scale and its inverse, given the keyword arguments `lower`, `upper`, `transform`,
# and `invtransform` of an estimator (see PosteriorEstimator; neural Bayes estimators take bounds only)
function _transforms(lower, upper, transform = identity, invtransform = identity)
    if !isnothing(lower) || !isnothing(upper)
        transform === invtransform === identity || throw(ArgumentError("Specify either `lower` and `upper`, or `transform` and `invtransform`, but not both"))
        a, b = Float64.(something(lower, -Inf)), Float64.(something(upper, Inf))
        @assert all(a .< b) "lower bounds must be strictly below upper bounds"
        a, b = map(x -> x isa AbstractVector && length(x) == 1 ? only(x) : x, (a, b)) # a single bound applies to every parameter, as a number does
        transform = _ScaledLogit(a, b)         # NB computed in Float64, the type of the bounds
        invtransform = _ScaledLogistic(a, b)
    end
    (transform === identity) == (invtransform === identity) || throw(ArgumentError("`transform` and `invtransform` must be specified together"))
    return transform, invtransform
end

# Parameter constraints of an estimator (only the estimators of the package that have the fields `transform` and
# `invtransform` have them, so that the fields of estimators defined by the user are never interpreted as constraints)
_hasbounds(e) = false
_transform(e) = _hasbounds(e) ? e.transform : identity
_invtransform(e) = _hasbounds(e) ? e.invtransform : identity

# Maps the parameters to and from the scale of the approximate distribution of a PosteriorEstimator or PriorEstimator, or
# the scale on which the networks of a neural Bayes estimator are trained: to the unbounded scale, and then standardised
_toqscale(e, θ) = _standardise(_unmark(_parameterstandardiser(e)), _transform(e)(θ))
_fromqscale(e, θ̃) = _invtransform(e)(_unstandardise(_unmark(_parameterstandardiser(e)), θ̃))

# Sets the parameter constraints of an estimator to those saved with the trained estimator (see _settransformations)
function _setbounds(e, ::Nothing)
    _savedbounds(e.transform) === nothing || @warn "The given estimator has parameter constraints (`lower`, `upper`, `transform`, or `invtransform`), but the trained estimator was saved without them, so they are removed"
    e = @set e.transform = identity
    return @set e.invtransform = identity
end
function _setbounds(e, b::NamedTuple{(:lower, :upper)})
    given = _savedbounds(e.transform)
    given === nothing || given == b || @warn "The parameter constraints of the given estimator differ from those saved with the trained estimator, which are used instead"
    e = @set e.transform = _ScaledLogit(b.lower, b.upper)
    return @set e.invtransform = _ScaledLogistic(b.lower, b.upper)
end
function _setbounds(e, ::Symbol)
    _savedbounds(e.transform) === :custom || throw(ArgumentError("The estimator was trained with a custom `transform` and `invtransform`, which cannot be saved: pass the same functions when constructing the estimator given to `loadestimator`"))
    return e
end

# ---- Persistence ----

# The standardisers and bounds of an estimator are saved as plain data (numbers, arrays, named tuples, and symbols, rather
# than structs or closures, which do not reliably survive serialisation) under the key `transformations` of the estimator
# file, so that loadestimator() restores them and the user need only give the architecture
const _TRANSFORMATIONS_VERSION = 1

function _transformations(e)
    return (version = _TRANSFORMATIONS_VERSION,
        data_standardiser = _saved(_unmark(_datastandardiser(e))),
        parameter_standardiser = _saved(_unmark(_parameterstandardiser(e))),
        bounds = _savedbounds(_transform(e)))
end
_saved(::Nothing) = nothing
_saved(::Standardiser{Nothing}) = :unfitted
_saved(::_AutoStandardiser) = :auto
_saved(s::Standardiser) = (mean = s.mean, std = s.std)
_saved(t::NamedTuple{(:Z, :S)}) = (Z = _saved(t.Z), S = _saved(t.S))
_savedbounds(::typeof(identity)) = nothing
_savedbounds(t::_ScaledLogit) = (lower = t.lower, upper = t.upper)
_savedbounds(::Any) = :custom

_fromsaved(::Nothing) = nothing
_fromsaved(s::Symbol) = s === :auto ? _AutoStandardiser() : Standardiser()
_fromsaved(t::NamedTuple{(:mean, :std)}) = Standardiser{typeof(t.mean)}(t.mean, t.std)
_fromsaved(t::NamedTuple{(:Z, :S)}) = (Z = _fromsaved(t.Z), S = _fromsaved(t.S))

# Files saved without the transformations: by versions of NeuralEstimators before 0.5.0, which had no standardisation or
# bounds, or manually (e.g., with Flux.state)
const _NO_TRANSFORMATIONS = (version = _TRANSFORMATIONS_VERSION, data_standardiser = nothing, parameter_standardiser = nothing, bounds = nothing)

"""
	_restoretransformations(estimator, saved::AbstractDict)
Sets the standardisers and bounds of `estimator`, which provides the architecture, to those saved in the estimator file
(read into `saved`), which is authoritative.
"""
_restoretransformations(e, saved::AbstractDict) = _settransformations(e, get(saved, :transformations, _NO_TRANSFORMATIONS))

function _settransformations(e, entry::NamedTuple)
    entry.version <= _TRANSFORMATIONS_VERSION || throw(ArgumentError("The estimator was saved by a newer version of NeuralEstimators.jl: please update NeuralEstimators.jl to load it"))
    if hasfield(typeof(e), :data_standardiser)
        s = _fromsaved(entry.data_standardiser)
        _warnifdiffers(e.data_standardiser, s, "data standardiser")
        e = @set e.data_standardiser = s
    end
    if hasfield(typeof(e), :parameter_standardiser)
        s = _fromsaved(entry.parameter_standardiser)
        _warnifdiffers(e.parameter_standardiser, s, "parameter standardiser")
        e = @set e.parameter_standardiser = s
    end
    _hasbounds(e) && (e = _setbounds(e, entry.bounds))
    return e
end
# The entry of an Ensemble (one per component) given to an estimator that is not an Ensemble
_settransformations(e, entries::AbstractVector) = throw(ArgumentError("The saved estimator is an Ensemble of $(length(entries)) estimators, but the given estimator is a $(nameof(typeof(e))): please ensure that the same architecture is used when loading"))

# A warning when the given estimator specifies fitted statistics that differ from the saved ones (the given estimator
# provides the architecture only, and unfitted standardisers are simply replaced)
_isfittedstandardiser(s::Standardiser) = !(s isa Standardiser{Nothing})
_isfittedstandardiser(t::NamedTuple) = any(_isfittedstandardiser, values(t))
_isfittedstandardiser(::Any) = false
function _warnifdiffers(given, saved, name)
    if _isfittedstandardiser(_unmark(given)) && _saved(_unmark(given)) != _saved(saved)
        @warn "The $name of the given estimator differs from the one saved with the trained estimator, which is used instead"
    end
    return nothing
end
