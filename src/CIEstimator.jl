#TODO documentation
#TODO could store information such as the significance level α
"""
    CIEstimator(l, u)


# Examples
```

```
"""
struct CIEstimator{F, G}
	l::F
	u::G
	# p
	# α
	# parameter_names
end
@functor CIEstimator
(e::CIEstimator)(Z) = vcat(e.l(Z), e.l(Z) .+ exp.(e.u(Z))) # exponentiate to avoid crossing


function confidenceinterval(ciestimator::CIEstimator, Z; parameter_names = nothing, use_gpu::Bool = true)

	# ci = ciestimator(Z)
	ci = _runondevice(ciestimator, Z, use_gpu)
	ci = cpu(ci)

	# TODO use this code if I decide to make parameter_names apart of CIEstimator
	# if isnothing(parameter_names)
	# 	parameter_names = ciestimator.parameter_names
	# end

	@assert size(ci, 1) % 2 == 0
	p = size(ci, 1) ÷ 2
	if isnothing(parameter_names)
		parameter_names = ["θ$i" for i ∈ 1:p]
	else
		@assert length(parameter_names) == p
	end

	labelconfidenceinterval(ci, parameter_names)
end
