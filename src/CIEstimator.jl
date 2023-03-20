#TODO documentation
"""
    CIEstimator(l, u)


# Examples
```

```
"""
struct CIEstimator{F, G}
	l::F
	u::G
end
@functor CIEstimator
(e::CIEstimator)(Z) = vcat(e.l(Z), e.l(Z) .+ exp.(e.u(Z))) # exponentiate to avoid crossing
