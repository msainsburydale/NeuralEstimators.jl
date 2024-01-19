# Output activation functions

```@index
Order = [:type, :function]
Pages   = ["activationfunctions.md"]
```

In addition to the standard activation functions provided by [Flux](https://fluxml.ai/Flux.jl/stable/models/activation/), the following layers can be used at the end of an architecture, to act as output activation functions that ensure valid estimates for certain models.

```@docs
Compress

CorrelationMatrix

CovarianceMatrix
```
