# NeuralEstimators

Neural estimators are a recent approach to likelihood-free inference, and have emerged as a promising alternative to the well-establish approximate Bayesian computation (ABC). `NeuralEstimators` aims to faciliate the development of neural estimators in a user-friendly manner. In particular, it aims to alleviate the user from the substantial amount of boilerplate code needed to implement neural estimators from scratch. To this end, the main task asked of the user is to implicitly define the statistical model by providing a function for simulating data; everything else is handled by `NeuralEstimators`.

Julia has many [attractive features](https://julialang.org/); in particular, it has been designed to alleviate the so-called two-language problem, whereby it aims to be both easily developed and fast. This means that users can easily write code for data simulation without needing to vectorise (i.e., `for` loops are fine!). Further, many Julia packages are written entirely in Julia and, hence, their source code is easily understood and extended; this includes `NeuralEstimators` and the deep learning framework on which it is built upon, [Flux](https://fluxml.ai/Flux.jl/stable/).


## Installation

Download [Julia](https://julialang.org/) if you haven't already. To install `NeuralEstimators` from Julia's package manager, use the following commands inside the Julia REPL:

```
using Pkg
Pkg.add("NeuralEstimators")
```

## Getting started

See our [Workflow overview](@ref).


## Supporting and citing

This software was developed as part of academic research. If you would like to help support it, please star the repository. If you use `NeuralEstimators` in your research or other activities, please use the following citation.

```
@article{
  <bibtex citation>
}
```
