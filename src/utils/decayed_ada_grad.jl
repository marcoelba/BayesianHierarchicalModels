# DecayedADAGrad    
const ϵ = 1e-8


"""
    MyDecayedADAGrad(η=0.1, pre=1.0, post=0.9)

Implements a decayed version of AdaGrad. It has parameter specific learning rates based on how frequently it is updated.

## Parameters
  - η: learning rate
  - pre: weight of new gradient norm
  - post: weight of histroy of gradient norms
```
## References
[ADAGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) optimiser.
Parameters don't need tuning.
"""
mutable struct MyDecayedADAGrad
    eta::Float64
    pre::Float64
    post::Float64

    acc::IdDict
end

MyDecayedADAGrad(η = 0.1, pre = 1.0, post = 0.9) = MyDecayedADAGrad(η, pre, post, IdDict())

function apply!(o::MyDecayedADAGrad, x, Δ)
    
    η = o.eta
    acc = get!(() -> fill!(similar(x), ϵ), o.acc, x)::typeof(x)

    @. acc = o.post * acc + o.pre * Δ^2
    @. Δ *= η / (√acc + ϵ)
end
