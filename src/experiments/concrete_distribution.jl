# Model with concrete distribution
using Flux
using Zygote
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra


# Bernoulli(alpha) ~= CD(ni, delta, alpha)
# ni ~ Unif(0, 1)
function logit(x)
    log.(x ./ (1 .- x))
end

function concrete_distribution(ni, delta, alpha)
    Flux.sigmoid_fast.((logit(alpha) .- logit(ni)) ./ delta)
end

function logit_concrete_distribution(ni, delta, alpha)
    (logit(alpha) .- logit(ni)) ./ delta
end

alpha = [0.1, 0.1, 0.1]
logit(alpha)
ni = rand(3)
delta = 0.01
concrete_distribution(ni, delta, alpha)

pdf.(Bernoulli.(alpha), round.(ni))
pdf.(Bernoulli.(alpha), ni)
