import Distributions
import Turing


EPS = 1e-16


struct LogitRelaxedBernoulli{P<:Real, L<:Real, T<:Real} <: Distributions.ContinuousUnivariateDistribution
    prob::P
    logit::L
    temp::T
end

LogitRelaxedBernoulli(prob, temp) = LogitRelaxedBernoulli(prob, Distributions.logit(prob), temp)
LogitRelaxedBernoulli(prob, logit, temp) = LogitRelaxedBernoulli(prob, logit, temp)

function relaxed_bernoulli_sample(prob::Real, temp::Real)
    unif_sample = Distributions.rand(Distributions.Uniform())

    logit = (log(unif_sample) - log1p(-unif_sample) + log(prob) - log1p(-prob)
        ) ./ temp
    
    return logit
end

Distributions.rand(d::LogitRelaxedBernoulli) = relaxed_bernoulli_sample(d.prob, d.temp)

function relaxed_bernoulli_logpdf(x, logit, temp)
    diff = logit - x * temp
    log(temp) + diff - 2. * log1p(exp(diff))
end

Distributions.logpdf(d::LogitRelaxedBernoulli, x::AbstractArray{<:Real}) = relaxed_bernoulli_logpdf(x, d.logit, d.temp)
Distributions.logpdf(d::LogitRelaxedBernoulli, x::Real) = relaxed_bernoulli_logpdf(x, d.logit, d.temp)

Distributions.minimum(d::LogitRelaxedBernoulli) = -Inf
Distributions.maximum(d::LogitRelaxedBernoulli) = +Inf


# Test
# probs = [0.1, 0.6]
# logits = log.(probs ./ (1 .- probs))
# temp = 0.1
# x = LogitRelaxedBernoulli.(probs, temp)

# sam = Distributions.rand.(x)
# Distributions.logpdf.(x, sam)

# probs = 0.3
# logits = log.(probs ./ (1 .- probs))
# temp = 0.1
# x = LogitRelaxedBernoulli(probs, temp)

# Distributions.logpdf(x, -10.)

# Turing.mean([Distributions.rand(x) for j in range(1, 10000)])

# sam = Distributions.rand(x)
# Distributions.logpdf(x, sam)

# probs = Float32.([0.1, 0.6])
# logits = log.(probs ./ (1 .- probs))
# temp = 0.1f0
# y = LogitRelaxedBernoulli(probs, temp)
