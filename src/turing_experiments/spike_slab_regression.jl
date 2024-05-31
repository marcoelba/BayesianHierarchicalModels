# Spike and Slab Regression
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using FillArrays

include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("relaxed_bernoulli.jl"))


Threads.nthreads()

n = 100
p = 10

# First coefficient is the intercept
true_beta = [1.5, 1., -1., -1.5, 1., 0., 0., 0., 0., 0.]
sigma_y = 1.

X = zeros(Float64, n, p)

Random.seed!(32143)

X_dist = Distributions.Normal(0., sigma_y)
X = Random.rand(X_dist, (n, p))
# Get y = X * beta + err ~ N(0, 1)
y = 1. .+ X * true_beta + sigma_y * Random.rand(Distributions.Normal(), n)


@model function lin_model(y, X)
    # Variance
    sigma2_y ~ Turing.truncated(Normal(0, 2))
    # beta (reg coefficients)
    beta ~ Turing.MultivariateNormal(zeros(p), 1.)
    beta0 ~ Turing.Normal(0., 1.)

    mu = beta0 .+ X * beta

    return y ~ MultivariateNormal(mu, sigma2_y)
end

sampler = NUTS()
mc_samples = 300
nchains = 4
burn = 50

model = lin_model(y, X)
chains = sample(model, sampler, MCMCThreads(), mc_samples, nchains; discard_initial = burn)

plot(chains[["beta[1]", "beta[2]", "beta[3]"]], legend=true)
plot(chains[["beta[$(p)]", "beta[$(p-1)]", "beta[$(p-2)]"]], legend=true)


# ---------------------------------
# Using a Spike and Slab prior
# ---------------------------------
@model function ss_model(y, X)
    # Variance
    # sigma2_y ~ Turing.truncated(Normal(0, 2))

    # beta (reg coefficients)
    # gamma_logit ~ LogitRelaxedBernoulli.(ones(p) * 0.5, 0.01)
    # gamma = 1. ./ (1. .+ exp.(-gamma_logit))
    # beta ~ GaussianSpikeSlab.(zeros(p), ones(p), gamma)

    relax_bern = LogitRelaxedBernoulli(0.01, 0.01)
    gamma_logit ~ Turing.filldist(relax_bern, p)
    gamma = 1. ./ (1. .+ exp.(-gamma_logit))

    beta ~ Turing.arraydist([GaussianSpikeSlab(0., 5., gg) for gg in gamma])

    beta0 ~ Turing.Normal(0., 5.)

    mu = beta0 .+ X * beta

    return y ~ MultivariateNormal(mu, 1.)
end

sampler = NUTS()

mc_samples = 300
nchains = 2
burn = 100

model = ss_model(y, X)
chains = sample(model, sampler, MCMCThreads(), mc_samples, nchains; discard_initial = burn)

plot(chains[["beta[1]", "beta[2]", "beta[9]", "beta[10]"]], legend=true)
plot(chains[["gamma_logit[1]", "gamma_logit[10]"]], legend=true)


# Variational Inference
@doc(Turing.Variational.vi)

# ADVI
advi = ADVI(10, 1000)

model = ss_model(y, X)

q = vi(model, advi)


Turing.AdvancedVI.elbo(advi, q, model(y, X), 10000) 

histogram(rand(q, 1_000)[1, :])
histogram(rand(q, 1_000)[10, :])

samples = rand(q, 10000)
size(samples)

histogram(rand(q, 1_000)[1, :])
histogram(samples[11, :])
histogram(samples[20, :])

