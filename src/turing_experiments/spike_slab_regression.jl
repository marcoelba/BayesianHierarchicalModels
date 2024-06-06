# Spike and Slab Regression
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections

using AdvancedVI
using StatsFuns
using Bijectors
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack

include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("relaxed_bernoulli.jl"))


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


# ---------------------------------
# Using a Spike and Slab prior
# ---------------------------------
@model function ss_model(y, X)
    # Variance
    # sigma2_y ~ Turing.truncated(Normal(0, 2))

    # beta (reg coefficients)
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


# ADVI
advi = ADVI(10, 1000)
model = ss_model(y, X)

q = vi(model, advi)


histogram(rand(q, 1_000)[1, :])
histogram(rand(q, 1_000)[10, :])

samples = rand(q, 10000)
size(samples)

histogram(samples[11, :])
histogram(samples[20, :])


# -----------------------------------------------
# Same inference via AdvancedVI library
# -----------------------------------------------

# Prior distributions
# Intercept
prior_beta0 = Normal(0., 5.)
log_prior_beta0(beta0) = Distributions.logpdf(prior_beta0, beta0)

# beta Spike and Slab
prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.01, 0.01), p)
log_prior_gamma_logit(gamma_logit) = Distributions.logpdf(prior_gamma_logit, gamma_logit)

function log_prior_beta(gamma, beta)
    Distributions.logpdf(
        Turing.arraydist([
            GaussianSpikeSlab(0., 2., gg) for gg in gamma
        ]),
        beta
    )
end

# Likelihood
function likelihood(X, beta, beta0)
    Distributions.MultivariateNormal(
        beta0 .+ X * beta, ones(n) * 1.
    )
end
log_likelihood(y, X, beta, beta0) = sum(Distributions.logpdf(likelihood(X, beta, beta0), y))
likelihood(X, true_beta, 0.)
log_likelihood(y, X, true_beta, 0.)

# Joint
function log_joint(theta_hat)
    beta0 = theta_hat[1]
    gamma_logit = theta_hat[2:(p+1)]
    beta = theta_hat[(p+2):(p+p+1)]
    
    log_prior = log_prior_beta0(beta0) +
        log_prior_gamma_logit(gamma_logit) +
        log_prior_beta(StatsFuns.logistic.(gamma_logit), beta)

    loglik = log_likelihood(y, X, beta, beta0)

    loglik + log_prior
end
theta_hat = ones(p+p+1) * 2
log_joint(ones(p+p+1))

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
# bijector transfrom FROM the latent space TO the REAL line

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_params = p+p+1
num_weights = num_params * 2
half_num_params = Int(num_weights / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:num_weights])
    )
end

getq(ones(num_weights))

advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_weights)*0.1)
# Check the ELBO
AdvancedVI.elbo(advi, q_full, log_joint, 1000)

# check the covariance matrix
Sigma_hat = q_full.Σ
mu_hat = q_full.μ

samples = rand(q_full, 1000)
size(samples)

histogram(samples[1, :], label="Intercept")

histogram(samples[2, :], label="gamma 1")
histogram!(samples[3, :], label="gamma 2")
histogram!(samples[p, :], label="gamma 9")
histogram!(samples[p+1, :], label="gamma 10")

true_beta
histogram(samples[2 + p, :], label="beta 1")
histogram!(samples[3 + p, :], label="beta 2")
histogram!(samples[p + p, :], label="beta 9")
histogram!(samples[p+1 + p, :], label="beta 10")
