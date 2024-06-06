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
    sigma2_y ~ Turing.truncated(Normal(0, 2))

    beta0 ~ Turing.Normal(0., 5.)

    # beta (reg coefficients)
    relax_bern = LogitRelaxedBernoulli(0.01, 0.01)
    gamma_logit ~ Turing.filldist(relax_bern, p)
    gamma = 1. ./ (1. .+ exp.(-gamma_logit))

    beta ~ Turing.arraydist([GaussianSpikeSlab(0., 5., gg) for gg in gamma])

    mu = beta0 .+ X * beta

    return y ~ MultivariateNormal(mu, sigma2_y)
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

samples = rand(q, 10000)
size(samples)

histogram(StatsFuns.softplus.(samples[1, :]))
histogram(samples[2, :])

histogram(samples[3, :])
histogram!(samples[4, :])
histogram!(samples[11, :])
histogram!(samples[12, :])

histogram(samples[3 + p, :])
histogram!(samples[4 + p, :])
histogram!(samples[11 + p, :])
histogram!(samples[12 + p, :])


# -----------------------------------------------
# Same inference via AdvancedVI library
# -----------------------------------------------

# Prior distributions
prior_sigma_y = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_y(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Intercept
prior_beta0 = Normal(0., 5.)
log_prior_beta0(beta0) = Distributions.logpdf(prior_beta0, beta0)

# beta Spike and Slab
prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.5, 0.01), p)
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
function likelihood(X, beta, beta0, sigma_y)
    Distributions.MultivariateNormal(
        beta0 .+ X * beta, ones(n) * sigma_y
    )
end
log_likelihood(y, X, beta, beta0, sigma_y) = sum(Distributions.logpdf(likelihood(X, beta, beta0, sigma_y), y))
likelihood(X, true_beta, 0., 1.)
log_likelihood(y, X, true_beta, 0., 1.)

# Joint
function log_joint(theta_hat)
    sigma_y = StatsFuns.softplus(theta_hat[1])
    beta0 = theta_hat[2]
    gamma_logit = theta_hat[3:(p+2)]
    beta = theta_hat[(p+3):(p+p+2)]
    
    log_prior = log_prior_beta0(beta0) +
        log_prior_sigma_y(sigma_y) +
        log_prior_gamma_logit(gamma_logit) +
        log_prior_beta(StatsFuns.logistic.(gamma_logit), beta)

    loglik = log_likelihood(y, X, beta, beta0, sigma_y)

    loglik + log_prior
end
theta_hat = ones(p+p+2) * 2
log_joint(theta_hat)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
# bijector transfrom FROM the latent space TO the REAL line

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_params = p+p+2
num_weights = num_params * 2
half_num_params = Int(num_weights / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:num_weights])
    )
end

getq(ones(num_weights))


# Variational Spike and Slab
num_params = p+p+2
num_weights = num_params * 2
half_num_params = Int(num_weights / 2)

prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.5, 0.01), p)
log_prior_gamma_logit(gamma_logit) = Distributions.logpdf(prior_gamma_logit, gamma_logit)

function log_prior_beta(gamma, beta)
    Distributions.logpdf(
        Turing.arraydist([
            GaussianSpikeSlab(0., 2., gg) for gg in gamma
        ]),
        beta
    )
end


function getq(theta)
    d = length(theta) ÷ 2
    
    q_sigma_y = [Normal(theta[1], theta[2])]
    q_beta0 = [Normal(theta[3], theta[4])]

    gamma_logit = theta[5:(p+5)]
    q_gamma_logit = [LogitRelaxedBernoulli(gamma_logit, 0.01) for jj in range(1, p)]
    
    mu_beta = theta[(p+6):(p+6+p)]
    sigma_beta = theta[(p+7+p):(p+7+p+p)]
    q_beta = [GaussianSpikeSlab(0., 2., StatsFuns.logistic(gg)) for gg in gamma_logit]


    mu_vec = @inbounds theta[1:d]
    sigma_vec = @inbounds theta[(d + 1):(2 * d)]
    Turing.arraydist([
        Normal(mu, StatsFuns.softplus(sigma)) for (mu, sigma) in zip(mu_vec, sigma_vec)
    ])
end


advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_weights)*0.1)
# Check the ELBO
AdvancedVI.elbo(advi, q_full, log_joint, 1000)

samples = rand(q_full, 1000)
size(samples)

histogram(StatsFuns.softplus.(samples[1, :]), label="sigma y")

histogram(samples[2, :], label="Intercept")

histogram(samples[3, :], label="gamma 1")
histogram!(samples[4, :], label="gamma 2")
histogram!(samples[p+1, :], label="gamma 9")
histogram!(samples[p+2, :], label="gamma 10")

true_beta
histogram(samples[3 + p, :], label="beta 1")
histogram!(samples[4 + p, :], label="beta 2")
histogram!(samples[p+1 + p, :], label="beta 9")
histogram!(samples[p+2 + p, :], label="beta 10")

