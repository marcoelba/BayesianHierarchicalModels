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
    sigma2_y ~ Turing.truncated(Normal(0, 2), 0., Inf64)

    beta0 ~ Turing.Normal(0., 5.)

    # beta (reg coefficients)
    relax_bern = LogitRelaxedBernoulli(0.01, 0.01)
    gamma_logit ~ Turing.filldist(relax_bern, p)
    gamma = 1. ./ (1. .+ exp.(-gamma_logit))

    s_beta ~ Turing.truncated(Normal(0, 2), 0., Inf64)

    beta ~ Turing.arraydist([GaussianSpikeSlab(0., s_beta, gg) for gg in gamma])

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
plot(chains[["s_beta"]])

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

# prior sigma beta Slab
prior_sigma_beta = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_beta(sigma_beta) = Distributions.logpdf(prior_sigma_beta, sigma_beta)

# prior beta
function log_prior_beta(gamma, sigma_beta, beta)
    Distributions.logpdf(
        Turing.arraydist([
            GaussianSpikeSlab(0., sigma_beta, gg) for gg in gamma
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

# likelihood(X, true_beta, 0., 1.)
# log_likelihood(y, X, true_beta, 0., 1.)

# Joint
function log_joint(theta_hat)
    sigma_y = StatsFuns.softplus(theta_hat[1])
    beta0 = theta_hat[2]
    sigma_beta = StatsFuns.softplus(theta_hat[3])
    gamma_logit = theta_hat[4:(p+3)]
    beta = theta_hat[(p+4):(p+p+3)]
    
    log_prior = log_prior_beta0(beta0) +
        log_prior_sigma_y(sigma_y) +
        log_prior_gamma_logit(gamma_logit) +
        log_prior_beta(StatsFuns.logistic.(gamma_logit), sigma_beta, beta)

    loglik = log_likelihood(y, X, beta, beta0, sigma_y)

    loglik + log_prior
end
theta_hat = ones(p+p+3) * -2
log_joint(theta_hat)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
# bijector transfrom FROM the latent space TO the REAL line

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_params = p+p+3
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
# num_weights = p+p+p+4
num_weights = p+p+p+4+p

function getq(theta)
    
    q_sigma_y = [Normal(theta[1], StatsFuns.softplus(theta[2]))]
    q_beta0 = [Normal(theta[3], StatsFuns.softplus(theta[4]))]

    # alpha_logit = theta[5:(p+4)]
    # q_gamma_logit = [LogitRelaxedBernoulli(StatsFuns.logistic(alpha), 0.01) for alpha in alpha_logit]

    a_beta = theta[5:(p+4)]
    b_beta = theta[(p+5):(p+4+p)]
    q_gamma_logit = [Distributions.Beta(StatsFuns.softplus(a), StatsFuns.softplus(b))  for (a, b) in zip(a_beta, b_beta)]
    
    # mu_beta = theta[(p+5):(p+4+p)]
    # sigma_beta = theta[(p+5+p):(p+4+p+p)]
    # q_beta = [GaussianSpikeSlab(mu, StatsFuns.softplus(sigma), StatsFuns.logistic(gg)) for (mu, sigma, gg) in zip(mu_beta, sigma_beta, alpha_logit)]
    # q_beta = [Normal(mu, StatsFuns.softplus(sigma)) * StatsFuns.logistic(gg) for (mu, sigma, gg) in zip(mu_beta, sigma_beta, alpha_logit)]

    mu_beta = theta[(p+5+p):(p+4+p+p)]
    sigma_beta = theta[(p+5+p+p):(p+4+p+p+p)]
    q_beta = [Normal(mu, StatsFuns.softplus(sigma)) *  for (mu, sigma) in zip(mu_beta, sigma_beta)]

    Turing.arraydist(vcat(q_sigma_y, q_beta0, q_gamma_logit, q_beta))
end
rand(getq(randn(num_weights)))
logpdf(getq(randn(num_weights)), randn(Int(num_weights/2)))

# Inference
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_weights)*0.1)
# Check the ELBO
AdvancedVI.elbo(advi, q_full, log_joint, 1000)

samples = rand(q_full, 1000)
size(samples)

histogram(StatsFuns.softplus.(samples[1, :]), label="sigma y")

histogram(samples[2, :], label="Intercept")

histogram(StatsFuns.softplus.(samples[3, :]), label="sigma beta")

histogram(samples[4, :], label="gamma 1")
histogram!(samples[5, :], label="gamma 2")
histogram!(samples[p+2, :], label="gamma 9")
histogram!(samples[p+3, :], label="gamma 10")

true_beta
histogram(samples[4 + p, :], label="beta 1")
histogram!(samples[5 + p, :], label="beta 2")
histogram!(samples[p+2 + p, :], label="beta 9")
histogram!(samples[p+3 + p, :], label="beta 10")

