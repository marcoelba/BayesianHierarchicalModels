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
using AdvancedVI
using StatsFuns
using Bijectors
using AdvancedVI
using StatsFuns
using ComponentArrays, UnPack

# Prior distribution
# Gaussian
prior = Distributions.MultivariateNormal(ones(p))
log_prior(beta) = Distributions.logpdf(prior, beta)

# Likelihood
likelihood(X, beta) = Distributions.MultivariateNormal(X * beta, ones(n))
log_likelihood(y, X, beta) = sum(Distributions.logpdf(likelihood(X, beta), y))
likelihood(X, true_beta)

# Joint
log_joint(beta) = log_likelihood(y, X, beta) + log_prior(beta)
log_joint(true_beta)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
# bijector transfrom FROM the latent space TO the REAL line

# base distribution
base_dist = Distributions.MultivariateNormal(zeros(p), ones(p))

proto_arr = ComponentArrays.ComponentArray(; L=zeros(p, 3), d=zeros(p), m=zeros(p))
proto_axes = ComponentArrays.getaxes(proto_arr)
num_params = length(proto_arr)

function getq(theta)
    L, d, m = begin
        UnPack.@unpack L, d, m = ComponentArrays.ComponentArray(theta, proto_axes)
        L, d, m
    end
    # The diagonal of the covariance matrix must be positive
    A = L * L'
    Lo = LinearAlgebra.LowerTriangular(A)
    D = Diagonal(StatsFuns.softplus.(d))
    S = Lo + D

    transformed_dist = m + S * base_dist
    
    return transformed_dist
end

theta = ones(num_params)
getq(theta)

advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_params)*0.1)
# Check the ELBO
AdvancedVI.elbo(advi, q_full, log_joint, 1000)

# check the covariance matrix
Sigma_hat = q_full.Σ
mu_hat = q_full.μ


# --------------------------------------------
# With Spike and Slab prior

# Spike and Slab
prior_relax_bern_logit = Turing.filldist(LogitRelaxedBernoulli(0.5, 0.01), p)
log_prior_relax_bern_logit(bern_logit) = Distributions.logpdf(relax_bern_logit, bern_logit)


function log_prior_beta(beta, gamma)
    prior_beta = Turing.arraydist(
        [GaussianSpikeSlab(0., 5., gg) for gg in gamma]
    )
    Distributions.logpdf(prior_beta, beta)
end
log_prior_beta(true_beta, StatsFuns.softmax(ones(p)))

prior_sigma_y = Distributions.truncated(Normal(0., 5.), 0., Inf)
log_prior_sigma(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)


# Likelihood
likelihood(X, beta, sigma_y) = Distributions.MultivariateNormal(X * beta, ones(n) * sigma_y)
log_likelihood(y, X, beta, sigma_y) = sum(Distributions.logpdf(likelihood(X, beta, sigma_y), y))
likelihood(X, true_beta, 2.)

# Joint
# tot number of parameters = p + p + 1
function log_joint(theta_hat)
    bern_logit = theta_hat[1:p]
    beta = theta_hat[(p + 1):(2 * p)]
    sigma_y = StatsFuns.softplus(theta_hat[p*2 + 1])
    log_likelihood(y, X, beta, sigma_y) + log_prior_relax_bern_logit(bern_logit) + log_prior_beta(beta, StatsFuns.softmax(bern_logit)) + log_prior_sigma(sigma_y)
end
# try
log_joint(ones(p+p+1))

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
# bijector transfrom FROM the latent space TO the REAL line

# MeanField -------------------------------
num_params = (p + p + 1) * 2
half_num_params = Int(num_params / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:half_num_params*2])
    )
end


# Factorised Normal -------------------------------
# base distribution
base_dist = Distributions.MultivariateNormal(zeros(p), ones(p))

proto_arr = ComponentArrays.ComponentArray(; L=zeros(p, 3), d=zeros(p), m=zeros(p))
proto_axes = ComponentArrays.getaxes(proto_arr)
num_params = length(proto_arr)

function getq(theta)
    L, d, m = begin
        UnPack.@unpack L, d, m = ComponentArrays.ComponentArray(theta, proto_axes)
        L, d, m
    end
    # The diagonal of the covariance matrix must be positive
    A = L * L'
    Lo = LinearAlgebra.LowerTriangular(A)
    D = Diagonal(StatsFuns.softplus.(d))
    S = Lo + D

    transformed_dist = m + S * base_dist
    
    return transformed_dist
end

theta = ones(num_params)
getq(theta)


# Inference
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_params)*0.1)
# Check the ELBO
AdvancedVI.elbo(advi, q_full, log_joint, 1000)

# check the covariance matrix
Sigma_hat = q_full.Σ
mu_hat = q_full.μ

true_beta
