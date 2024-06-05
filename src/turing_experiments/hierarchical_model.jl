# Hierarchical model
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames

using AdvancedVI
using StatsFuns
using Bijectors
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack


# Generate hierarchical model data
n_samples = 1000
# groups are the individuals (patients)
n_groups = 10
n_per_group = Int(n_samples / n_groups)

p = 4
prop_non_zero = 0.5
p1 = Int(p * prop_non_zero)
p0 = p - p1

overall_beta = vcat(zeros(p0), Random.rand([-1., -2., 1, 2], p1))
overall_intercept = 2

# create random intercept
random_intercept = Random.rand([-1., -0.5, 0.5, 1], n_groups) .+ overall_intercept

# Fixed effects, same values for all groups
Xfix = Random.rand(n_per_group, p)

# create random beta
random_beta = Random.randn(n_groups, p) * 0.1 .+ overall_beta'
size(random_beta)
random_beta[:, 1]

Xrand = Random.randn(n_groups, n_per_group, p)

# Outcome
y = zeros(n_groups, n_per_group)

# only fixed effects and random intercept
for gg in range(1, n_groups)
    # Only random intercept
    # y[gg, :] = random_intercept[gg] .+ Random.randn(n_per_group) * 0.5

    # Random intercept + Fixed effects
    # y[gg, :] = random_intercept[gg] .+ Xfix * overall_beta + Random.randn(n_per_group) * 0.5

    # Random intercept + Random betas
    y[gg, :] = random_intercept[gg] .+ Xrand[gg, :, :] * random_beta[gg, :] + Random.randn(n_per_group) * 0.5

end
#


# define function using Turing syntax
@model function h_model(y, Xfix, Xrand)
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Covariates
    beta_overall ~ Turing.MultivariateNormal(zeros(p), 5.)
    beta_random ~ Turing.filldist(
        Turing.MultivariateNormal(beta_overall, 1.), n_groups
    )

    # Intercept
    beta0_overall ~ Turing.Normal(0., 5.)
    sigma_beta0 ~ truncated(Normal(0., 1.), 0., Inf)
    beta0 ~ Turing.filldist(Turing.Normal(beta0_overall, sigma_beta0), n_groups)

    for gg in range(1, n_groups)
        # Only random intercept
        # mu_gg = ones(n_per_group) .* beta0[gg]

        # Random intercept + fixed effects
        # mu_gg = beta0[gg] .+ Xfix * beta_overall

        # Random intercept + random betas
        mu_gg = beta0[gg] .+ Xrand[gg, :, :] * beta_random[:, gg]

        y[gg, :] ~ Turing.MultivariateNormal(mu_gg, sigma_y)
    end

end

model = h_model(y, Xfix, Xrand)

nuts_lm = sample(model, NUTS(0.65), 1000)

# Overall Beta
plot(nuts_lm[["beta_overall[1]", "beta_overall[2]", "beta_overall[3]", "beta_overall[4]"]])
overall_beta

# Random beta g=1
plot(nuts_lm[["beta_random[1,1]", "beta_random[2,1]", "beta_random[3,1]", "beta_random[4,1]"]])
# Random beta g=5
plot(nuts_lm[["beta_random[1,5]", "beta_random[2,5]", "beta_random[3,5]", "beta_random[4,5]"]])
# j=1
plot(nuts_lm[["beta_random[1,1]", "beta_random[1,2]", "beta_random[1,3]", "beta_random[1,4]"]])

# Intercept
plot(nuts_lm[["beta0_overall", "beta0[1]" ,"beta0[2]", "beta0[5]"]])



"""
Using Variational Inference
"""
model = h_model(y, Xfix, Xrand)

# ADVI
advi = ADVI(10, 1000)
q = vi(model, advi)

samples = rand(q, 1000)
size(samples)

histogram(samples[1, :], label="sigma")
histogram(samples[3, :], label="beta Overall 1")
histogram!(samples[5, :], label="beta Overall 2")
histogram!(samples[7, :], label="beta Overall 3")
histogram!(samples[9, :], label="beta Overall 4")


# Prior distributions

# Variance
prior_sigma_y = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_y(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Covariates
prior_beta_overall = Distributions.MultivariateNormal(zeros(p), 5.)
log_prior_beta_overall(beta_overall) = Distributions.logpdf(prior_beta_overall, beta_overall)

function log_prior_beta_random(beta_overall, beta_random)
    Distributions.logpdf(
        Turing.filldist(
            Distributions.MultivariateNormal(beta_overall, 1.), n_groups
        ),
        beta_random
    )
end

# Intercept
prior_beta0_overall = Distributions.Normal(0., 5.)
log_prior_beta0_overall(beta0_overall) = Distributions.logpdf(prior_beta0_overall, beta0_overall)

prior_sigma_beta0 = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_beta0(sigma_beta0) = Distributions.logpdf(prior_sigma_beta0, sigma_beta0)

function log_prior_beta0(beta0_overall, sigma_beta0, beta0)
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(beta0_overall, sigma_beta0), n_groups),
        beta0
    )
end

# Likelihood
function likelihood(Xrand, beta_random, beta0, sigma_y)
    Turing.arraydist([
        Distributions.MultivariateNormal(
            beta0[gg] .+ Xrand[gg, :, :] * beta_random[:, gg],
            ones(n_per_group) * sigma_y
        ) for gg in range(1, n_groups)
    ])
end
likelihood(Xrand, random_beta, random_intercept, 1.)

log_likelihood(y, Xrand, beta_random, beta0, sigma_y) = sum(
    Distributions.logpdf(likelihood(Xrand, beta_random, beta0, sigma_y), y)
)
log_likelihood(y, Xrand, random_beta, random_intercept, 1.)

# Joint
function log_joint(theta_hat)
    beta = theta_hat[1:p]
    sigma_y = StatsFuns.softplus(theta_hat[p+1])
    log_likelihood(y, X, beta, sigma_y) + log_prior_beta(beta) + log_prior_sigma(sigma_y)
end
log_joint([1. ,0., 1., 0., -1.])

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_params = (p + 1) * 2
half_num_params = Int(num_params / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:half_num_params*2])
    )
end

getq([1., 2., 0., -1., 1., 2., 0., -1., 1., -1.])

# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(num_params*2))
