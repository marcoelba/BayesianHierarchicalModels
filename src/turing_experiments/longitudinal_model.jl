# Hierarchical model
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


# Generate longitudinal model data
n_individuals = 100
n_per_ind = 5

n_tot = n_individuals * n_per_ind

# tot covariates
p = 10
prop_non_zero = 0.5
p1 = Int(p * prop_non_zero)
p0 = p - p1

# Covariates with random effects
d = 2

# > FIXED EFFETCS <
# Fixed effects, baseline covariates (DO NOT change over time)
Xfix = Random.rand(n_individuals, p)
size(Xfix)

# Coeffcients
overall_beta = vcat(zeros(p0), Random.rand([-1., -2., 1, 2], p1))
overall_intercept = 2

# > RANDOM EFFETCS <
Xrand = Xfix[:, (p-d+1):p]

# Random Intercept (one per individual)
random_intercept = Random.rand([-1., -0.5, 0.5, 1], n_individuals) .+ overall_intercept

# Beta Coefficients (only random deviations)
random_beta = Random.randn(n_individuals, d) * 0.5
# .+ overall_beta[(p-d+1):p]'
size(random_beta)
random_beta[:, 1]
mean(random_beta, dims=1)

# Outcome
y = zeros(n_individuals, n_per_ind)

# only fixed effects and random intercept
for ii in range(1, n_individuals)
    # Only random intercept
    # y[gg, :] = random_intercept[gg] .+ Random.randn(n_per_group) * 0.5

    # Random intercept + Fixed effects
    # y[gg, :] = random_intercept[gg] .+ Xfix * overall_beta + Random.randn(n_per_group) * 0.5

    # Random intercept + Random betas
    y[ii, :] = random_intercept[ii] .+ Xfix[ii, :]' * overall_beta .+ Xrand[ii, :]' * random_beta[ii, :] .+ Random.randn(n_per_ind) * 0.5

end

random_intercept + Xfix * overall_beta + sum(Xrand .* random_beta, dims=2)[:, 1]


@model function longitudinal_model(y, Xfix, Xrand)
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Covariates
    beta_fixed ~ Turing.MultivariateNormal(zeros(p), 5.)
    sigma_beta ~ truncated(Normal(0., 5.), 0., Inf)
    beta_random ~ Turing.filldist(
        Turing.Normal(0., sigma_beta), n_individuals, d
    )

    # Intercept
    beta0_fixed ~ Turing.Normal(0., 5.)
    sigma_beta0 ~ truncated(Normal(0., 5.), 0., Inf)
    beta0_random ~ Turing.filldist(Turing.Normal(0., sigma_beta0), n_individuals)

    mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ sum(Xrand .* beta_random, dims=2)[:, 1]

    y ~ Turing.filldist(Turing.MultivariateNormal(mu, ones(n_individuals) .* sigma_y), n_per_ind)

end

model = longitudinal_model(y, Xfix, Xrand)

nuts_lm = sample(model, NUTS(0.65), 1000)

plot(nuts_lm[["sigma_y"]])

# Overall Beta
plot(nuts_lm[["beta_fixed[1]", "beta_fixed[2]", "beta_fixed[9]", "beta_fixed[10]"]])
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
model = longitudinal_model(y, Xfix, Xrand)

# ADVI
advi = ADVI(10, 1000)
q = vi(model, advi)

mu_est = q.dist.m
mu_est[2:p+1]

samples = rand(q, 1000)
size(samples)

histogram(samples[1, :], label="sigma_y")
