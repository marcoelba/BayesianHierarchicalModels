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

