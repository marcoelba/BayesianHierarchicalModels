# Hierarchical Model
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


# Generate hierarchical model data
# groups are the individuals (patients)
n_individuals = 10
n_per_ind = 50

n_total = n_individuals * n_per_ind

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
random_beta = Random.rand([-0.5, 0., 0.5], n_individuals, d)
# .+ overall_beta[(p-d+1):p]'
size(random_beta)
random_beta[:, 1]
mean(random_beta, dims=1)

# Outcome
y = zeros(n_individuals, n_per_ind)

# only fixed effects and random intercept
for ii in range(1, n_individuals)
    # Only random intercept
    # y[ii, :] = random_intercept[ii] .+ Random.randn(n_per_ind) * 0.5

    # Random intercept + Fixed effects
    y[ii, :] = random_intercept[ii] .+ Xfix[ii, :]' * overall_beta .+ Random.randn(n_per_ind) * 0.5

    # Random intercept + Random betas
    # y[ii, :] = random_intercept[ii] .+ Xfix[ii, :]' * overall_beta .+ Xrand[ii, :]' * random_beta[ii, :] .+ Random.randn(n_per_ind) * 0.5

end

random_intercept + Xfix * overall_beta + sum(Xrand .* random_beta, dims=2)[:, 1]


# define function using Turing syntax
@model function longitudinal_model(y, Xfix, Xrand)
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Intercept
    beta0_fixed ~ Turing.Normal(0., 5.)
    sigma_beta0 ~ truncated(Normal(0., 5.), 0., Inf)
    beta0_random ~ Turing.filldist(Turing.Normal(0., sigma_beta0), n_individuals)
    
    # Covariates
    beta_fixed ~ Turing.MultivariateNormal(zeros(p), 5.)

    # sigma_beta ~ truncated(Normal(0., 5.), 0., Inf)

    # beta_random ~ Turing.filldist(
    #     Turing.Normal(0., sigma_beta), n_individuals, d
    # )

    # mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ sum(Xrand .* beta_random, dims=2)[:, 1]

    mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed

    y ~ Distributions.MatrixNormal(ones(1, n_per_ind) .* mu, diagm(ones(n_individuals) * sigma_y) * I, diagm(ones(n_per_ind)))

    # y ~ Turing.filldist(Turing.MultivariateNormal(mu, ones(n_individuals) .* sigma_y), n_per_ind)

end


model = longitudinal_model(y, Xfix, Xrand)

nuts_lm = sample(model, NUTS(0.65), 1000)

# Overall Beta
plot(nuts_lm[["beta_fixed[1]", "beta_fixed[2]", "beta_fixed[$(p-1)]", "beta_fixed[$(p)]"]])
overall_beta

plot(nuts_lm[["sigma_y"]])
plot(nuts_lm[["sigma_beta"]])
plot(nuts_lm[["sigma_beta0"]])

# Random beta g=1
plot(nuts_lm[["beta_random[1,1]", "beta_random[2,1]"]])

# Intercept
plot(nuts_lm[["beta0_fixed", "beta0_random[1]" ,"beta0_random[2]", "beta0_random[5]"]])
random_intercept


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
params_dict = OrderedDict()
num_params = 0

# Variance
params_dict["sigma_y"] = OrderedDict("size" => (1), "from" => 1, "to" => 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_y = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_y(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Fixed effects
params_dict["beta_overall"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
num_params += p
prior_beta_overall = Distributions.MultivariateNormal(zeros(p), 5.)
log_prior_beta_overall(beta_overall) = Distributions.logpdf(prior_beta_overall, beta_overall)

# Random effects
params_dict["beta_random"] = OrderedDict("size" => (p, n_groups), "from" => num_params+1, "to" => num_params + p*n_groups, "bij" => identity)
num_params += p*n_groups
function log_prior_beta_random(beta_overall, beta_random)
    Distributions.logpdf(
        Turing.filldist(
            Distributions.MultivariateNormal(beta_overall, 1.), n_groups
        ),
        beta_random
    )
end

# Intercept
params_dict["beta0_overall"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => identity)
num_params += 1
prior_beta0_overall = Distributions.Normal(0., 5.)
log_prior_beta0_overall(beta0_overall) = Distributions.logpdf(prior_beta0_overall, beta0_overall)

params_dict["sigma_beta0"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta0 = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_beta0(sigma_beta0) = Distributions.logpdf(prior_sigma_beta0, sigma_beta0)

# Random Intercept
params_dict["beta0"] = OrderedDict("size" => (n_groups), "from" => num_params+1, "to" => num_params + n_groups, "bij" => identity)
num_params += n_groups
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
            ones(n_per_group) .* sigma_y
        ) for gg in range(1, n_groups)
    ])
end
likelihood(Xrand, random_beta', random_intercept, 1.)

log_likelihood(y, Xrand, beta_random, beta0, sigma_y) = sum(
    Distributions.logpdf(likelihood(Xrand, beta_random, beta0, sigma_y), y)
)
log_likelihood(y', Xrand, random_beta', random_intercept, 1.)

# ddd = Turing.arraydist([MultivariateNormal(ones(5) * mm, 1.) for mm in [0., 1., 2.]])
# Distributions.logpdf(ddd, ones(5, 3))

# Distributions.logpdf(MultivariateNormal(ones(5) * 0., 1.), ones(5)) +
# Distributions.logpdf(MultivariateNormal(ones(5) * 1., 1.), ones(5)) +
# Distributions.logpdf(MultivariateNormal(ones(5) * 2., 1.), ones(5))

# Joint

function log_joint(theta_hat)
    sigma_y = reshape(
        params_dict["sigma_y"]["bij"].(theta_hat[params_dict["sigma_y"]["from"]:params_dict["sigma_y"]["to"]]),
        params_dict["sigma_y"]["size"]
    )
    beta_overall = reshape(
        params_dict["beta_overall"]["bij"](theta_hat[params_dict["beta_overall"]["from"]:params_dict["beta_overall"]["to"]]),
        params_dict["beta_overall"]["size"]
    )
    beta_random = reshape(
        params_dict["beta_random"]["bij"](theta_hat[params_dict["beta_random"]["from"]:params_dict["beta_random"]["to"]]),
        params_dict["beta_random"]["size"]
    )
    beta0_overall = params_dict["beta0_overall"]["bij"](theta_hat[params_dict["beta0_overall"]["from"]:params_dict["beta0_overall"]["to"]])
    sigma_beta0 = params_dict["sigma_beta0"]["bij"].(theta_hat[params_dict["sigma_beta0"]["from"]:params_dict["sigma_beta0"]["to"]])
    beta0 = params_dict["beta0"]["bij"](theta_hat[params_dict["beta0"]["from"]:params_dict["beta0"]["to"]])

    loglik = log_likelihood(y', Xrand, beta_random, beta0, sigma_y)
    log_prior = log_prior_sigma_y(sigma_y[1]) +
        log_prior_beta_overall(beta_overall) +
        log_prior_beta_random(beta_overall, beta_random) + 
        log_prior_beta0_overall(beta0_overall[1]) +
        log_prior_sigma_beta0(sigma_beta0[1]) +
        log_prior_beta0(beta0_overall[1], sigma_beta0[1], beta0)
    
    loglik + log_prior
end
theta_hat = ones(num_params)
log_joint(ones(num_params))

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_weights = num_params * 2
half_num_params = Int(num_weights / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:half_num_params*2])
    )
end

getq(ones(num_weights))

# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(num_weights))

samples = rand(q, 1000)
size(samples)

params_dict
overall_beta

histogram(params_dict["sigma_y"]["bij"].(samples[1, :]), label="sigma")

histogram(samples[2, :], label="beta Overall 1")
histogram!(samples[3, :], label="beta Overall 2")
histogram!(samples[4, :], label="beta Overall 3")
histogram!(samples[5, :], label="beta Overall 4")

histogram(samples[46, :], label="beta 0 Overall")

histogram(params_dict["sigma_beta0"]["bij"].(samples[47, :]), label="beta Overall 1")

random_intercept
histogram(samples[48, :], label="beta 0 1")
histogram!(samples[49, :], label="beta 0 2")
histogram!(samples[50, :], label="beta 0 3")

