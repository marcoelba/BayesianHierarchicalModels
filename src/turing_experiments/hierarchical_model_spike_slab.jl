# Hierarchical Model with Spike and Slab prior
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


# Generate hierarchical model data
# groups are the individuals (patients)
n_individuals = 200
n_per_ind = 10

n_total = n_individuals * n_per_ind

# tot covariates
p = 50
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1

# Covariates with random effects
d = 2

Random.seed!(234)
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
random_intercept = Random.rand([-1., -0.5, 0.5, 1], n_individuals)
random_intercept = Random.randn(n_individuals) * 0.3

# Random Time effect
random_time_effect = Random.rand([-0.5, -0.3, 0.4, 0.6], n_per_ind)
random_time_effect = Random.randn(n_per_ind) * 0.5

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
    # No random components
    # y[ii, :] .= overall_intercept .+ Xfix[ii, :]' * overall_beta

    # Only random intercept
    # y[ii, :] = overall_intercept .+ random_intercept[ii]

    # Random intercept + Fixed effects
    # y[ii, :] .= random_intercept[ii] .+ Xfix[ii, :]' * overall_beta

    # Random time intercept + Fixed effects
    y[ii, :] = overall_intercept .+ random_intercept[ii] .+ random_time_effect .+ Xfix[ii, :]' * overall_beta

    # Random intercept + Random betas
    # y[ii, :] = random_intercept[ii] .+ Xfix[ii, :]' * overall_beta .+ Xrand[ii, :]' * random_beta[ii, :] .+ Random.randn(n_per_ind) * 0.5

end

# Add random noise

for tt in range(1, n_per_ind)
    y[:, tt] += Random.randn(n_individuals) * 1.
end


@model function longitudinal_model(y, Xfix, Xrand)
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Intercept
    beta0_fixed ~ Turing.Normal(0., 5.)

    sigma_beta0 ~ truncated(Normal(0., 5.), 0., Inf)

    beta0_random ~ Turing.MultivariateNormal(zeros(n_individuals), sigma_beta0)

    # beta0_random ~ Turing.filldist(Turing.Normal(0., sigma_beta0), n_individuals)
    
    sigma_beta_time ~ truncated(Normal(0., 1.), 0., Inf)
    beta_time ~ Turing.filldist(Turing.Normal(0., sigma_beta_time), n_per_ind)
    # beta_time ~ Turing.Normal(0., 2.)

    # Covariates Fixed Effects
    gamma_logit ~ Turing.filldist(LogitRelaxedBernoulli(0.1, 0.01), p)
    gamma = StatsFuns.logistic.(gamma_logit)

    sigma_beta_slab ~ Turing.truncated(Normal(0., 2.), 0., Inf64)

    # beta_fixed ~ Turing.arraydist([GaussianSpikeSlab(0., sigma_beta_slab, gg) for gg in gamma])

    beta_fixed ~ Turing.arraydist([
        Distributions.MixtureModel(Normal[
            Normal(0., 10. * sigma_beta_slab),
            Normal(0., sigma_beta_slab)
        ], [gg, 1. - gg]) for gg in gamma
    ])

    # sigma_beta ~ truncated(Normal(0., 5.), 0., Inf)

    # beta_random ~ Turing.filldist(
    #     Turing.Normal(0., sigma_beta), n_individuals, d
    # )

    # mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ sum(Xrand .* beta_random, dims=2)[:, 1]

    # mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed
    # y ~ Turing.filldist(
    #     Turing.MultivariateNormal(mu, ones(n_individuals) .* sigma_y),
    #     n_per_ind
    # )

    # time intercept
    mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed
    y ~ Turing.arraydist([
        Turing.MultivariateNormal(mu .+ beta_time[tt] , ones(n_individuals) .* sigma_y)
        for tt in range(1, n_per_ind)
    ])

end


model = longitudinal_model(y, Xfix, Xrand)
nuts_lm = sample(model, NUTS(0.65), 1000)


"""
Using Variational Inference
"""
model = longitudinal_model(y, Xfix, Xrand)

# ADVI
advi = ADVI(5, 1000)
q = vi(model, advi)

samples = rand(q, 1000)
size(samples)
tot_params = size(samples)[1]


# <<<<<<<< Custom made model >>>>>>>>>>>

# Prior distributions
params_dict = OrderedDict()
num_params = 0

# Variance
params_dict["sigma_y"] = OrderedDict("size" => (1), "from" => 1, "to" => 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_y = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_y(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Fixed effects
params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
num_params += p
prior_beta_fixed = Distributions.MultivariateNormal(zeros(p), 5.)
log_prior_beta_fixed(beta_fixed) = Distributions.logpdf(prior_beta_fixed, beta_fixed)

# Random effects
# params_dict["beta_random"] = OrderedDict("size" => (p, n_groups), "from" => num_params+1, "to" => num_params + p*n_groups, "bij" => identity)
# num_params += p*n_groups
# function log_prior_beta_random(beta_overall, beta_random)
#     Distributions.logpdf(
#         Turing.filldist(
#             Distributions.MultivariateNormal(beta_overall, 1.), n_groups
#         ),
#         beta_random
#     )
# end

# Intercept
params_dict["beta0_fixed"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => identity)
num_params += 1
prior_beta0_fixed = Distributions.Normal(0., 5.)
log_prior_beta0_fixed(beta0_fixed) = Distributions.logpdf(prior_beta0_fixed, beta0_fixed)

params_dict["sigma_beta0"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta0 = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_beta0(sigma_beta0) = Distributions.logpdf(prior_sigma_beta0, sigma_beta0)

# Random Intercept
params_dict["beta0_random"] = OrderedDict("size" => (n_individuals), "from" => num_params+1, "to" => num_params + n_individuals, "bij" => identity)
num_params += n_individuals
function log_prior_beta0_random(beta0_fixed, sigma_beta0, beta0_random)
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(beta0_fixed, sigma_beta0), n_individuals),
        beta0_random
    )
end

# Random time component
params_dict["sigma_beta_time"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta_time = truncated(Normal(0., 1.), 0., Inf)
log_prior_sigma_beta_time(sigma_beta_time) = Distributions.logpdf(prior_sigma_beta_time, sigma_beta_time)

params_dict["beta_time"] = OrderedDict("size" => (n_per_ind), "from" => num_params+1, "to" => num_params + n_per_ind, "bij" => identity)
num_params += n_per_ind
function log_prior_beta_time(sigma_beta_time, beta_time)
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(0., sigma_beta_time), n_per_ind),
        beta_time
    )
end

# Likelihood
function likelihood(;beta0_fixed, beta0_random, beta_fixed, beta_time, sigma_y, Xfix, beta_random=NaN, Xrand=NaN)
    Turing.arraydist([
        Distributions.MultivariateNormal(
            beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ beta_time[tt],
            ones(n_individuals) .* sigma_y
        ) for tt in range(1, n_per_ind)
    ])
end
likelihood(
    beta0_fixed=overall_intercept,
    beta0_random=randn(n_individuals),
    beta_fixed=overall_beta,
    beta_time=random_time_effect,
    sigma_y=1.,
    Xfix=Xfix
)

log_likelihood(;y, beta0_fixed, beta0_random, beta_fixed, beta_time, sigma_y, Xfix, beta_random=NaN, Xrand=NaN) = sum(
    Distributions.logpdf(likelihood(
        beta0_fixed=beta0_fixed,
        beta0_random=beta0_random,
        beta_fixed=beta_fixed,
        beta_time=beta_time,
        sigma_y=sigma_y,
        Xfix=Xfix
    ), y)
)
log_likelihood(
    y=y,
    beta0_fixed=overall_intercept,
    beta0_random=randn(n_individuals),
    beta_fixed=overall_beta,
    beta_time=random_time_effect,
    sigma_y=1.,
    Xfix=Xfix
)

# Joint
params_names = tuple(Symbol.(params_dict.keys)...)
proto_array = ComponentArray(;
    [Symbol(pp) => randn(params_dict[pp]["size"]) for pp in params_dict.keys]...
)
proto_axes = getaxes(proto_array)
num_params = length(proto_array)


proto_array = ComponentArray(;
    x=randn(3), y=randn(6)
)
proto_axes = getaxes(proto_array)
num_params = length(proto_array)


(gg, ff) = begin
    @unpack (x, y) = ComponentArray(ones(num_params), proto_axes)
    (x, y)
end



function log_joint(theta_hat)
    params_names = begin
        @unpack params_names = ComponentArray(theta_hat, proto_axes)
        params_names
    end

    sigma_y = params_dict["sigma_y"]["bij"].(
        theta_hat[params_dict["sigma_y"]["from"]]

    )

    beta_fixed = params_dict["beta_fixed"]["bij"].(theta_hat[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]])

    # beta_random = reshape(
    #     params_dict["beta_random"]["bij"](theta_hat[params_dict["beta_random"]["from"]:params_dict["beta_random"]["to"]]),
    #     params_dict["beta_random"]["size"]
    # )
    beta0_fixed = params_dict["beta0_fixed"]["bij"](theta_hat[params_dict["beta0_fixed"]["from"]])
    sigma_beta0 = params_dict["sigma_beta0"]["bij"].(theta_hat[params_dict["sigma_beta0"]["from"]])
    beta0_random = params_dict["beta0_random"]["bij"](theta_hat[params_dict["beta0_random"]["from"]:params_dict["beta0_random"]["to"]])

    sigma_beta_time = params_dict["sigma_beta_time"]["bij"].(theta_hat[params_dict["sigma_beta_time"]["from"]])
    beta_time = params_dict["beta_time"]["bij"](theta_hat[params_dict["beta_time"]["from"]:params_dict["beta_time"]["to"]])

    loglik = log_likelihood(
        y=y,
        beta0_fixed=beta0_fixed,
        beta0_random=beta0_random,
        beta_fixed=beta_fixed,
        beta_time=beta_time,
        sigma_y=sigma_y,
        Xfix=Xfix
    )

    log_prior = log_prior_sigma_y(sigma_y) +
        log_prior_beta_fixed(beta_fixed) +
        log_prior_beta0_fixed(beta0_fixed) +
        log_prior_sigma_beta0(sigma_beta0) +
        log_prior_beta0_random(beta0_fixed, sigma_beta0, beta0_random) +
        log_prior_sigma_beta_time(sigma_beta_time) +
        log_prior_beta_time(sigma_beta_time, beta_time)
    
    loglik + log_prior
end
theta_hat = ones(num_params)
log_joint(theta_hat)

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
advi = AdvancedVI.ADVI(10, 5_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(num_weights))

samples = rand(q, 1000)
size(samples)

params_dict
overall_beta

histogram(params_dict["sigma_y"]["bij"].(samples[1, :]), label="sigma_y")

histogram(samples[2, :], label="beta Overall 1")
histogram!(samples[3, :], label="beta Overall 2")
histogram!(samples[4, :], label="beta Overall 3")
histogram!(samples[5, :], label="beta Overall 4")

histogram(samples[48, :], label="beta Overall 1")
histogram!(samples[49, :], label="beta Overall 2")
histogram!(samples[50, :], label="beta Overall 3")
histogram!(samples[51, :], label="beta Overall 4")

overall_intercept
histogram(samples[52, :], label="beta 0 fixed")

histogram(params_dict["sigma_beta0"]["bij"].(samples[53, :]), label="sigma beta 0")

histogram(params_dict["sigma_beta_time"]["bij"].(samples[254, :]), label="sigma beta time")

# beta time
params_dict["beta_time"]
random_time_effect
plt = histogram(samples[params_dict["beta_time"]["from"], :], label="beta time $(1)")
for tt in range(2, n_per_ind)
    histogram!(samples[params_dict["beta_time"]["from"] + tt - 1, :], label="beta time $(tt)", alpha=0.5)
end
display(plt)

random_intercept
histogram(samples[48, :], label="beta 0 1")
histogram!(samples[49, :], label="beta 0 2")
histogram!(samples[50, :], label="beta 0 3")

