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
using ADTypes

include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("relaxed_bernoulli.jl"))
include(joinpath("../utils/classification_metrics.jl"))
include(joinpath("../utils/classification_metrics.jl"))


# Generate hierarchical model data
# groups are the individuals (patients)
n_individuals = 200
n_per_ind = 10

n_total = n_individuals * n_per_ind

# tot covariates
p = 200
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1

# Covariates with random effects
d = 2


function generate_mixed_model_data(;
    n_individuals, n_time_points,
    p, p1, p0, beta_pool=[-1., -2., 1, 2], obs_noise_sd=1.,
    include_random_int=true, random_intercept_sd=0.3,
    include_random_time=true, random_time_sd=0.5,
    include_random_slope=false, p_random_covs=0, random_slope_sd=0.5
    )

    data_dict = Dict()

    Random.seed!(234)
    # > FIXED EFFETCS <
    # Fixed effects, baseline covariates (DO NOT change over time)
    Xfix = Random.rand(n_individuals, p)
    data_dict["Xfix"] = Xfix

    # Fixed Coeffcients
    beta_fixed = vcat(zeros(p0), Random.rand(beta_pool, p1))
    beta0_fixed = 1.

    data_dict["beta_fixed"] = beta_fixed
    data_dict["beta0_fixed"] = beta0_fixed
    
    # > RANDOM EFFETCS <
    
    # Random Intercept (one per individual)
    if include_random_int
        beta0_random = Random.randn(n_individuals) * random_intercept_sd
        data_dict["beta0_random"] = beta0_random
    end
    
    # Random Time effect
    if include_random_time
        beta_time = Random.randn(n_time_points) * random_time_sd
        data_dict["beta_time"] = beta_time
    end

    # Beta Coefficients (only random deviations)
    if include_random_slope
        Xrand = Xfix[:, (p - p_random_covs + 1):p]
        beta_random = Random.randn(n_individuals, d) * random_slope_sd
        data_dict["beta_random"] = beta_random
    end
    
    # Outcome
    y = zeros(n_individuals, n_time_points)
    
    # only fixed effects and random intercept
    for ii in range(1, n_individuals)
        y[ii, :] .= beta0_fixed .+ Xfix[ii, :]' * beta_fixed

        if include_random_int
            y[ii, :] = y[ii, :] .+ beta0_random[ii]
        end

        if include_random_time
            y[ii, :] = y[ii, :] .+ beta_time
        end
    
        if include_random_slope
            y[ii, :] = y[ii, :] .+ Xrand[ii, :] * beta_random[ii, :]
        end

        # obs noise
        y[ii, :] += Random.randn(n_time_points) * obs_noise_sd
    end
    data_dict["y"] = y

    return data_dict
end

data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_per_ind, p=p, p1=p1, p0=p0,
    include_random_int=true, random_intercept_sd=0.3,
    include_random_time=true, random_time_sd=0.5
)


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

# Normal Distribution
# params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
# num_params += p
# prior_beta_fixed = Distributions.MultivariateNormal(zeros(p), 5.)
# log_prior_beta_fixed(beta_fixed) = Distributions.logpdf(prior_beta_fixed, beta_fixed)

# Spike and Slab distribution
# prob Spike and Slab
params_dict["gamma_logit"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => StatsFuns.logistic)
num_params += p
prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.1, 0.1), p)
log_prior_gamma_logit(gamma_logit) = Distributions.logpdf(prior_gamma_logit, gamma_logit)

# prior sigma beta Slab
params_dict["sigma_slab"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_slab = truncated(Normal(0., 0.1), 0., Inf)
log_prior_sigma_slab(sigma_slab) = Distributions.logpdf(prior_sigma_slab, sigma_slab)

# prior beta
# SS
# function log_prior_beta(gamma, sigma_beta, beta)
#     Distributions.logpdf(
#         Turing.arraydist([
#             GaussianSpikeSlab(0., sigma_beta, gg) for gg in gamma
#         ]),
#         beta
#     )
# end

# Continuous Mixture
params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
num_params += p
function log_prior_beta_fixed(gamma, sigma_slab, beta_fixed)
    Distributions.logpdf(
        Turing.arraydist([
            Distributions.MixtureModel(Normal[
                Normal(0., 10. * sigma_slab),
                Normal(0., sigma_slab)
            ], [gg, 1. - gg]) for gg in gamma
        ]),
        beta_fixed
    )
end

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

function log_likelihood(;y, beta0_fixed, beta0_random, beta_fixed, beta_time, sigma_y, Xfix, beta_random=NaN, Xrand=NaN)
    sum(
        Distributions.logpdf(likelihood(
            beta0_fixed=beta0_fixed,
            beta0_random=beta0_random,
            beta_fixed=beta_fixed,
            beta_time=beta_time,
            sigma_y=sigma_y,
            Xfix=Xfix
        ), y)
    )
end

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
    [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn(params_dict[pp]["size"]), randn(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
)
proto_axes = getaxes(proto_array)
num_params = length(proto_array)


function log_joint(theta_hat)
    begin
        params_names = ComponentArray(theta_hat, proto_axes)
    end

    sigma_y = params_dict["sigma_y"]["bij"].(params_names.sigma_y)

    gamma = params_dict["gamma_logit"]["bij"].(params_names.gamma_logit)
    sigma_slab = params_dict["sigma_slab"]["bij"].(params_names.sigma_slab)
    beta_fixed = params_names.beta_fixed

    # beta_random = reshape(
    #     params_dict["beta_random"]["bij"](theta_hat[params_dict["beta_random"]["from"]:params_dict["beta_random"]["to"]]),
    #     params_dict["beta_random"]["size"]
    # )
    beta0_fixed = params_names.beta0_fixed
    sigma_beta0 = params_dict["sigma_beta0"]["bij"].(params_names.sigma_beta0)
    beta0_random = params_names.beta0_random

    sigma_beta_time = params_dict["sigma_beta_time"]["bij"].(params_names.sigma_beta_time)
    beta_time = params_names.beta_time

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
        log_prior_gamma_logit(params_names.gamma_logit) +
        log_prior_sigma_slab(sigma_slab) +
        log_prior_beta_fixed(gamma, sigma_slab, beta_fixed) +
        log_prior_beta0_fixed(beta0_fixed) +
        log_prior_sigma_beta0(sigma_beta0) +
        log_prior_beta0_random(beta0_fixed, sigma_beta0, beta0_random) +
        log_prior_sigma_beta_time(sigma_beta_time) +
        log_prior_beta_time(sigma_beta_time, beta_time)
    
    loglik + log_prior
end
theta_hat = ones(num_params)*0.5
log_joint(theta_hat)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_weights = num_params * 2
half_num_params = num_params

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:half_num_params*2])
    )
end

getq(ones(num_weights))

# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 5_000, adtype=ADTypes.AutoTracker() )
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(num_weights))

samples = rand(q, 2000)
size(samples)

params_dict
overall_beta

function posterior_summary(samples, param, param_dict; fun)
    fun(
        param_dict[param]["bij"].(
        samples[params_dict[param]["from"]:params_dict[param]["to"], :]
        ),
        dims=2
    )
end


function hist_posterior(samples, param, param_dict; plot_label=true)
    from = param_dict[param]["from"]
    to = param_dict[param]["to"]

    label = false
    if plot_label
        label = "$(param)_1"
    end
    plt = histogram(param_dict[param]["bij"].(samples[from, :]), label=label)

    if (to - from) > 1
        for pp in range(from+1, to)
            if plot_label
                label = "$(param)_$(pp)"
            end
            histogram!(param_dict[param]["bij"].(samples[pp, :]), label=label)
        end
    end
    display(plt)
end

hist_posterior(samples, "sigma_y", params_dict)

hist_posterior(samples, "beta_fixed", params_dict; plot_label=false)
scatter(posterior_summary(samples, "beta_fixed", params_dict; fun=mean))

hist_posterior(samples, "gamma_logit", params_dict; plot_label=false)
scatter(posterior_summary(samples, "gamma_logit", params_dict; fun=mean))

hist_posterior(samples, "sigma_slab", params_dict; plot_label=false)

hist_posterior(samples, "beta0_fixed", params_dict; plot_label=false)

hist_posterior(samples, "sigma_beta0", params_dict; plot_label=false)

hist_posterior(samples, "beta_time", params_dict; plot_label=false)

hist_posterior(samples, "beta0_random", params_dict; plot_label=false)

boxplot(vec(samples[params_dict["beta0_random"]["from"]:params_dict["beta0_random"]["to"], :]))


# FDR
classification_metrics.wrapper_metrics(
    overall_beta .!= 0.,
    posterior_summary(samples, "gamma_logit", params_dict; fun=mean)[:,1] .> 0.5
)
