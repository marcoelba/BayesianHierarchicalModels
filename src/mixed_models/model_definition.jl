# Model definition

using Distributions
using Random
using LinearAlgebra
using OrderedCollections

using StatsFuns
using Bijectors
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack
using Flux
using Turing
using AdvancedVI

include(joinpath("relaxed_bernoulli.jl"))


# Likelihood
function likelihood(;
    beta0_fixed::Float32,
    beta0_random::AbstractArray{Float32},
    beta_fixed::AbstractArray{Float32},
    beta_time::AbstractArray{Float32},
    sigma_y::Float32,
    Xfix::AbstractArray{Float32},
    beta_random::AbstractArray{Float32}=zeros32(1),
    Xrand::AbstractArray{Float32}=zeros32(1)
    )
    Turing.arraydist([
        Distributions.MultivariateNormal(
            beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ beta_time[tt],
            ones32(n_individuals) .* sigma_y
        ) for tt in range(1, n_per_ind)
    ])
end


function log_likelihood(;
    y::AbstractArray{Float32},
    beta0_fixed::Float32,
    beta0_random::AbstractArray{Float32},
    beta_fixed::AbstractArray{Float32},
    beta_time::AbstractArray{Float32},
    sigma_y::Float32,
    Xfix::AbstractArray{Float32},
    beta_random::AbstractArray{Float32}=zeros32(1),
    Xrand::AbstractArray{Float32}=zeros32(1)
    )
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


function log_spike_slab_distribution(
    w::AbstractArray{<:Float32},
    sd_spike::Float32,
    x::AbstractArray{<:Float32};
    mu=Float32(0),
    slab_multiplier=Float32(20.)
    )
    sd = hcat(sd_spike * slab_multiplier, sd_spike)

    w_ext = hcat(w, 1f0 .- w)
    xstd = -0.5f0 .* ((x .- mu) ./ sd).^2f0
    wstd = w_ext ./ (sqrt(2f0 .* Float32(pi)) .* sd)
    offset = maximum(xstd .* wstd, dims=2)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=2)
    sum(log.(s) .+ offset)
end


function update_priors_set(params_dict::OrderedDict; name::String, size, bij)

    params_dict[name] = OrderedDict(
        "size" => (size),
        "bij" => bij
    )

    return params_dict
end


function get_log_prior(; p, n_individuals, n_per_ind)
    # Prior distributions
    params_dict = OrderedDict()
    logpdf_functions_dict = OrderedDict()

    # Variance y
    update_priors_set(
        params_dict,
        name="sigma_y",
        size=1,
        bij=StatsFuns.softplus
    )

    logpdf_functions_dict["sigma_y"] = x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )

    # Fixed effects
    # Spike and Slab distribution
    update_priors_set(params_dict, name="gamma_logit", size=p, bij=StatsFuns.logistic)

    logpdf_functions_dict["gamma_logit"] = x::AbstractArray{Float32} -> Distributions.logpdf(
        Turing.filldist(LogitRelaxedBernoulli(0.1f0, 0.1f0), length(x)),
        x
    )

    # prior sigma beta Slab
    update_priors_set(params_dict, name="sigma_spike", size=1, bij=StatsFuns.softplus)

    logpdf_functions_dict["sigma_spike"] = x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 0.1f0), 0f0, Inf32),
        x
    )

    # Continuous Mixture
    update_priors_set(params_dict, name="beta_fixed", size=p, bij=identity)

    logpdf_functions_dict["beta_fixed"] = (w::AbstractArray{<:Float32}, sd_spike::Float32, x::AbstractArray{<:Float32}) -> log_spike_slab_distribution(w, sd_spike, x)

    # Intercept
    update_priors_set(params_dict, name="beta0_fixed", size=1, bij=identity)

    logpdf_functions_dict["beta0_fixed"] = x::Float32 -> Distributions.logpdf(
        Distributions.Normal(0f0, 5f0),
        x
    )

    # Random Intercept
    update_priors_set(params_dict, name="sigma_beta0", size=1, bij=StatsFuns.softplus)

    logpdf_functions_dict["sigma_beta0"] = x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf),
        x
    )

    update_priors_set(params_dict, name="beta0_random", size=n_individuals, bij=identity)

    logpdf_functions_dict["beta0_random"] = (mean_beta0::Float32, sigma_beta0::Float32, x::AbstractArray{Float32}) -> Distributions.logpdf(
        Turing.filldist(Distributions.Normal(mean_beta0, sigma_beta0), length(x)),
        x
    )

    # Random time component
    update_priors_set(params_dict, name="sigma_beta_time", size=1, bij=StatsFuns.softplus)

    logpdf_functions_dict["sigma_beta_time"] = x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )

    update_priors_set(params_dict, name="beta_time", size=n_per_ind, bij=identity)

    logpdf_functions_dict["beta_time"] = (sigma_beta_time::Float32, x::AbstractArray{Float32}) -> Distributions.logpdf(
        Turing.filldist(Distributions.Normal(0f0, sigma_beta_time), length(x)),
        x
    )

    return params_dict, logpdf_functions_dict

end


params_dict, logpdf_functions_dict = get_log_prior(p=0, n_individuals=5, n_per_ind=2)

# num_params = length(proto_array)


function log_joint(theta_hat::AbstractArray{Float32}, params_dict)
    # Joint
    params_names = tuple(Symbol.(params_dict.keys)...)
    proto_array = ComponentArray(;
        [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn32(params_dict[pp]["size"]), randn32(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
    )
    proto_axes = getaxes(proto_array)

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
        y=data_dict["y"],
        beta0_fixed=beta0_fixed,
        beta0_random=beta0_random,
        beta_fixed=beta_fixed,
        beta_time=beta_time,
        sigma_y=sigma_y,
        Xfix=data_dict["Xfix"]
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

# theta_hat = ones32(num_params) * 0.5f0
# log_joint(theta_hat)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
dim_q = num_params * 2
half_dim_q = num_params

function getq(theta::AbstractArray{Float32})
    Distributions.MultivariateNormal(
        theta[1:half_dim_q],
        StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
    )
end
