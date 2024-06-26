# Hierarchical Model with Spike and Slab prior
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections
using ProgressMeter

using StatsFuns
using Bijectors
using DiffResults
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack
using ADTypes
using Flux
using Zygote
using Turing
using AdvancedVI

include(joinpath("mixed_models_data_generation.jl"))
include(joinpath("mirror_statistic.jl"))
include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("relaxed_bernoulli.jl"))
include(joinpath("plot_functions.jl"))
include(joinpath("decayed_ada_grad.jl"))
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


data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_per_ind, p=p, p1=p1, p0=p0,
    include_random_int=true, random_intercept_sd=0.3,
    include_random_time=true, random_time_sd=0.5,
    include_random_slope=false, dtype=Float64
)


# <<<<<<<< Custom made model >>>>>>>>>>>

# Prior distributions
params_dict = OrderedDict()
num_params = 0

# Variance
params_dict["sigma_y"] = OrderedDict("size" => (1), "from" => 1, "to" => 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_y = truncated(Normal(0f0, 1f0), 0f0, Inf32)
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
prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.1f0, 0.1f0), p)
log_prior_gamma_logit(gamma_logit) = Distributions.logpdf(prior_gamma_logit, gamma_logit)

# prior sigma beta Slab
params_dict["sigma_slab"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_slab = truncated(Normal(0f0, 0.1f0), 0f0, Inf32)
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
                Normal(0f0, 5f0 * sigma_slab),
                Normal(0f0, sigma_slab)
            ], [gg, 1f0 - gg]) for gg in gamma
        ]),
        beta_fixed
    )
end

# Custom function
function log_prior_beta_fixed(
    w::AbstractArray{<:Real},
    sd_spike::Real,
    x::AbstractArray{<:Real};
    mu=0.,
    slab_multiplier=5.
    )
    sd::AbstractArray{<:Real} = [sd_spike;; sd_spike * slab_multiplier]

    w_ext = hcat(w, 1f0 .- w)
    xstd = -0.5f0 .* ((x .- mu) ./ sd).^2f0
    wstd = w_ext ./ (sqrt(2f0 .* pi) .* sd)
    offset = maximum(xstd .* wstd, dims=2)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=2)
    sum(log.(s) .+ offset)
end

# w = [0.1, 0.5, 0.9]
# sd_spike = 0.2
# x = [1., 0.1, 2.]
# log_prior_beta_fixed(w, sd_spike, x)
# logpdf_mixture_normal(w, sd_spike, x)

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
prior_beta0_fixed = Distributions.Normal(0f0, 5f0)
log_prior_beta0_fixed(beta0_fixed) = Distributions.logpdf(prior_beta0_fixed, beta0_fixed)

params_dict["sigma_beta0"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta0 = truncated(Normal(0f0, 1f0), 0f0, Inf)
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
prior_sigma_beta_time = truncated(Normal(0f0, 1f0), 0f0, Inf32)
log_prior_sigma_beta_time(sigma_beta_time) = Distributions.logpdf(prior_sigma_beta_time, sigma_beta_time)

params_dict["beta_time"] = OrderedDict("size" => (n_per_ind), "from" => num_params+1, "to" => num_params + n_per_ind, "bij" => identity)
num_params += n_per_ind
function log_prior_beta_time(sigma_beta_time, beta_time)
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(0f0, sigma_beta_time), n_per_ind),
        beta_time
    )
end

# Likelihood
function likelihood(;
    beta0_fixed, 
    beta0_random,
    beta_fixed,
    beta_time,
    sigma_y,
    Xfix,
    beta_random=zeros32(1),
    Xrand=zeros32(1)
    )
    Turing.arraydist([
        Distributions.MultivariateNormal(
            beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ beta_time[tt],
            ones32(n_individuals) .* sigma_y
        ) for tt in range(1, n_per_ind)
    ])
end
likelihood(
    beta0_fixed=data_dict["beta0_fixed"],
    beta0_random=randn32(n_individuals),
    beta_fixed=data_dict["beta_fixed"],
    beta_time=data_dict["beta_time"],
    sigma_y=1f0,
    Xfix=data_dict["Xfix"]
)

function log_likelihood(;
    y,
    beta0_fixed,
    beta0_random,
    beta_fixed,
    beta_time,
    sigma_y,
    Xfix,
    beta_random=zeros32(1),
    Xrand=zeros32(1)
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

log_likelihood(
    y=data_dict["y"],
    beta0_fixed=data_dict["beta0_fixed"],
    beta0_random=randn32(n_individuals),
    beta_fixed=data_dict["beta_fixed"],
    beta_time=data_dict["beta_time"],
    sigma_y=1f0,
    Xfix=data_dict["Xfix"]
)

# Joint
params_names = tuple(Symbol.(params_dict.keys)...)
proto_array = ComponentArray(;
    [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn32(params_dict[pp]["size"]), randn32(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
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
theta_hat = ones32(num_params) * 0.5f0
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

getq(ones32(num_weights))


# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 5_000, adtype=ADTypes.AutoZygote())
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
theta_init = randn(num_weights)
q = vi(log_joint, advi, getq, theta_init)

samples = rand(q, 2000)
size(samples)


histogram_posterior(samples, "sigma_y", params_dict)

density_posterior(samples, "beta_fixed", params_dict; plot_label=false)
scatter(posterior_summary(samples, "beta_fixed", params_dict; fun=mean))

density_posterior(samples, "gamma_logit", params_dict; plot_label=false)
scatter(posterior_summary(samples, "gamma_logit", params_dict; fun=mean))

histogram_posterior(samples, "sigma_slab", params_dict; plot_label=false)

histogram_posterior(samples, "beta0_fixed", params_dict; plot_label=false)

histogram_posterior(samples, "sigma_beta0", params_dict; plot_label=false)

density_posterior(samples, "beta_time", params_dict; plot_label=false)

density_posterior(samples, "beta0_random", params_dict; plot_label=false)

boxplot(vec(samples[params_dict["beta0_random"]["from"]:params_dict["beta0_random"]["to"], :]))


# FDR
classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    posterior_summary(samples, "gamma_logit", params_dict; fun=mean)[:,1] .> 0.5
)

# Mirror Statistic - FDR control
beta_post = samples[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"], :]
gamma_post = params_dict["gamma_logit"]["bij"].(samples[params_dict["gamma_logit"]["from"]:params_dict["gamma_logit"]["to"], :])
sigma_slab_post = params_dict["sigma_slab"]["bij"].(samples[params_dict["sigma_slab"]["from"], :])

sigma_slab_mean = mean(sigma_slab_post)
gamma_mean = mean(gamma_post, dims=2)

beta_ss_post = (Random.randn(p, 2000) .* sigma_slab_mean) .* (gamma_mean .< 0.5) .+ 
    beta_post .* (gamma_mean .>= 0.5)

plt = density(beta_ss_post[1, :], label=false)
for j in range(2, 10)
    density!(beta_ss_post[j, :], label=false)
end
for j in range(p0+1, p)
    density!(beta_ss_post[j, :], label=false)
end
display(plt)

posterior_ms = posterior_mirror_stat(
    beta_ss_post,
    fdr_target=0.3
)

posterior_ms = posterior_mirror_stat(
    beta_post,
    fdr_target=0.1
)

plt = density(posterior_ms["posterior_ms_coefs"][1, :], label=false)
for j in range(2, p0)
    density!(posterior_ms["posterior_ms_coefs"][j, :], label=false)
end
display(plt)

point_ms_coefs = mean(posterior_ms["posterior_ms_inclusion"], dims=2)

classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0,
    point_ms_coefs[:, 1] .> 0.5
)

boxplot(point_ms_coefs[:, 1])


# >>>>>>>>>>>>>>>> Manual training loop <<<<<<<<<<<<<<<<<

# Define objective
variational_objective = Turing.Variational.ELBO()

# Optimizer
optimizer = Turing.Variational.DecayedADAGrad()
optimizer = DecayedADAGrad()
optimizer = Flux.AdaGrad()
optimizer = Flux.Adam(0.001)

# VI algorithm
num_steps = 1000
samples_per_step = 1
alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoTracker())
alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

# --- Train loop ---
converged = false
step = 1
theta = randn(num_weights) * 0.1f0
elbo_trace = zeros(num_steps)

prog = ProgressMeter.Progress(num_steps, 1)
diff_results = DiffResults.GradientResult(theta)

while (step ≤ num_steps) && !converged
    # 1. Compute gradient and objective value; results are stored in `diff_results`
    AdvancedVI.grad!(variational_objective, alg, getq, log_joint, theta, diff_results, samples_per_step)

    # 2. Extract gradient from `diff_result`
    gradient_step = DiffResults.gradient(diff_results)

    # 3. Apply optimizer, e.g. multiplying by step-size
    diff_grad = VIAdaGrad.apply!(optimizer, theta, gradient_step)

    # 4. Update parameters
    @. theta = theta - diff_grad

    # 5. Do whatever analysis you want - Store ELBO value
    elbo_trace[step] = AdvancedVI.elbo(alg, getq(theta), log_joint, samples_per_step)

    step += 1

    ProgressMeter.next!(prog)
end

plot(elbo_trace, label="ELBO")
plot(elbo_trace[100:num_steps], label="ELBO")

q = getq(theta)
samples = rand(q, 2000)
size(samples)


#########

