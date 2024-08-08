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

include(joinpath("decayed_ada_grad.jl"))
include(joinpath("mixed_models_data_generation.jl"))
include(joinpath("mirror_statistic.jl"))
include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("../mixed_models/relaxed_bernoulli.jl"))
include(joinpath("plot_functions.jl"))
include(joinpath("../utils/classification_metrics.jl"))


# Generate hierarchical model data
# groups are the individuals (patients)
n_individuals = 200
n_per_ind = 5

n_total = n_individuals * n_per_ind

# tot covariates
p = 500
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.2

# Covariates with random effects
d = 2


data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_per_ind,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    include_random_int=true, random_intercept_sd=0.5,
    include_random_time=true, random_time_sd=0.5,
    include_random_slope=false
)

cor(data_dict["y"], dims=1)
mean(cor(data_dict["y"], dims=2), dims=1)

# Check ESS
println(Turing.ess(data_dict["y"], kind=:basic))
println(Turing.ess(data_dict["y"][1, :], kind=:basic))
println(Turing.ess(data_dict["y"][:, 1], kind=:basic))
println(Turing.ess(reshape(data_dict["y"], (n_individuals*n_per_ind)), kind=:basic))


"""
Using Variational Inference
"""

# Prior distributions
params_dict = OrderedDict()
num_params = 0

# Variance
params_dict["sigma_y"] = OrderedDict("size" => (1), "from" => 1, "to" => 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_y = truncated(Normal(0f0, 1f0), 0f0, Inf32)
log_prior_sigma_y(sigma_y::Float32) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Fixed effects

# Spike and Slab distribution
# prob Spike and Slab
params_dict["gamma_logit"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => StatsFuns.logistic)
num_params += p
prior_gamma_logit = Turing.filldist(LogitRelaxedBernoulli(0.1f0, 0.1f0), p)
log_prior_gamma_logit(gamma_logit::AbstractArray{Float32}) = Distributions.logpdf(prior_gamma_logit, gamma_logit)

# prior sigma beta Slab
params_dict["sigma_slab"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_slab = truncated(Normal(0f0, 0.1f0), 0f0, Inf32)
log_prior_sigma_slab(sigma_slab::Float32) = Distributions.logpdf(prior_sigma_slab, sigma_slab)

# prior beta
# SS
function log_prior_beta_fixed(gamma, sigma_beta, beta)
    Distributions.logpdf(
        Turing.arraydist([
            GaussianSpikeSlab(0., sigma_beta, gg) for gg in gamma
        ]),
        beta
    )
end

params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
num_params += p

# OR Continuous Mixture
# Custom function
function log_prior_beta_fixed(
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
log_prior_beta0_fixed(beta0_fixed::Float32) = Distributions.logpdf(prior_beta0_fixed, beta0_fixed)

params_dict["sigma_beta0"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta0 = truncated(Normal(0f0, 1f0), 0f0, Inf)
log_prior_sigma_beta0(sigma_beta0::Float32) = Distributions.logpdf(prior_sigma_beta0, sigma_beta0)

# Random Intercept
params_dict["beta0_random"] = OrderedDict("size" => (n_individuals), "from" => num_params+1, "to" => num_params + n_individuals, "bij" => identity)
num_params += n_individuals
function log_prior_beta0_random(sigma_beta0::Float32, beta0_random::AbstractArray{Float32})
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(0f0, sigma_beta0), n_individuals),
        beta0_random
    )
end

# Random time component
params_dict["sigma_beta_time"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
num_params += 1
prior_sigma_beta_time = truncated(Normal(0f0, 1f0), 0f0, Inf32)
log_prior_sigma_beta_time(sigma_beta_time::Float32) = Distributions.logpdf(prior_sigma_beta_time, sigma_beta_time)

params_dict["beta_time"] = OrderedDict("size" => (n_per_ind), "from" => num_params+1, "to" => num_params + n_per_ind, "bij" => identity)
num_params += n_per_ind
function log_prior_beta_time(sigma_beta_time::Float32, beta_time::AbstractArray{Float32})
    Distributions.logpdf(
        Turing.filldist(Distributions.Normal(0f0, sigma_beta_time), n_per_ind),
        beta_time
    )
end

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
likelihood(
    beta0_fixed=data_dict["beta0_fixed"],
    beta0_random=randn32(n_individuals),
    beta_fixed=data_dict["beta_fixed"],
    beta_time=data_dict["beta_time"],
    sigma_y=1f0,
    Xfix=data_dict["Xfix"]
)

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


function log_joint(theta_hat::AbstractArray{Float32})
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
        log_prior_beta0_random(sigma_beta0, beta0_random) +
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
dim_q = num_params * 2
half_dim_q = num_params

function getq(theta::AbstractArray{Float32})
    Distributions.MultivariateNormal(
        theta[1:half_dim_q],
        StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
    )

    # mu_vec = theta[1:half_dim_q]
    # sigma_vec = StatsFuns.softplus.(theta[(half_dim_q+1):(half_dim_q*2)])

    # Turing.DistributionsAD.arraydist([
    #     Normal(mu_vec[w], sigma_vec[w]) for w in range(1, num_params)
    # ])
end

q = getq(ones32(dim_q))
rand(q)


# >>>>>>>>>>>>>>>> Manual training loop <<<<<<<<<<<<<<<<<
num_steps = 5000
samples_per_step = 2

n_runs = 2
elbo_trace = zeros32(num_steps, n_runs)
theta_trace = zeros32(num_steps, dim_q)
posteriors = Dict()

for chain in range(1, n_runs)

    println("Chain number: ", chain)

    # Define objective
    variational_objective = Turing.Variational.ELBO()

    # Optimizer
    optimizer = DecayedADAGrad()

    # VI algorithm
    alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

    # --- Train loop ---
    converged = false
    step = 1
    theta = randn32(dim_q) * 0.5f0 .- 0.5f0

    prog = ProgressMeter.Progress(num_steps, 1)
    diff_results = DiffResults.GradientResult(theta)

    while (step ≤ num_steps) && !converged
        # 1. Compute gradient and objective value; results are stored in `diff_results`
        AdvancedVI.grad!(variational_objective, alg, getq, log_joint, theta, diff_results, samples_per_step)

        # vo(theta) = -variational_objective(alg, getq(theta), log_joint, samples_per_step)
        
        # loss, grads = Zygote.withgradient(theta) do params
        #     vo(params)
        # end
        # Flux.Optimise.update!(optimizer, theta, grads[1])
        # apply!(optimizer, theta, grads[1])
        # @. theta = theta - grads[1]

        # # 2. Extract gradient from `diff_result`
        gradient_step = DiffResults.gradient(diff_results)

        # # 3. Apply optimizer, e.g. multiplying by step-size
        diff_grad = apply!(optimizer, theta, gradient_step)

        # 4. Update parameters
        @. theta = theta - diff_grad

        # 5. Do whatever analysis you want - Store ELBO value
        elbo_trace[step, chain] = AdvancedVI.elbo(alg, getq(theta), log_joint, samples_per_step)
        # theta_trace[step, :] = deepcopy(theta)

        step += 1

        ProgressMeter.next!(prog)
    end

    q = getq(theta)
    posteriors["$(chain)"] = q

end

plot(elbo_trace, label="ELBO")
plot(elbo_trace[300:num_steps, :], label="ELBO")


MC_SAMPLES = 2000

posterior_samples = []
for chain = 1:n_runs
    push!(posterior_samples, rand(posteriors["$(chain)"], MC_SAMPLES))
end

plt = plot()
for pp in range(1, n_runs)
    plt = scatter!(posterior_summary(posterior_samples[pp], "gamma_logit", params_dict; fun=mean))
end
display(plt)

density_posterior(posterior_samples, "sigma_y", params_dict)

density_posterior(posterior_samples, "beta_fixed", params_dict; plot_label=false)

density_posterior(posterior_samples, "gamma_logit", params_dict; plot_label=false)

density_posterior(posterior_samples, "sigma_slab", params_dict; plot_label=false)

density_posterior(posterior_samples, "beta0_fixed", params_dict; plot_label=false)

density_posterior(posterior_samples, "sigma_beta0", params_dict; plot_label=true)

density_posterior(posterior_samples, "beta_time", params_dict; plot_label=true)

density_posterior(posterior_samples, "beta0_random", params_dict; plot_label=false)



inclusion_probs = zeros(p, n_runs)
for chain in range(1, n_runs)
    samples = rand(posteriors["$(chain)"], MC_SAMPLES)
    inclusion_probs[:, chain] = posterior_summary(samples, "gamma_logit", params_dict; fun=mean)[:,1]
end
median_inc_prob = median(inclusion_probs, dims=2)[:, 1]
mean_inc_prob = mean(inclusion_probs, dims=2)[:, 1]
sum(median_inc_prob .> 0.5)
sum(mean_inc_prob .> 0.5)

# FDR
classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    median_inc_prob .> 0.5
)
classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    mean_inc_prob .> 0.5
)

# ------ Mirror Statistic ------

# Retrieve the Posterior distributions of the betas
posterior_beta_mean = zeros(p, n_runs)
posterior_beta_sigma = zeros(p, n_runs)

for chain = 1:n_runs
    posterior_beta_mean[:, chain] = posteriors["$(chain)"].μ[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]]
    posterior_beta_sigma[:, chain] = sqrt.(posteriors["$(chain)"].Σ.diag[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]])
end

scatter(posterior_beta_mean)
scatter!(mean(posterior_beta_mean, dims=2))

weighted_posterior_beta_mean = mean(posterior_beta_mean, dims=2)[:, 1] .* mean_inc_prob
weighted_posterior_beta_mean = mean(posterior_beta_mean, dims=2)[:, 1] .* median_inc_prob

scatter(weighted_posterior_beta_mean)

# Variational distribution is a Gaussian
posterior_beta = MultivariateNormal(
    weighted_posterior_beta_mean,
    maximum(posterior_beta_sigma, dims=2)[:, 1]
)

fdr_target = 0.1

fdr_distribution = zeros(MC_SAMPLES)
tpr_distribution = zeros(MC_SAMPLES)
n_selected_distribution = zeros(MC_SAMPLES)
selection_matrix = zeros(p, MC_SAMPLES)

for nn = 1:MC_SAMPLES
    beta_1 = rand(posterior_beta)
    beta_2 = rand(posterior_beta)

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)
    n_selected_distribution[nn] = n_selected
    selection_matrix[:, nn] = (mirror_coeffs .> opt_t) * 1

    metrics = classification_metrics.wrapper_metrics(
        data_dict["beta_fixed"] .!= 0.,
        mirror_coeffs .> opt_t
    )
    
    fdr_distribution[nn] = metrics.fdr
    tpr_distribution[nn] = metrics.tpr
end

mean(fdr_distribution)
median(fdr_distribution)

mean(tpr_distribution)
median(tpr_distribution)

histogram(n_selected_distribution, label="FDR", normalize=:probability)

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

fdr_plot = histogram(fdr_distribution, label="FDR", normalize=:probability)
frequencies(fdr_distribution)
# savefig(fdr_plot, joinpath(abs_project_path, "results", "ms_analysis", "bayesian_fdr.pdf"))

tpr_plot = histogram(tpr_distribution, label="TPR", normalize=:probability)
frequencies(tpr_distribution)
