# Test on linear model
using OrderedCollections
using Distributions
using ComponentArrays, UnPack
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 200
n_time_points = 5

p = 10
prop_non_zero = 0.5
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5
beta_time=Float32.([0, 2, 0, 0, 0])
beta_pool=Float32.([-1., 1])

data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_time_points,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor, beta_pool=beta_pool,
    include_random_int=true, random_int_from_pool=false, random_intercept_sd=0.5,
    include_random_time=true, beta_time=beta_time,
    include_random_slope=false, random_seed=12
)

n_chains = 2
num_iter = 2000
MC_SAMPLES=2000

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    size=1,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)
update_parameters_dict(
    params_dict;
    name="sigma_beta0",
    size=n_individuals,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, 0.1f0), 0f0, Inf32),
        x
    ))
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    size=n_individuals,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta0"]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    size=p,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, 1f0), 0f0, Inf32),
        x
    ))
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    size=p,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

# beta time
update_parameters_dict(
    params_dict;
    name="sigma_beta_time",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)
update_parameters_dict(
    params_dict;
    name="beta_time",
    size=n_time_points,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::Float32) -> DistributionsLogPdf.log_normal(x, sigma=Float32.(ones(n_time_points)).*sigma),
    dependency=["sigma_beta_time"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)

theta_axes, _ = get_parameters_axes(params_dict)

# model predictions
model(theta_components) = Predictors.linear_random_intercept_model(
    theta_components;
    Xfix=data_dict["Xfix"]
)

# model
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)
partial_log_joint(Float32.(randn(params_dict["tot_params"])))

# VI distribution
vi_dist(z::AbstractArray) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])
z_init = Float32.(randn(params_dict["tot_params"]*2))

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z=z_init,
    n_iter=num_iter,
    n_chains=n_chains,
    samples_per_step=4
)

plot(res["elbo_trace"][200:end, :])

vi_posterior = average_posterior(
    res["posteriors"],
    Distributions.MultivariateNormal
)

samples_posterior = posterior_samples(
    vi_posterior=vi_posterior,
    params_dict=params_dict,
    n_samples=MC_SAMPLES
)
beta_samples = extract_parameter(
    prior="beta_fixed",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)

plt = density(beta_samples', label=false)

# Mirror Statistic
ms_dist = MirrorStatistic.posterior_ms_coefficients(
    vi_posterior=vi_posterior,
    prior="beta_fixed",
    params_dict=params_dict
)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

class_distributions = MirrorStatistic.fdr_distribution(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta_fixed"],
    fdr_target=0.1
)

plt_fdr = histogram(class_distributions.fdr_distribution, label=false, normalize=true)
xlabel!("FDR", labelfontsize=15)
vline!([mean(class_distributions.fdr_distribution)], color="red", label="Mean", linewidth=2)
plt_n = histogram(class_distributions.n_selected_distribution, label=false, normalize=true)
xlabel!("# Included Variables", labelfontsize=15)
vline!([mean(class_distributions.n_selected_distribution)], color="red", label="Mean", linewidth=2)
plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdr_n_hist.pdf"))
