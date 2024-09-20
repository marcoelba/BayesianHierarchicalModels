# Test on linear model
using OrderedCollections
using Distributions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 100

p = 200
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

data_dict = generate_logistic_model_data(;
    n_individuals, class_threshold=0.5f0,
    p, p1, p0, beta_pool=Float32.([-2., 2]), obs_noise_sd=0.5, corr_factor=0.5,
    random_seed=124, dtype=Float32
)

n_chains = 2
num_iter = 2000
MC_SAMPLES=2000

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    size=1,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
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
    name="beta",
    size=p,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

theta_axes, _ = get_parameters_axes(params_dict)

# model predictions
model(theta_components) = Predictors.linear_predictor(
    theta_components;
    X=data_dict["X"]
)
model(get_parameters_axes(params_dict)[2])

DistributionsLogPdf.log_bernoulli_from_logit(data_dict["y"], model(get_parameters_axes(params_dict)[2])...)

# model
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
    label=data_dict["y"]
)
partial_log_joint(Float32.(randn(params_dict["tot_params"])))

# VI distribution
vi_dist(z::AbstractArray) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z_dim=params_dict["tot_params"]*2,
    n_iter=num_iter,
    n_chains=n_chains,
    samples_per_step=4
)

plot(res["elbo_trace"][100:end, :])

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
    prior="beta",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)

plt = density(beta_samples', label=false)

# Mirror Statistic
ms_dist = MirrorStatistic.posterior_ms_coefficients(
    vi_posterior=vi_posterior,
    prior="beta",
    params_dict=params_dict
)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)

# distribution
boxplot(metrics.fdr_distribution, label="FDR")
boxplot!(metrics.tpr_distribution, label="TPR")

plt = fdr_n_hist(metrics)

# range
boxplot(metrics.fdr_range, label="FDR")
boxplot!(metrics.tpr_range, label="TPR")

metrics.metrics_mean
metrics.metrics_median

plt = scatter_sel_matrix(metrics.inclusion_matrix, p0=p0)
