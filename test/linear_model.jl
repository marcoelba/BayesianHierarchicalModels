# linear model
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


n_individuals = 200

p = 1000
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5
# beta_time=Float32.([0, 2, 0, 0, 0])
# beta_pool=Float32.([-1., 1])

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=12
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
    name="tau_beta",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(x, sigma=Float32.(0.1))
)
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    size=p,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(x, sigma=Float32.(ones(p)*1))
)
update_parameters_dict(
    params_dict;
    name="beta",
    size=p,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    )
)

theta_axes, _ = get_parameters_axes(params_dict)

# model predictions
model(theta_components) = Predictors.linear_model(
    theta_components;
    X=data_dict["X"]
)
model(get_parameters_axes(params_dict)[2])

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

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z_dim=params_dict["tot_params"]*2,
    n_iter=num_iter,
    n_chains=n_chains,
    samples_per_step=2
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

sy_samples = extract_parameter(
    prior="sigma_y",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(sy_samples', label=false)

tau_samples = extract_parameter(
    prior="tau_beta",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(tau_samples', label=false)

s_samples = extract_parameter(
    prior="sigma_beta",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(s_samples', label=false)

# MS
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

boxplot(metrics.fdr_distribution, label="FDR")
boxplot!(metrics.tpr_distribution, label="TPR")

metrics.metrics_mean
metrics.metrics_median
scatter(mean(metrics.inclusion_matrix, dims=2))


MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    metrics.inclusion_matrix[:, 10] .!= 0.
)

sum_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
sum_per_mc = sum(metrics.inclusion_matrix, dims=1)

fdr = []
n_in = []
for mc = 1:MC_SAMPLES
    mm = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        metrics.inclusion_matrix[:, mc]
    )
    push!(fdr, mm.fdr)
    push!(n_in, sum(metrics.inclusion_matrix[:, mc]))
end

histogram(fdr)
mean(fdr)
histogram(n_in)
mean(n_in)
scatter(n_in, fdr)
fdr_n_in = hcat(fdr, n_in)
fdr_n_in

histogram(fdr_n_in[n_in .== 20, 1])
mean(fdr_n_in[n_in .== 20, 1])
selection = sum(metrics.inclusion_matrix[:, n_in .== 20], dims=2)[:, 1] .> 0
MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)


# older method
metrics = MirrorStatistic.fdr_distribution(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)

plt = fdr_n_hist(metrics)

plt = scatter_sel_matrix(metrics, p0=p0)
display(plt)

sort_indeces = sortperm(mean(metrics.selection_matrix, dims=2)[:,1], rev=true)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(mean(metrics.n_selected_distribution)))]] .= 1.
MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    sel_vec .> 0.
)

MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    metrics.selection_matrix[:, 20] .!= 0.
)
