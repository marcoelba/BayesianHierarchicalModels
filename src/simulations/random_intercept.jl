# Simulations linear random Intercept model
using CSV
using DataFrames

using OrderedCollections
using Distributions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 200
n_time_points = 5

p = 1000
prop_non_zero = 0.025
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5
beta_time=Float32.([0, 2, 0, 0, 0])
beta_pool=Float32.([-1., 1])

n_chains = 2
num_iter = 2000
MC_SAMPLES=2000
fdr_target = 0.1

n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

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
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(n_individuals) * 0.1)
    )
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    size=n_individuals,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma
    ),
    dependency=["sigma_beta0"]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    size=p,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(x, sigma=Float32.(ones(p)*1))
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
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(x, sigma=1f0)
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
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    )
)

theta_axes, _ = get_parameters_axes(params_dict)

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_mixed_model_data(;
        n_individuals=n_individuals, n_time_points=n_time_points,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor, beta_pool=beta_pool,
        include_random_int=true, random_int_from_pool=false, random_intercept_sd=0.5,
        include_random_time=true, beta_time=beta_time,
        include_random_slope=false, random_seed=random_seed+simu
    )
    
    # model predictions
    model(theta_components) = Predictors.linear_random_intercept_model(
        theta_components;
        Xfix=data_dict["Xfix"]
    )

    # model log joint
    partial_log_joint(theta) = log_joint(
        theta;
        params_dict=params_dict,
        theta_axes=theta_axes,
        model=model,
        log_likelihood=DistributionsLogPdf.log_normal,
        label=data_dict["y"]
    )

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

    vi_posterior = average_posterior(
        res["posteriors"],
        Distributions.MultivariateNormal
    )
    
    simulations_models[simu] = (vi_posterior, res["elbo_trace"], params_dict)

    # ------ Mirror Statistic ------

    ms_dist = MirrorStatistic.posterior_ms_coefficients(
        vi_posterior=vi_posterior,
        prior="beta_fixed",
        params_dict=params_dict
    )
    
    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=data_dict["beta_fixed"],
        fdr_target=fdr_target
    )
    metrics_dict = Dict()

    metrics_dict["fdr_range"] = metrics.fdr_range
    metrics_dict["tpr_range"] = metrics.tpr_range
    
    metrics_dict["metrics_mean"] = metrics.metrics_mean
    metrics_dict["metrics_median"] = metrics.metrics_median
    
    simulations_metrics[simu] = metrics_dict

end


label_files = "algo_HS_rand_int_n$(n_individuals)_T$(n_time_points)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

median_fdr = []
median_tpr = []

mean_fdr = []
mean_tpr = []

for simu = 1:n_simulations
    push!(median_fdr, simulations_metrics[simu]["metrics_median"].fdr)
    push!(median_tpr, simulations_metrics[simu]["metrics_median"].tpr)

    push!(mean_fdr, simulations_metrics[simu]["metrics_mean"].fdr)
    push!(mean_tpr, simulations_metrics[simu]["metrics_mean"].tpr)

end

all_metrics = hcat(mean_fdr, median_fdr, mean_tpr, median_tpr)
df = DataFrame(all_metrics, ["mean_fdr", "median_fdr", "mean_tpr", "median_tpr"])

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt_tpr = boxplot(mean_tpr, label=false)
boxplot!(median_tpr, label=false)
xticks!([1, 2], ["Mean", "Median"], tickfontsize=10)
title!("TPR", titlefontsize=20)

plt_fdr = boxplot(mean_fdr, label=false)
boxplot!(median_fdr, label=false)
xticks!([1, 2], ["Mean", "Median"], tickfontsize=10)
title!("FDR", titlefontsize=20)

plt = plot(plt_fdr, plt_tpr)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))


# ----------- Data Splitting -----------
include(joinpath(abs_project_path, "src", "utils", "variable_selection_plus_inference.jl"))
ds_fdr = []
ds_tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    res = variable_selection_plus_inference.lasso_plus_ols(;
        X1=Float64.(data_dict["X"][1:Int(n_individuals/2), :]),
        X2=Float64.(data_dict["X"][Int(n_individuals/2)+1:end, :]),
        y1=Float64.(data_dict["y"][1:Int(n_individuals/2)]),
        y2=Float64.(data_dict["y"][Int(n_individuals/2)+1:end]),
        add_intercept=true,
        alpha_lasso=1.
    )

    beta_1 = res[1]
    beta_2 = res[2]

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(ds_fdr, metrics.fdr)
    push!(ds_tpr, metrics.tpr)

end


plt_tpr = boxplot(ds_tpr, label=false)
xaxis!([])
title!("TPR", titlefontsize=20)

plt_fdr = boxplot(ds_fdr, label=false)
xaxis!([])
title!("FDR", titlefontsize=20)

plt = plot(plt_fdr, plt_tpr)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_DS.pdf"))


# --------------------------------------------------------------------
# Single Run of Bayesian Model

data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_time_points,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor, beta_pool=beta_pool,
    include_random_int=true, random_int_from_pool=true, random_intercept_sd=0.5,
    include_random_time=true, beta_time=beta_time,
    include_random_slope=false, random_seed=random_seed
)

# model predictions
model(theta_components) = Predictors.linear_random_intercept_model(
    theta_components;
    Xfix=data_dict["Xfix"]
)

# model log joint
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)

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
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))


beta0_samples = extract_parameter(
    prior="beta0_random",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta0_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

beta_time_samples = extract_parameter(
    prior="beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_time_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# ------ Mirror Statistic ------

ms_dist = MirrorStatistic.posterior_ms_coefficients(
    vi_posterior=vi_posterior,
    prior="beta_fixed",
    params_dict=params_dict
)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta_fixed"],
    fdr_target=fdr_target
)

# distribution
boxplot(metrics.fdr_distribution, label="FDR")
boxplot!(metrics.tpr_distribution, label="TPR")

plt = fdr_n_hist(metrics)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdr_n_hist.pdf"))

# range
boxplot(metrics.fdr_range, label="FDR")
boxplot!(metrics.tpr_range, label="TPR")

metrics.metrics_mean
metrics.metrics_median

plt = scatter_sel_matrix(metrics.inclusion_matrix, p0=p0)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_inclusion_probs.pdf"))
