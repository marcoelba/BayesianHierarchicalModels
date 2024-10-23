# Simulations linear model
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
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 200

p = 1000
prop_non_zero = 0.025
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

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
    name="beta0",
    dimension=(1, ),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(x, sigma=Float32.(ones(p)*1))
)
update_parameters_dict(
    params_dict;
    name="beta",
    dimension=(p, ),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dimension=(1, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 0.2f0), 0f0, Inf32),
        x
    )
)

theta_axes, _ = get_parameters_axes(params_dict)

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )
    
    # model predictions
    model(theta_components) = Predictors.linear_model(
        theta_components;
        X=data_dict["X"]
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
        prior="beta",
        params_dict=params_dict
    )

    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=data_dict["beta"],
        fdr_target=0.1
    )
    metrics_dict = Dict()

    # Posterior
    inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)

    metrics_posterior = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        selection
    )

    metrics_dict["fdr_range"] = metrics.fdr_range
    metrics_dict["tpr_range"] = metrics.tpr_range
    
    metrics_dict["metrics_mean"] = metrics.metrics_mean
    metrics_dict["metrics_median"] = metrics.metrics_median
    metrics_dict["metrics_posterior"] = metrics_posterior

    simulations_metrics[simu] = metrics_dict

end


label_files = "algo_HS_linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"


mean_fdr = []
mean_tpr = []
posterior_fdr = []
posterior_tpr = []

for simu = 1:n_simulations
    push!(mean_fdr, simulations_metrics[simu]["metrics_mean"].fdr)
    push!(mean_tpr, simulations_metrics[simu]["metrics_mean"].tpr)

    push!(posterior_fdr, simulations_metrics[simu]["metrics_posterior"].fdr)
    push!(posterior_tpr, simulations_metrics[simu]["metrics_posterior"].tpr)

end

all_metrics = hcat(mean_fdr, posterior_fdr, mean_tpr, posterior_tpr)
df = DataFrame(all_metrics, ["mean_fdr", "posterior_fdr", "mean_tpr", "posterior_tpr"])

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt_tpr = boxplot(mean_tpr, label=false)
boxplot!(posterior_tpr, label=false)
xticks!([1, 2], ["Mean", "Posterior"], tickfontsize=10)
title!("TPR", titlefontsize=20)

plt_fdr = boxplot(mean_fdr, label=false)
boxplot!(posterior_fdr, label=false)
xticks!([1, 2], ["Mean", "Posterior"], tickfontsize=10)
title!("FDR", titlefontsize=20)

plt = plot(plt_fdr, plt_tpr)

plt = boxplot(posterior_fdr, label=false)
boxplot!(posterior_tpr, label=false)
xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1))

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

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed+1
)

# model predictions
model(theta_components) = Predictors.linear_model(
    theta_components;
    X=data_dict["X"]
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
    n_iter=4000,
    n_chains=1,
    samples_per_step=2,
    use_noisy_grads=true,
    n_cycles=1
)

plot(res["elbo_trace"][num_iter:end])

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
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# ------ Mirror Statistic ------

ms_dist = MirrorStatistic.posterior_ms_coefficients(
    vi_posterior=vi_posterior,
    prior="beta",
    params_dict=params_dict
)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
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

metrics.metrics_relative

range_p = collect(range(1, p))
scatter(metrics.relative_inclusion_freq, markersize=3, label="Sorted relative inclusion freq")
hline!([metrics.min_inclusion_freq], label="Cutoff inclusion", linewidth=2)
sum(metrics.relative_inclusion_freq .> metrics.min_inclusion_freq)

excluded = metrics.relative_inclusion_freq .<= metrics.min_inclusion_freq
included = metrics.relative_inclusion_freq .> metrics.min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], metrics.relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], metrics.relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

# Posterior
inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)
sum(selection)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)

MirrorStatistic.posterior_ms_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)


metrics = MirrorStatistic.fdr_distribution(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=fdr_target
)

metrics.metrics_relative

excluded = metrics.relative_inclusion_freq .<= metrics.min_inclusion_freq
included = metrics.relative_inclusion_freq .> metrics.min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], metrics.relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], metrics.relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
