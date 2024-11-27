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
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))


n_individuals = 200
n_time_points = 5

p = 5000
prop_non_zero = 0.01
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5
beta_time=Float32.([0, 2, 1, 0, 0])
beta_pool=Float32.([-1., 1])

n_chains = 1
num_iter = 4000
MC_SAMPLES = 2000
fdr_target = 0.1

n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

label_files = "algo_HS_rand_int_n$(n_individuals)_T$(n_time_points)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"


params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    dimension=(1,),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)
update_parameters_dict(
    params_dict;
    name="sigma_beta0",
    dimension=(n_individuals,),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(n_individuals) * 0.1)
    )
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    dimension=(n_individuals,),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma
    ),
    dependency=["sigma_beta0"]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p,),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(p)*1)
    )
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dimension=(p,),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma
    ),
    dependency=["sigma_beta"]
)

# beta time
update_parameters_dict(
    params_dict;
    name="sigma_beta_time",
    dimension=(1,),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32(p1/n_individuals)
    )
)
update_parameters_dict(
    params_dict;
    name="beta_time",
    dimension=(n_time_points,),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::Float32) -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_time_points)) .* sigma
    ),
    dependency=["sigma_beta_time"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dimension=(1,),
    bij=VectorizedBijectors.softplus,
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
        samples_per_step=2,
        sd_init=0.5f0,
        use_noisy_grads=true,
        n_cycles=1
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

    # Posterior
    inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)

    metrics_posterior = MirrorStatistic.wrapper_metrics(
        data_dict["beta_fixed"] .!= 0.,
        selection
    )
    metrics_dict["metrics_posterior"] = metrics_posterior

    metrics_dict["fdr_range"] = metrics.fdr_range
    metrics_dict["tpr_range"] = metrics.tpr_range
    
    metrics_dict["metrics_mean"] = metrics.metrics_mean
    metrics_dict["metrics_median"] = metrics.metrics_median
    
    simulations_metrics[simu] = metrics_dict

end

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

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))


plt = violin([1], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], posterior_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

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
    n_iter=3000,
    n_chains=n_chains,
    samples_per_step=2,
    sd_init=0.5f0,
    use_noisy_grads=true,
    n_cycles=1
)

plot(res["elbo_trace"])
plot(res["elbo_trace"][2000:end])

vi_posterior = average_posterior(
    res["posteriors"],
    Distributions.MultivariateNormal
)
samples_posterior = posterior_samples(
    vi_posterior=vi_posterior,
    params_dict=params_dict,
    n_samples=500
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
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta0.pdf"))

beta_time_samples = extract_parameter(
    prior="beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_time_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta_time.pdf"))

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

# Posterior
plt_n = histogram(metrics.n_inclusion_per_mc, bins=10, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([mean(metrics.n_inclusion_per_mc)], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "simulations", "$(label_files)_n_vars_included.pdf"))

n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

MirrorStatistic.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    selection
)

plt_probs = scatter(
    findall((1 .- mean_inclusion_per_coef) .> c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .> c_opt)],
    label=false, markersize=3,
)
scatter!(
    findall((1 .- mean_inclusion_per_coef) .<= c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .<= c_opt)],
    label="Selected", markersize=5,
)
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "simulations", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_n_and_probs.pdf"))


# Look in to the details!!
fp_prob = 1. .- mean_inclusion_per_coef
c_opt = 0.

for c in sort(mean_inclusion_per_coef, rev=false)
    lower_than_c = mean_inclusion_per_coef .> c
    if (sum(fp_prob .* lower_than_c) / sum(lower_than_c)) <= fdr_target
        c_opt = c
        break
    end
end
sum(mean_inclusion_per_coef .> c_opt)


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
