# Simulations logistic model
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

p = 500
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

n_chains = 2
num_iter = 2000
MC_SAMPLES = 2000
fdr_target = 0.1
n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

label_files = "algo_HS_logistic_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dimension=(1,),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p,),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, Float32(0.1)), 0f0, Inf32),
        x
    ))
)
update_parameters_dict(
    params_dict;
    name="beta",
    dimension=(p,),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

theta_axes, _ = get_parameters_axes(params_dict)


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_logistic_model_data(;
        n_individuals, class_threshold=0.5f0,
        p, p1, p0, beta_pool=Float32.([-2., 2]), obs_noise_sd=0.5, corr_factor=0.5,
        random_seed=124 + simu, dtype=Float32
    )
    
    # model predictions
    model(theta_components) = Predictors.linear_predictor(
        theta_components;
        X=data_dict["X"]
    )
    
    # model
    partial_log_joint(theta) = log_joint(
        theta;
        params_dict=params_dict,
        theta_axes=theta_axes,
        model=model,
        log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
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

plt = violin([1], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], posterior_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))


# --------------------------------------------------------------------
# Single Run of Bayesian Model
n = n_individuals * 2
data_dict = generate_logistic_model_data(;
    n_individuals=n, class_threshold=0.5f0,
    p, p1, p0, beta_pool=Float32.([-2., 2]), obs_noise_sd=0.5, corr_factor=0.5,
    random_seed=124, dtype=Float32
)

n_train = Int(n / 2)
n_test = n - n_train
train_ids = sample(1:n, n_train, replace=false)
test_ids = setdiff(1:n, train_ids)

X_train = data_dict["X"][train_ids, :]
X_test = data_dict["X"][test_ids, :]
y_train = data_dict["y"][train_ids]
y_test = data_dict["y"][test_ids]


tau2_vec = [0.0001, 0.001, 0.01, 0.1, 0.5, 1., 5.]
loss = []

for tau2 in tau2_vec
    println("tau = : $(tau2)")

    params_dict = OrderedDict()

    # beta 0
    update_parameters_dict(
        params_dict;
        name="beta0",
        dimension=(1,),
        log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
    )

    # beta fixed
    update_parameters_dict(
        params_dict;
        name="sigma_beta",
        dimension=(p,),
        bij=VectorizedBijectors.softplus,
        log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
            truncated(Cauchy(0f0, Float32(tau2)), 0f0, Inf32),
            x
        ))
    )
    update_parameters_dict(
        params_dict;
        name="beta",
        dimension=(p,),
        log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
        dependency=["sigma_beta"]
    )

    theta_axes, _ = get_parameters_axes(params_dict)


    # model predictions
    model(theta_components) = Predictors.linear_predictor(
        theta_components;
        X=X_train
    )

    # model
    partial_log_joint(theta) = log_joint(
        theta;
        params_dict=params_dict,
        theta_axes=theta_axes,
        model=model,
        log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
        label=y_train
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

    # plot(res["elbo_trace"][num_iter:end, 1])

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
    # plt = density(beta_samples', label=false)
    # ylabel!("Density")
    # savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

    beta0_samples = extract_parameter(
        prior="beta0",
        params_dict=params_dict,
        samples_posterior=samples_posterior
    )

    # Prediction
    y_pred = zeros(n_individuals, MC_SAMPLES)
    for mc = 1:MC_SAMPLES
        y_pred[:, mc] = X_test * beta_samples[:, mc] .+ beta0_samples[mc]
    end
    loglik = sum(DistributionsLogPdf.log_bernoulli_from_logit(
        y_test,
        mean(y_pred, dims=2)[:, 1]
    ))
    push!(loss, loglik)
    println(loglik)

end


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

plt_n = histogram(metrics.n_inclusion_per_mc, bins=10, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([mean(metrics.n_inclusion_per_mc)], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
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
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_n_and_probs.pdf"))
