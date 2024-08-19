# Simulations linear model
using CSV
using DataFrames

abs_src_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_src_path, "vi_models", "linear_model.jl"))
include(joinpath(abs_src_path, "turing_experiments", "mixed_models_data_generation.jl"))
include(joinpath(abs_src_path, "turing_experiments", "mirror_statistic.jl"))
include(joinpath(abs_src_path, "turing_experiments", "plot_functions.jl"))
include(joinpath(abs_src_path, "utils", "classification_metrics.jl"))


n_individuals = 200
p = 1000
prop_non_zero = 0.025
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

n_runs = 3
num_steps = 5000
MC_SAMPLES = 5000
fdr_target = 0.1

n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    posteriors, elbo_trace, elbo_trace_batch, params_dict = linear_model(;
        data_dict=data_dict,
        prior_inc_prob=0.9f0,
        prior_sigma_spike=0.1f0,
        num_steps=num_steps,
        n_runs=n_runs
    )

    simulations_models[simu] = (posteriors, elbo_trace, params_dict)

    # ------ Mirror Statistic ------

    # Retrieve the Posterior distributions of the betas
    posterior_beta_mean = zeros(p, n_runs)
    posterior_beta_sigma = zeros(p, n_runs)

    for chain = 1:n_runs
        posterior_beta_mean[:, chain] = posteriors["$(chain)"].μ[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]]
        posterior_beta_sigma[:, chain] = sqrt.(posteriors["$(chain)"].Σ.diag[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]])
    end

    weighted_posterior_beta_mean = mean(posterior_beta_mean, dims=2)[:, 1]

    # Variational distribution is a Gaussian
    posterior_beta = MultivariateNormal(
        weighted_posterior_beta_mean,
        mean(posterior_beta_sigma, dims=2)[:, 1]
    )

    # density(rand(posterior_beta, MC_SAMPLES)')

    fdr_distribution = zeros(MC_SAMPLES)
    tpr_distribution = zeros(MC_SAMPLES)
    n_selected_distribution = zeros(MC_SAMPLES)
    selection_matrix = zeros(p, MC_SAMPLES)
    mirror_coefficients = zeros(p, MC_SAMPLES)

    for nn = 1:MC_SAMPLES
        beta_1 = rand(posterior_beta)
        beta_2 = rand(posterior_beta)

        mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)
        mirror_coefficients[:, nn] = mirror_coeffs

        opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
        n_selected = sum(mirror_coeffs .> opt_t)
        n_selected_distribution[nn] = n_selected
        selection_matrix[:, nn] = (mirror_coeffs .> opt_t) * 1

        metrics = classification_metrics.wrapper_metrics(
            data_dict["beta"] .!= 0.,
            mirror_coeffs .> opt_t
        )
        
        fdr_distribution[nn] = metrics.fdr
        tpr_distribution[nn] = metrics.tpr
    end

    metrics_dict = Dict()
    mean_selection_matrix = mean(selection_matrix, dims=2)[:, 1]
    sort_indeces = sortperm(mean_selection_matrix, rev=true)

    sel_vec = zeros(p)
    sel_vec[sort_indeces[1:Int(mode(n_selected_distribution))]] .= 1.
    metrics_dict["mode_selection_matrix"] = classification_metrics.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        sel_vec .> 0.
    )

    sel_vec = zeros(p)
    sel_vec[sort_indeces[1:Int(round(mean(n_selected_distribution)))]] .= 1.
    metrics_dict["mean_selection_matrix"] = classification_metrics.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        sel_vec .> 0.
    )

    metrics_dict["mode_fdr"] = mean(fdr_distribution[n_selected_distribution .== mode(n_selected_distribution)])
    metrics_dict["mode_tpr"] = mean(tpr_distribution[n_selected_distribution .== mode(n_selected_distribution)])


    range_included = Int.(maximum(n_selected_distribution))
    fdr_vec = []
    tpr_vec = []

    for jj = 1:range_included
        sel_vec = zeros(p)
        sel_vec[sort_indeces[1:jj]] .= 1.
        met = classification_metrics.wrapper_metrics(
            data_dict["beta"] .!= 0.,
            sel_vec .> 0.
        )
        push!(fdr_vec, met.fdr)
        push!(tpr_vec, met.tpr)
    end
    histogram(fdr_vec)

    simulations_metrics[simu] = metrics_dict

end

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
label_files = "linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"


total_fdr = []
total_tpr = []

modal_fdr = []
modal_tpr = []

mean_fdr = []
mean_tpr = []

for simu = 1:n_simulations
    push!(total_fdr, simulations_metrics[simu]["mode_fdr"])
    push!(total_tpr, simulations_metrics[simu]["mode_tpr"])

    push!(modal_fdr, simulations_metrics[simu]["mode_selection_matrix"].fdr)
    push!(modal_tpr, simulations_metrics[simu]["mode_selection_matrix"].tpr)

    push!(mean_fdr, simulations_metrics[simu]["mean_selection_matrix"].fdr)
    push!(mean_tpr, simulations_metrics[simu]["mean_selection_matrix"].tpr)

end

all_metrics = hcat(total_fdr, mean_fdr, modal_fdr, total_tpr, mean_tpr, modal_tpr)
df = DataFrame(all_metrics, ["total_fdr", "mean_fdr", "modal_fdr", "total_tpr", "mean_tpr", "modal_tpr"])

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt = boxplot(modal_tpr, label="Modal TPR")
boxplot!(mean_tpr, label="Mean TPR")
boxplot!(total_tpr, label="Total TPR")

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_boxplot_tpr.pdf"))

plt = boxplot(modal_fdr, label=false)
boxplot!(mean_fdr, label=false)
boxplot!(total_fdr, label=false)
xticks!([1, 2, 3], ["1", "2", "3"])
title!("FDR", titlefontsize=20)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_boxplot_fdr.pdf"))



plt = plot()
for chain = 1:n_runs
    plot!(elbo_trace[300:num_steps, chain], label="ELBO $(chain)")
end
display(plt)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_elbo.pdf"))


posterior_samples = []
for chain = 1:n_runs
    push!(posterior_samples, rand(posteriors["$(chain)"], MC_SAMPLES))
end

plt = plot()
for pp in range(1, n_runs)
    plt = scatter!(
        posterior_summary(posterior_samples[pp], "gamma_logit", params_dict; fun=mean),
        markersize=2.
        )
end
display(plt)


density_posterior(posterior_samples, "sigma_y", params_dict)

plt = density_posterior(posterior_samples[1:1], "beta_fixed", params_dict; plot_label=false)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_density_beta.pdf"))

density_posterior(posterior_samples, "sigma_spike", params_dict; plot_label=false)

density_posterior(posterior_samples, "beta0_fixed", params_dict; plot_label=false)

density_posterior(posterior_samples, "sigma_beta0", params_dict; plot_label=true)

inclusion_probs = zeros(p, n_runs)
for chain in range(1, n_runs)
    inclusion_probs[:, chain] = posterior_summary(posterior_samples[chain], "gamma_logit", params_dict; fun=mean)[:,1]
end

inclusion_probs = 1 .- inclusion_probs
inclusion_probs = (inclusion_probs .- minimum(inclusion_probs, dims=1)) ./ (maximum(inclusion_probs, dims=1) - minimum(inclusion_probs, dims=1))

plt = plot()
for pp in range(1, n_runs)
    plt = scatter!(inclusion_probs[:, pp], label="Run $(pp)", markersize=2.)
end
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_inclusion_probs.pdf"))


# ------ Mirror Statistic ------

# Retrieve the Posterior distributions of the betas
posterior_beta_mean = zeros(p, n_runs)
posterior_beta_sigma = zeros(p, n_runs)

for chain = 1:n_runs
    posterior_beta_mean[:, chain] = posteriors["$(chain)"].μ[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]]
    posterior_beta_sigma[:, chain] = sqrt.(posteriors["$(chain)"].Σ.diag[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]])
end

weighted_posterior_beta_mean = mean(posterior_beta_mean, dims=2)[:, 1]

# Variational distribution is a Gaussian
posterior_beta = MultivariateNormal(
    weighted_posterior_beta_mean,
    mean(posterior_beta_sigma, dims=2)[:, 1]
)


plt = density(rand(posterior_beta, MC_SAMPLES)', label=false)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_average_beta.pdf"))


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
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )
    
    fdr_distribution[nn] = metrics.fdr
    tpr_distribution[nn] = metrics.tpr
end

mean(fdr_distribution)
median(fdr_distribution)

mean(tpr_distribution)
median(tpr_distribution)

for mc = 1:MC_SAMPLES
    selection_matrix[:, mc] .* fdr_distribution[mc]
end

plt_hist = histogram(n_selected_distribution, label="Covs included", normalize=:probability)
plt_fdr = histogram(fdr_distribution, label="FDR", normalize=:probability)

plt = plot(plt_hist, plt_fdr, layout = (1, 2))
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdr_dist_and_hist.pdf"))

plt = scatter(mean(selection_matrix, dims=2), label="Mirror Coefficients inclusion freq")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_selection_matrix.pdf"))



# ----------- Data Splitting -----------
include(joinpath(abs_src_path, "utils", "variable_selection_plus_inference.jl"))
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

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = classification_metrics.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(ds_fdr, metrics.fdr)
    push!(ds_tpr, metrics.tpr)

end

