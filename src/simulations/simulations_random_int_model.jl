# Simulations linear model
using CSV
using DataFrames

abs_src_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_src_path, "vi_models", "random_int_model.jl"))
include(joinpath(abs_src_path, "utils", "mixed_models_data_generation.jl"))
include(joinpath(abs_src_path, "utils", "mirror_statistic.jl"))
include(joinpath(abs_src_path, "utils", "plot_functions.jl"))
include(joinpath(abs_src_path, "utils", "classification_metrics.jl"))


n_individuals = 200
n_per_ind = 5

p = 500
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

n_runs = 3
num_steps = 3000
MC_SAMPLES = 3000
fdr_target = 0.1

n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_mixed_model_data(;
        n_individuals=n_individuals, n_time_points=n_per_ind,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        include_random_int=true, random_intercept_sd=0.5,
        include_random_time=true, random_time_sd=0.5,
        include_random_slope=false, random_seed=random_seed + simu
    )

    posteriors, elbo_trace, params_dict = random_intercept_model(
        data_dict=data_dict,
        num_steps=num_steps,
        n_runs=n_runs,
        samples_per_step=4,
        cauchy_scale=1f0
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

    metrics_dict = Dict()
    mean_selection_matrix = mean(selection_matrix, dims=2)[:, 1]
    sort_indeces = sortperm(mean_selection_matrix, rev=true)

    sel_vec = zeros(p)
    sel_vec[sort_indeces[1:Int(mode(n_selected_distribution))]] .= 1.
    metrics_dict["mode_selection_matrix"] = classification_metrics.wrapper_metrics(
        data_dict["beta_fixed"] .!= 0.,
        sel_vec .> 0.
    )

    sel_vec = zeros(p)
    sel_vec[sort_indeces[1:Int(round(mean(n_selected_distribution)))]] .= 1.
    metrics_dict["mean_selection_matrix"] = classification_metrics.wrapper_metrics(
        data_dict["beta_fixed"] .!= 0.,
        sel_vec .> 0.
    )
    simulations_metrics[simu] = metrics_dict

    selection = mean_selection_matrix .> 0.5
    extra = Int(round(sum(selection) * (0.1 / 0.9)))
    selection[
        sort_indeces[sum(selection) + 1:sum(selection) + extra]
    ] .= 1
    metrics_dict["extra_selection_matrix"] = classification_metrics.wrapper_metrics(
        data_dict["beta_fixed"] .!= 0.,
        selection .> 0
    )

    simulations_metrics[simu] = metrics_dict

end

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
label_files = "algo_HS_rand_int_n$(n_individuals)_T$(n_per_ind)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"


modal_fdr = []
modal_tpr = []

mean_fdr = []
mean_tpr = []

extra_fdr = []
extra_tpr = []

for simu = 1:n_simulations
    push!(modal_fdr, simulations_metrics[simu]["mode_selection_matrix"].fdr)
    push!(modal_tpr, simulations_metrics[simu]["mode_selection_matrix"].tpr)

    push!(mean_fdr, simulations_metrics[simu]["mean_selection_matrix"].fdr)
    push!(mean_tpr, simulations_metrics[simu]["mean_selection_matrix"].tpr)

    push!(extra_fdr, simulations_metrics[simu]["extra_selection_matrix"].fdr)
    push!(extra_tpr, simulations_metrics[simu]["extra_selection_matrix"].tpr)

end

all_metrics = hcat(mean_fdr, modal_fdr, extra_fdr, mean_tpr, modal_tpr, extra_tpr)
df = DataFrame(
    all_metrics,
    ["mean_fdr", "modal_fdr", "extra_fdr", "mean_tpr", "modal_tpr", "extra_tpr"]
)

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt_tpr = boxplot(mean_tpr, label=false)
boxplot!(modal_tpr, label=false)
boxplot!(extra_tpr, label=false)
xticks!([1, 2, 3], ["Mean", "Mode", "Extra"], tickfontsize=10)
title!("TPR", titlefontsize=20)

plt_fdr = boxplot(mean_fdr, label=false)
boxplot!(modal_fdr, label=false)
boxplot!(extra_fdr, label=false)
xticks!([1, 2, 3], ["Mean", "Mode", "Extra"], tickfontsize=10)
title!("FDR", titlefontsize=20)

plt = plot(plt_fdr, plt_tpr)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))


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
    n_individuals=n_individuals, n_time_points=n_per_ind,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    include_random_int=true, random_intercept_sd=0.5,
    include_random_time=true, random_time_sd=0.5,
    include_random_slope=false, random_seed=23
)

posteriors, elbo_trace, params_dict = random_intercept_model(
    data_dict=data_dict,
    num_steps=num_steps,
    n_runs=n_runs,
    samples_per_step=4,
    cauchy_scale=1f0
)

# ------ Mirror Statistic ------

# Retrieve the Posterior distributions of the betas
posterior_beta_mean = zeros(p, n_runs)
posterior_beta_sigma = zeros(p, n_runs)

for chain = 1:n_runs
    posterior_beta_mean[:, chain] = posteriors["$(chain)"].μ[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]]
    posterior_beta_sigma[:, chain] = sqrt.(posteriors["$(chain)"].Σ.diag[params_dict["beta_fixed"]["from"]:params_dict["beta_fixed"]["to"]])
end

posterior_beta_mean = mean(posterior_beta_mean, dims=2)[:, 1]
posterior_beta_sigma = mean(posterior_beta_sigma, dims=2)[:, 1]

# Variational distribution is a Gaussian
posterior_beta = MultivariateNormal(
    weighted_posterior_beta_mean,
    mean(posterior_beta_sigma, dims=2)[:, 1]
)

plt = density(rand(posterior_beta, MC_SAMPLES)', label=false)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

scatter(weighted_posterior_beta_mean)
scatter(mean(posterior_beta_sigma, dims=2)[:, 1])

fdr_distribution = zeros(MC_SAMPLES)
tpr_distribution = zeros(MC_SAMPLES)
n_selected_distribution = zeros(MC_SAMPLES)
selection_matrix = zeros(p, MC_SAMPLES)
ms = zeros(p, MC_SAMPLES)
optimal_t = zeros(MC_SAMPLES)

fdr_target = 0.1

for nn = 1:MC_SAMPLES
    beta_1 = rand(posterior_beta)
    beta_2 = rand(posterior_beta)

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)
    scatter(mirror_coeffs)
    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    hline!([opt_t])
    n_selected = sum(mirror_coeffs .> opt_t)

    ms[:, nn] = mirror_coeffs

    optimal_t[nn] = opt_t
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

plt_fdr = histogram(fdr_distribution, label=false, normalize=true)
xlabel!("FDR", labelfontsize=15)
plt_n = histogram(n_selected_distribution, label=false, normalize=true)
xlabel!("# Included Variables", labelfontsize=15)
plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdr_n_hist.pdf"))

density(ms', label=false)
vline!([mean(optimal_t)], label=false)

mean_selection_matrix = mean(selection_matrix, dims=2)[:, 1]
scatter(mean_selection_matrix, label=false)
vline!([p0+1], label=false)

sort_indeces = sortperm(mean_selection_matrix, rev=true)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(mode(n_selected_distribution))]] .= 1.
classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    sel_vec .> 0.
)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(mean(n_selected_distribution)))]] .= 1.
classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    sel_vec .> 0.
)

selection = mean_selection_matrix .> 0.5
sum(selection)
extra = Int(round(sum(selection) * (0.1 / 0.9)))
selection[
    sort_indeces[sum(selection) + 1:sum(selection) + extra]
] .= 1

classification_metrics.wrapper_metrics(
    data_dict["beta_fixed"] .!= 0.,
    selection .> 0
)

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

plt = scatter(fdr_vec, label="FDR")
scatter!(tpr_vec, label="TPR")
xlabel!("# Included Variables", labelfontsize=15)
# title!("FDR", titlefontsize=20)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_scatter.pdf"))


beta_1 = vcat(rand(posterior_beta, 200)...)
beta_2 = vcat(rand(posterior_beta, 200)...)

N1 = beta_1 .+ beta_2
N2 = beta_1 .- beta_2
A1 = abs.(N1)
A2 = abs.(N2)
mean(A2)
std(A2) * sqrt(2) / sqrt(pi)

scatter(N1, label=false)
scatter(N2, label=false)

scatter(A1, label=false)
scatter(A2, label=false)

mirror_coeffs = abs.(beta_1 .+ beta_2) .- abs.(beta_1 .- beta_2)
mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)
scatter(mirror_coeffs, label=false)
opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=0.1)
hline!([opt_t])
sum(mirror_coeffs .> opt_t)

# Analysis of distribution
pp = p
Apos = truncated.(Normal.(1., 0.1), 0, Inf)
Aneg = truncated.(Normal.(0., 0.1), 0, Inf)

Rpos = rand(Apos, 10000)
Rneg = rand(Aneg, 10000)

MS = Rpos .- Rneg

density(Rpos)
density!(Rneg)
density!(MS)

var(Apos) + var(Aneg)
var(MS)

mean(Apos) - mean(Aneg)
mean(MS)

# mean of Aneg = sigma * sqrt(2)/sqrt(pi), Half-Normal
function mean_folded_normal(mu, sigma)
    sigma * sqrt(2/pi) * exp(-0.5 *(mu/sigma)^2) + mu * (1 - 2*cdf(Normal(), -(mu/sigma)))
end

function var_folded_normal(mu, sigma)
    mu^2 + sigma^2 - mean_folded_normal(mu, sigma)^2
end

mean_ms = mean_folded_normal(1, 0.1) - mean_folded_normal(0., 0.1)
var_ms = var_folded_normal(1, 0.1) + var_folded_normal(0., 0.1)
dist_ms = Normal(mean_ms, sqrt(var_ms))


density!(rand(Normal(mean(MS), std(MS)), 10000), color="red", label=false)
density!(rand(dist_ms, 10000), color="black", label=false)

quantile(MS, 0.1)
quantile(dist_ms, 0.1)

MirrorStatistic.get_t(MS; fdr_q=0.1)


mean_vec = mean_folded_normal.(posterior_beta_mean, posterior_beta_sigma) .- 
    mean_folded_normal.(0., posterior_beta_sigma)
var_vec = var_folded_normal.(posterior_beta_mean, posterior_beta_sigma) .+ 
    var_folded_normal.(0., posterior_beta_sigma)

ms_dist_vec = arraydist([
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
])
scatter(rand(ms_dist_vec, 100), label=false)
