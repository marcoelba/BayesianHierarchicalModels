# Simulations linear random Intercept model
using CSV
using DataFrames

using Optimisers
using Distributions
using DistributionsAD
using LogExpFunctions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "my_optimisers.jl"))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "training_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "bijectors_extension.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))

include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))


n_individuals = 100
n_repeated_measures = 5
p = 500
p1 = Int(p * 0.05)
p0 = p - p1
corr_factor = 0.5

# label to save results
label_files = "algo_random_int_n$(n_individuals)_M$(n_repeated_measures)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"


params_dict = OrderedDict()

update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
    init_z=[0., 0.1]
)

# Random intercept beta0 - HS
update_parameters_dict(
    params_dict;
    name="sigma_beta0",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=1.
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2) * 0.05
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    dim_theta=(n_individuals, ),
    logpdf_prior=(x::AbstractArray, mu::Real, sigma::Real) -> DistributionsLogPdf.log_normal(
        x, mu=ones(n_individuals) * mu, sigma=ones(n_individuals) * sigma
    ),
    dim_z=n_individuals * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(n_individuals * 2)*0.05,
    dependency=["beta0_fixed", "sigma_beta0"],
    random_variable=true
)

# Beta
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2) .* 0.1,
    random_variable=true
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 0.5, 0.5),
    dim_z=p * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=randn(p * 2) * 0.01,
    random_variable=true
)

update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma_beta::AbstractArray, tau::Real) -> DistributionsLogPdf.log_normal(
        x,
        mu=zeros(p),
        sigma=tau .* sigma_beta
    ),
    dim_z=p * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(p * 2) * 0.01,
    dependency=["sigma_beta", "tau_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> Distributions.logpdf(
        truncated(Normal(0., 5.), 0., Inf),
        x
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2) * 0.05
)

prior_position = params_dict["tuple_prior_position"]

fdr_target = 0.1
MC_SAMPLES = 1000
n_iter = 5000
n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_multiple_measurements_data(
        n_individuals=n_individuals,
        n_repeated_measures=n_repeated_measures,
        p=p, p1=p1, p0=p0,
        beta_pool=Float32.([-1., -2., 1, 2]),
        sd_noise_beta_reps=0.,
        obs_noise_sd=0.5,
        corr_factor=corr_factor,
        beta0_fixed=0.,
        random_int_from_pool=false,
        random_intercept_sd=2.,
        random_seed=random_seed + simu,
        dtype=Float32
    )
        
    # model predictions
    model(theta, rep_index; X) = Predictors.random_intercept_model(
        theta,
        rep_index;
        X
    )
    
    # Training
    z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
    optimiser = MyOptimisers.DecayedADAGrad()
    # optimiser = Optimisers.RMSProp(0.01)
    
    res = hybrid_training_loop(
        z=z,
        y=data_dict["y"],
        X=data_dict["Xfix"],
        params_dict=params_dict,
        model=model,
        log_likelihood=DistributionsLogPdf.log_normal,
        log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
        n_iter=n_iter,
        optimiser=optimiser,
        save_all=false,
        use_noisy_grads=false,
        elbo_samples=1,
        n_repeated_measures=n_repeated_measures
    )

    best_z = res["best_iter_dict"]["best_z"]

    q = VariationalDistributions.get_variational_dist(
        best_z,
        params_dict["vi_family_array"],
        params_dict["ranges_z"]
    )

    simulations_models[simu] = (q, res["loss_dict"]["loss"], params_dict)

    # ------ Mirror Statistic ------
    metrics_dict = Dict()

    ms_dist = MirrorStatistic.posterior_ms_coefficients(
        q[prior_position[:beta_fixed]].dist
    )
    
    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=vcat(data_dict["beta_reg"]...),
        fdr_target=0.1
    )
    #
    n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
    mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]
    
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )
    
    metrics_posterior = MirrorStatistic.wrapper_metrics(
        vcat(data_dict["beta_reg"]...) .!= 0.,
        selection
    )
    metrics_dict["metrics_posterior"] = metrics_posterior
    
    # Monte Carlo loop
    mc_samples = 2000
    ms_samples = Int(mc_samples / 2)
    # mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
    beta = rand(q[prior_position[:beta_fixed]], mc_samples)
    
    ms_coeffs = MirrorStatistic.mirror_statistic(
        beta[:, 1:ms_samples],
        beta[:, ms_samples+1:mc_samples]
    )
    opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
    inclusion_matrix = ms_coeffs .> opt_t
    mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]
    
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )
    
    metrics_mc = MirrorStatistic.wrapper_metrics(
        vcat(data_dict["beta_reg"]...) .!= 0.,
        selection
    )

    metrics_dict["metrics_mc"] = metrics_mc
    
    simulations_metrics[simu] = metrics_dict

end

mc_fdr = []
mc_tpr = []
posterior_fdr = []
posterior_tpr = []

for simu = 1:n_simulations
    push!(mc_fdr, simulations_metrics[simu]["metrics_mc"].fdr)
    push!(mc_tpr, simulations_metrics[simu]["metrics_mc"].tpr)

    push!(posterior_fdr, simulations_metrics[simu]["metrics_posterior"].fdr)
    push!(posterior_tpr, simulations_metrics[simu]["metrics_posterior"].tpr)
end

all_metrics = hcat(mc_fdr, posterior_fdr, mc_tpr, posterior_tpr)
df = DataFrame(all_metrics, ["mc_fdr", "posterior_fdr", "mc_tpr", "posterior_tpr"])

mean(posterior_fdr)
mean(mc_fdr)

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)

# Plot FDR-TPR
plt = violin([1], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], posterior_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_formula.pdf"))


plt = violin([1], mc_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], mc_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], mc_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], mc_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))



# --------------------------------------------------------------------
# Single Run of Bayesian Model

data_dict = generate_multiple_measurements_data(
    n_individuals=n_individuals,
    n_repeated_measures=n_repeated_measures,
    p=p, p1=p1, p0=p0,
    beta_pool=Float32.([-1., -2., 1, 2]),
    sd_noise_beta_reps=0.,
    obs_noise_sd=0.5,
    corr_factor=corr_factor,
    beta0_fixed=0.,
    random_int_from_pool=false,
    random_intercept_sd=2.,
    random_seed=12,
    dtype=Float32
)


plot(data_dict["y"][1, :], label=false)
plot!(data_dict["y"][2, :], label=false)
plot!(data_dict["y"][3, :], label=false)
plot!(data_dict["y"][4, :], label=false)


# model predictions
model(theta, rep_index; X) = Predictors.random_intercept_model(
    theta,
    rep_index;
    X
)

# get ONE VI distribution
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
q_dist_array = VariationalDistributions.get_variational_dist(
    z, params_dict["vi_family_array"], params_dict["ranges_z"]
)

# sample
VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=false)

# sample with log-jacobian
theta, jac = VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=params_dict["random_weights"])

# Entropy
for dist in q_dist_array
    println(VariationalDistributions.entropy(dist))
end

theta_axes = get_parameters_axes(params_dict)
theta = ComponentArray(theta, theta_axes)

model(theta, 1; X=data_dict["Xfix"])
model(theta, 3; X=data_dict["Xfix"])

compute_logpdf_prior(theta; params_dict=params_dict)

# Training
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MyOptimisers.DecayedADAGrad()
# optimiser = Optimisers.RMSProp(0.01)

res = hybrid_training_loop(
    z=z,
    y=data_dict["y"],
    X=data_dict["Xfix"],
    params_dict=params_dict,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=5000,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=1,
    n_repeated_measures=n_repeated_measures
)

plot(res["loss_dict"]["loss"])
plot(res["loss_dict"]["loss"][300:end])
plot(res["loss_dict"]["loss"][4000:end])

res["best_iter_dict"]
res["loss_dict"]["loss"]

best_z = res["best_iter_dict"]["best_z"]
best_z = res["best_iter_dict"]["final_z"]


q = VariationalDistributions.get_variational_dist(
    best_z,
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)

# beta probs
beta_probs = rand(q[prior_position[:sigma_beta]], 1000)'
scatter(mean(beta_probs, dims=1)', label=false)

# beta tau
beta_tau = rand(q[prior_position[:tau_beta]], 1000)
scatter(mean(beta_tau, dims=1)', label=false)

# beta
beta = rand(q[prior_position[:beta_fixed]], 1000)'
density(beta, label=false)
plt = density(beta, label=false)

ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))


# beta0 random int
beta0_random_samples = rand(q[prior_position[:beta0_random]], 1000)'
plt = density(beta0_random_samples, label=false)

# sigma y
sigma_y_samples = rand(q[prior_position[:sigma_y]], 1000)
plt = density(sigma_y_samples, label=true)


# Mirror Statistic
ms_dist = MirrorStatistic.posterior_ms_coefficients(
    q[prior_position[:beta_fixed]].dist
)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=vcat(data_dict["beta_reg"]...),
    fdr_target=0.1
)
#
n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

MirrorStatistic.wrapper_metrics(
    vcat(data_dict["beta_reg"]...) .!= 0.,
    selection
)

# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)
mean_sigma = rand(q[prior_position[:sigma_beta]], mc_samples)
tau = rand(q[prior_position[:tau_beta]], mc_samples)'

beta = rand(q[prior_position[:beta_fixed]], mc_samples) .* mean_sigma .* tau
density(beta', label=false)

ms_coeffs = MirrorStatistic.mirror_statistic(
    beta[:, 1:ms_samples],
    beta[:, ms_samples+1:mc_samples]
)
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
inclusion_matrix = ms_coeffs .> opt_t
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)

metrics_mc = MirrorStatistic.wrapper_metrics(
    vcat(data_dict["beta_reg"]...) .!= 0.,
    selection
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
