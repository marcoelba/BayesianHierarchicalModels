# Time interaction model with repeated measurements
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
n_time_points = 4
n_repeated_measures = 5
p = 100
p1 = Int(p * 0.1)
p0 = p - p1
corr_factor = 0.5

# label to save results
label_files = "algo_time_int_repeated_n$(n_individuals)_t$(n_time_points)_M$(n_repeated_measures)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

# time effect dummies - first one is the baseline intercept
beta_time = [1., 1., 2., 0.]
beta_time = beta_time .+ randn(n_time_points, n_repeated_measures) * 0.05

p_int_t2 = 0
p_int_t3 = 0
p_int_t4 = 0

beta_time_int = hcat(
    vcat(zeros(p - p_int_t2), ones(p_int_t2)),
    vcat(zeros(p - p_int_t3), ones(p_int_t3)),
    vcat(zeros(p - p_int_t4), ones(p_int_t4))
)

#
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
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2) * 0.05
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    dim_theta=(n_individuals, ),
    logpdf_prior=(x::AbstractArray, sigma::Real) -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_individuals) * sigma)
    ),
    dim_z=n_individuals * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(n_individuals * 2)*0.05,
    dependency=["sigma_beta0"]
)


# beta fixed - Regressors
# HS hierarchical prior

# group prior on the variance of the half-cauchy
hyperprior_sigma = ones(p, n_time_points)


# update_parameters_dict(
#     params_dict;
#     name="sigma_beta_group",
#     dim_theta=(p, n_time_points),
#     logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_half_cauchy(
#         x, sigma=hyperprior_sigma
#     ),
#     dim_z=p*n_time_points*2,
#     vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
#     init_z=randn(p*n_time_points*2)*0.05
# )
# update_parameters_dict(
#     params_dict;
#     name="mu_beta_group",
#     dim_theta=(p, n_time_points),
#     logpdf_prior=(x::AbstractArray, sigma::AbstractArray) -> DistributionsLogPdf.log_normal(
#         x, sigma=sigma
#     ),
#     dim_z=p*n_time_points*2,
#     vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
#     init_z=randn(p*n_time_points*2)*0.05,
#     dependency=["sigma_beta_group"]
# )

# # beta given mu
# update_parameters_dict(
#     params_dict;
#     name="beta_fixed",
#     dim_theta=(p, n_time_points, n_repeated_measures),
#     logpdf_prior=(x::AbstractArray, mu::AbstractArray) -> DistributionsLogPdf.log_normal(
#         x,
#         mu=mu .* ones(p, n_time_points, n_repeated_measures),
#         sigma=ones(p, n_time_points, n_repeated_measures)
#     ),
#     dim_z=p * n_time_points * n_repeated_measures * 2,
#     vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
#     init_z=randn(p*n_time_points*n_repeated_measures*2)*0.05,
#     dependency=["mu_beta_group"]
# )


"""
    Product prior
"""

update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(p, n_time_points),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_half_cauchy(x, sigma=hyperprior_sigma),
    dim_z=p*n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(p*n_time_points*2) .* 0.1
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, n_time_points, n_repeated_measures),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 1., 1.),
    dim_z=p * n_time_points * n_repeated_measures * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=randn(p * n_time_points * n_repeated_measures * 2) * 0.1
)

update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dim_theta=(p, n_time_points, n_repeated_measures),
    logpdf_prior=(x::AbstractArray, sigma_beta::AbstractArray, tau::AbstractArray) -> DistributionsLogPdf.log_normal(
        x,
        mu=zeros(p, n_time_points, n_repeated_measures),
        sigma=tau .* sigma_beta
    ),
    dim_z=p * n_time_points * n_repeated_measures * 2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(p * n_time_points * n_repeated_measures * 2) * 0.1,
    dependency=["sigma_beta", "tau_beta"]
)


# beta time
update_parameters_dict(
    params_dict;
    name="sigma_beta_time",
    dim_theta=(1,),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32(1)
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2)*0.05
)
update_parameters_dict(
    params_dict;
    name="mu_beta_time",
    dim_theta=(n_time_points,),
    logpdf_prior=(x::AbstractArray, sigma::Real) -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_time_points)) .* sigma
    ),
    dim_z=n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(n_time_points*2)*0.05,
    dependency=["sigma_beta_time"]
)

update_parameters_dict(
    params_dict;
    name="beta_time",
    dim_theta=(n_time_points, n_repeated_measures),
    logpdf_prior=(x::AbstractArray, mu::AbstractArray, sigma::Real) -> DistributionsLogPdf.log_normal(
        x, mu=mu, sigma=Float32.(ones(n_time_points)) .* sigma
    ),
    dim_z=n_time_points*n_repeated_measures*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(n_time_points*n_repeated_measures*2)*0.05,
    dependency=["mu_beta_time", "sigma_beta_time"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1,),
    logpdf_prior=x::Real -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2)*0.05
)

prior_position = params_dict["tuple_prior_position"]

fdr_target = 0.1
MC_SAMPLES = 1000
n_iter = 4000
n_simulations = 30
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

# Run inference
for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_time_interaction_multiple_measurements_data(
        n_individuals=n_individuals,
        n_time_points=n_time_points,
        n_repeated_measures=n_repeated_measures,
        p=p, p1=p1, p0=p0,
        beta_pool=Float32.([-1., -2., 1, 2]),
        sd_noise_beta_reps=0.1,
        obs_noise_sd=0.5,
        corr_factor=corr_factor,
        include_random_int=true, random_int_from_pool=false,
        random_intercept_sd=5.,
        beta_time=beta_time,
        random_seed=random_seed + simu,
        dtype=Float32
    )

    model(theta_components, rep; X) = Predictors.linear_time_random_intercept_model(
        theta_components,
        rep;
        X
    )
    
    # Training
    z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
    optimiser = MyOptimisers.DecayedADAGrad()
    
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
        elbo_samples=3,
        n_repeated_measures=n_repeated_measures
    )
    best_z = res["best_iter_dict"]["best_z"]
    
    q = VariationalDistributions.get_variational_dist(
        best_z,
        params_dict["vi_family_array"],
        params_dict["ranges_z"]
    )
    
    simulations_models[simu] = (q, params_dict)

    # Mirror Statistic
    metrics_dict = Dict()

    ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta_fixed]].dist)
    
    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=vcat(data_dict["beta_fixed"]...),
        fdr_target=fdr_target
    )
    
    n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:, 1]
    mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:, 1]    
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )
    #
    metrics_posterior = MirrorStatistic.wrapper_metrics(
        vcat(data_dict["beta_fixed"]...) .!= 0.,
        selection
    )
    metrics_dict["metrics_posterior"] = metrics_posterior
    
    # Monte Carlo loop
    mc_samples = MC_SAMPLES * 2
    ms_samples = Int(mc_samples / 2)
    mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
    beta = rand(q[prior_position[:beta_fixed]], mc_samples) .* mean_sigma
    
    ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
    opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=fdr_target)
    inclusion_matrix = ms_coeffs .> opt_t
    mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]
    
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )
    
    metrics_mc = MirrorStatistic.wrapper_metrics(
        vcat(data_dict["beta_fixed"]...) .!= 0.,
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


# ----------------------------------------------------------------------
# DETAILED ANALYSIS

data_dict = generate_time_interaction_multiple_measurements_data(
    n_individuals=n_individuals,
    n_time_points=n_time_points,
    n_repeated_measures=n_repeated_measures,
    p=p, p1=p1, p0=p0,
    beta_pool=Float32.([-1., -2., 1, 2]),
    sd_noise_beta_reps=0.1,
    obs_noise_sd=0.5,
    corr_factor=corr_factor,
    include_random_int=true, random_int_from_pool=false,
    random_intercept_sd=5.,
    beta_time=beta_time,
    random_seed=124,
    dtype=Float32
)

plt = plot()
for ii = 1:n_individuals
    plot!(data_dict["y"][ii, :, 1], label=false)
end
display(plt)

plt = plot()
for ii = 1:n_repeated_measures
    plot!(data_dict["y"][1, :, ii], label=false)
end
display(plt)


model(theta_components, rep; X) = Predictors.linear_time_random_intercept_model(
    theta_components,
    rep;
    X
)

# get ONE VI distribution
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
q_dist_array = VariationalDistributions.get_variational_dist(
    z, params_dict["vi_family_array"], params_dict["ranges_z"]
)

# sample
VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=true)

# sample with log-jacobian
theta, jac = VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=params_dict["random_weights"])

# Entropy
for dist in q_dist_array
    println(VariationalDistributions.entropy(dist))
end

theta_axes = get_parameters_axes(params_dict)
theta = ComponentArray(theta, theta_axes)

model(theta, 1; X=data_dict["Xfix"])[1]
model(theta, 1; X=data_dict["Xfix"])[2]

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
    n_iter=4000,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=3,
    n_repeated_measures=n_repeated_measures
)

plot(res["loss_dict"]["loss"])
plot(res["loss_dict"]["loss"][300:end])

# Get VI distribution
res["best_iter_dict"]["best_iter"]
res["best_iter_dict"]["best_z"]
res["best_iter_dict"]["final_z"]

best_z = res["best_iter_dict"]["best_z"]

q = VariationalDistributions.get_variational_dist(
    best_z,
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)

# mu beta - constant over repeated measurements
mu_beta_group_samples = rand(q[prior_position[:mu_beta_group]], 1000)'

plt = density(mu_beta_group_samples[:, 1:p], label=false)
plt = density(mu_beta_group_samples[:, p+1:2*p], label=false)
plt = density(mu_beta_group_samples[:, 2*p+1:3*p], label=false)

plt = density(mu_beta_group_samples, label=false)

scatter(vcat(data_dict["beta_reg"]...))
scatter!(mean(mu_beta_group_samples', dims=2))

# beta probs
beta_probs = rand(q[prior_position[:sigma_beta]], 1000)'
scatter(mean(beta_probs, dims=1)', label=false)

# beta tau
beta_tau = rand(q[prior_position[:tau_beta]], 1000)'
scatter(mean(beta_tau, dims=1)', label=false)

# beta
beta = rand(q[prior_position[:beta_fixed]], 1000)'
density(beta, label=false)
density(beta .* mean(beta_probs, dims=1), label=false)


# MC, p, T, M
beta_samples = zeros(1000, size(ComponentArray(theta, theta_axes)[:beta_fixed])...)
for ii in 1:1000
    theta_c = ComponentArray(beta[ii, :], theta_axes)

    beta_samples[ii, :, :, :] .= theta_c[:sigma_beta] .* theta_c[:beta_fixed]
end

plt = density(beta_samples[:, :, 1, 1], label=false)
plt = density(beta_samples[:, p, :, 1], label=false)

# beta time
beta_time_samples = rand(q[prior_position[:beta_time]], 1000)'

plt = density(beta_time_samples, label=true)
beta_time

mu_beta_time_samples = rand(q[prior_position[:mu_beta_time]], 1000)'
plt = density(mu_beta_time_samples, label=true)
beta_time


# beta0 random int
beta0_random_samples = rand(q[prior_position[:beta0_random]], 1000)'
plt = density(beta0_random_samples, label=false)

# sigma y
sigma_y_samples = rand(q[prior_position[:sigma_y]], 1000)
plt = density(sigma_y_samples, label=true)


# Mirror Statistic
ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta_fixed]].dist)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=vcat(data_dict["beta_reg"]...),
    fdr_target=0.1
)
#
metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=vcat(data_dict["beta_fixed"]...),
    fdr_target=0.1
)

# distribution
plt = fdr_n_hist(metrics)

# Newton's rule
n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:, 1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    0.1
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = MirrorStatistic.wrapper_metrics(
    vcat(data_dict["beta_reg"]...) .!= 0.,
    selection
)
#
metrics = MirrorStatistic.wrapper_metrics(
    vcat(data_dict["beta_fixed"]...) .!= 0.,
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


# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)
mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
scatter(mean_sigma)
beta = rand(q[prior_position[:beta_fixed]], mc_samples) .* mean_sigma
density(beta', label=false)

ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
inclusion_matrix = ms_coeffs .> opt_t
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)

metrics_mc = MirrorStatistic.wrapper_metrics(
    vcat(data_dict["beta_fixed"]...) .!= 0.,
    selection
)


# ---------------------------------------------------------
# Predictions
selection0 = (selection .== 0)
beta_range = collect(params_dict["priors"]["beta_fixed"]["range"])

obs = 10
mu_pred = []

for theta in eachcol(samples_posterior)
    t_temp = copy(theta)
    # t_temp[beta_range[selection0]] .= 0.
    theta_components = ComponentArray(t_temp, theta_axes)

    lin_pred = Predictors.linear_time_random_intercept_model(
        theta_components;
        n_individuals=n_individuals,
        n_time_points=n_time_points,
        n_repeated_measures=n_repeated_measures    
    )
    push!(mu_pred, lin_pred[1])
    sigma = lin_pred[2]
end

mu_pred = vcat(mu_pred...)
plot(mu_pred[:,:, 1]', label=false, color="lightgrey")
plot!(data_dict["y"][obs, :], linewidth=3, col=2, label="True")

# with the mean
mean_posterior = mean(samples_posterior, dims=2)[:, 1]
mean_posterior[beta_range[selection0]] .= 0.

mean_components = ComponentArray(mean_posterior, theta_axes)
lin_pred = Predictors.linear_time_model(
    mean_components;
    X=data_dict["Xfix"]
)
mu_pred = lin_pred[1]
sigma_pred = lin_pred[2]

plot(mu_pred[2, :])
plot!(data_dict["y"][2, :])

plot(mu_pred[1, :])
plot!(data_dict["y"][1, :])

plot(mu_pred[3, :])
plot!(data_dict["y"][3, :])
