# Linear time dummies model
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
n_time_points = 5
p = 100
p1 = Int(p * 0.1)
p0 = p - p1
corr_factor = 0.5

# label to save results
label_files = "algo_prod_time_int_n$(n_individuals)_t$(n_time_points)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

# time effect dummies - first one is the baseline intercept
beta_time = [1., 1., 2., 1., 0.]

p_int_t2 = 10
p_int_t3 = 5
p_int_t4 = 5
p_int_t5 = 0

beta_time_int = hcat(
    vcat(zeros(p - p_int_t2), ones(p_int_t2)),
    vcat(zeros(p - p_int_t3), ones(p_int_t3)),
    vcat(zeros(p - p_int_t4), ones(p_int_t4)),
    vcat(zeros(p - p_int_t5), ones(p_int_t5))
)

p_tot = p * n_time_points
sigma_beta_prior = [1, 0.1, 0.1, 0.1, 0.1]

n_chains = 1
num_iter = 2000
MC_SAMPLES = 2000

params_dict = OrderedDict()

# beta fixed - HS
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, n_time_points),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=ones(p, n_time_points) .* sigma_beta_prior'
    ),
    dim_z=p*n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(p*n_time_points*2)*0.05
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dim_theta=(p, n_time_points),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma
    ),
    dim_z=p*n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(p*n_time_points*2)*0.05,
    dependency=["sigma_beta"]
)


# beta fixed - Prod
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(n_time_points, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=ones(n_time_points)
    ),
    dim_z=n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(n_time_points*2)*0.05
)
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, n_time_points),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(
        x, 1., 1.
    ),
    dim_z=p*n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=randn(p*n_time_points*2)*0.05
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dim_theta=(p, n_time_points),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::AbstractArray) -> DistributionsLogPdf.log_normal(
        x, sigma=sigma .* tau'
    ),
    dim_z=p*n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(p*n_time_points*2)*0.05,
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
    name="beta_time",
    dim_theta=(n_time_points,),
    logpdf_prior=(x::AbstractArray, sigma::Real) -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_time_points)) .* sigma
    ),
    dim_z=n_time_points*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=randn(n_time_points*2)*0.05,
    dependency=["sigma_beta_time"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1,),
    logpdf_prior=x::Real -> Distributions.logpdf(
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(2)*0.05
)

prior_position = params_dict["tuple_prior_position"]

fdr_target = 0.1
n_iter = 2000
n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_time_interaction_model_data(
        n_individuals=n_individuals,
        n_time_points=n_time_points,
        p=p, p1=p1, p0=p0,
        beta_pool=Float32.([-1., -2., 1, 2]),
        obs_noise_sd=0.5,
        corr_factor=corr_factor,
        beta_time=beta_time,
        beta_time_int=beta_time_int,
        random_seed=random_seed + simu,
        dtype=Float32
    )

    model(theta; X) = Predictors.linear_time_model(
        theta;
        X,
        prior_position=prior_position
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
        n_iter=num_iter,
        optimiser=optimiser,
        save_all=false,
        use_noisy_grads=false,
        elbo_samples=3
    )

    vi_posterior = average_posterior(
        res["posteriors"],
        Distributions.MultivariateNormal
    )

    simulations_models[simu] = (vi_posterior, res["elbo_trace"], params_dict)

    # Mirror Statistic
    ms_dist = MirrorStatistic.posterior_ms_coefficients(
        vi_posterior=vi_posterior,
        prior="beta_fixed",
        params_dict=params_dict
    )
    plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=vcat(data_dict["beta_fixed"]...),
        fdr_target=fdr_target
    )

    metrics_dict = Dict()

    # Posterior
    inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        inclusion_probs,
        fdr_target
    )

    metrics_posterior = MirrorStatistic.wrapper_metrics(
        vcat(data_dict["beta_fixed"]...) .!= 0.,
        selection
    )
    
    metrics_dict["fdr_range"] = metrics.fdr_range
    metrics_dict["tpr_range"] = metrics.tpr_range
    metrics_dict["metrics_posterior"] = metrics_posterior

    simulations_metrics[simu] = metrics_dict

end

posterior_fdr = []
posterior_tpr = []
for simu = 1:n_simulations
    push!(posterior_fdr, simulations_metrics[simu]["metrics_posterior"].fdr)
    push!(posterior_tpr, simulations_metrics[simu]["metrics_posterior"].tpr)
end

all_metrics = hcat(posterior_fdr, posterior_tpr)
df = DataFrame(all_metrics, ["posterior_fdr", "posterior_tpr"])
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

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))


# ----------------------------------------------------------------------
# DETAILED ANALYSIS

data_dict = generate_time_interaction_model_data(
    n_individuals=n_individuals,
    n_time_points=n_time_points,
    p=p, p1=p1, p0=p0,
    beta_pool=Float32.([-1., -2., 1, 2]),
    obs_noise_sd=0.5,
    corr_factor=corr_factor,
    beta_time=beta_time,
    beta_time_int=beta_time_int,
    random_seed=random_seed,
    dtype=Float32
)

model(theta; X) = Predictors.linear_time_model(
    theta;
    X
)

function model(
    theta::ComponentArray;
    X::AbstractArray
    )
    n, p = size(X)
    n_time = length(theta[:beta_time])

    # baseline
    mu_inc = [
        theta[:beta_time][tt] .+ X * theta[:beta_fixed][:, tt] .* theta[:sigma_beta][:, tt] for tt = 1:n_time
    ]

    mu = cumsum(reduce(hcat, mu_inc), dims=2)
    sigma = reduce(hcat, [Float32.(ones(n)) .* theta[:sigma_y] for tt = 1:n_time])

    return (mu, sigma)
end

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
model(theta; X=data_dict["Xfix"])

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
    n_iter=num_iter,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=3
)


plot(res["loss_dict"]["z_trace"])
plot(res["loss_dict"]["loss"])
plot(res["loss_dict"]["loss"][1500:end])

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
theta = VariationalDistributions.rand_array(q; reduce_to_vec=false)

beta = rand(q[prior_position[:beta_fixed]], 1000)'
beta = rand(q[prior_position[:beta_fixed]], 1000)' .* rand(q[prior_position[:sigma_beta]], 1000)'
density(beta[:, 1:p], label=false)
p1

# beta baseline
plt = density(beta_samples[1:p, :]', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", 
        "$(label_files)_density_beta_t0.pdf"))

plt_t1 = density(beta_samples[p+1:2*p, :]', label=false)
ylabel!("Time 1")

plt_t2 = density(beta_samples[2*p+1:p*3, :]', label=false)
ylabel!("Time 2")

plt_t3 = density(beta_samples[3*p+1:p*4, :]', label=false)
ylabel!("Time 3")

plt_t4 = density(beta_samples[4*p+1:p*5, :]', label=false)
ylabel!("Time 4")

plt = plot(plt_t1, plt_t2, plt_t3, plt_t4)
savefig(plt, joinpath(abs_project_path, "results", "simulations", 
        "$(label_files)_density_beta_time_interactions.pdf")
)


beta_time_samples = rand(q[prior_position[:beta_time]], 1000)

plt = plot()
for tt = 1:n_time_points
    density!(beta_time_samples[tt, :], label="t=$(tt) - gt=$(Int(beta_time[tt]))", fontsize=10)
end
ylabel!("Density")
display(plt)

savefig(
    plt,
    joinpath(
        abs_project_path, "results", "simulations", 
        "$(label_files)_density_beta_time.pdf"
    )
)


# Mirror Statistic
ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta_fixed]].dist)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=vcat(data_dict["beta_fixed"]...),
    fdr_target=0.1
)

# distribution
plt = fdr_n_hist(metrics)

# Newton's rule
n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    0.1
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

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
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "simulations", 
    "$(label_files)_inclusion_probs.pdf")
)


# ---------------------------------------------------------
# Predictions
selection0 = (selection .== 0)
beta_range = collect(params_dict["priors"]["beta_fixed"]["range"])

obs = 2
mu_pred = []

for theta in eachcol(samples_posterior)
    t_temp = copy(theta)
    t_temp[beta_range[selection0]] .= 0.
    theta_components = ComponentArray(t_temp, theta_axes)

    lin_pred = Predictors.linear_time_model(
        theta_components;
        X=data_dict["Xfix"][obs:obs, :]
    )
    push!(mu_pred, lin_pred[1])
    sigma = lin_pred[2]
end

mu_pred = vcat(mu_pred...)
plot(mu_pred', label=false, color="lightgrey")
plot!(data_dict["y"][obs, :], linewidth=3, col=2, label="True")
