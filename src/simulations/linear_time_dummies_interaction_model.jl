# Linear time dummies model
using OrderedCollections
using Distributions
using StatsPlots
using DataFrames
using CSV

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


n_individuals = 100
n_time_points = 5
p = 100
p1 = Int(p * 0.1)
p0 = p - p1
corr_factor = 0.5

# label to save results
label_files = "algo_time_int_n$(n_individuals)_t$(n_time_points)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

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

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p, n_time_points),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(p, n_time_points) .* sigma_beta_prior')
    )
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dimension=(p, n_time_points),
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
        x, sigma=Float32(1)
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


    model(theta_components) = Predictors.linear_time_model(
        theta_components;
        X=data_dict["Xfix"]
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
        n_iter=n_iter,
        n_chains=n_chains,
        samples_per_step=2,
        use_noisy_grads=false,
        n_cycles=1
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
    random_seed=13435,
    dtype=Float32
)


model(theta_components) = Predictors.linear_time_model(
    theta_components;
    X=data_dict["Xfix"]
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
    n_iter=n_iter,
    n_chains=n_chains,
    samples_per_step=2,
    use_noisy_grads=false,
    n_cycles=1
)


plot(res["elbo_trace"][1:end, :])
plot(res["elbo_trace"][3000:end, :])

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
        "$(label_files)_density_beta_time_interactions.pdf"))

beta_time_samples = extract_parameter(
    prior="beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)

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
