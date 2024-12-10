# Test on interactions
using OrderedCollections
using Distributions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))


n_individuals = 500
n_time_points = 3
p = 1500
p1 = Int(p * 0.05)
p0 = p - p1

# time effect dummies - first one is the baseline intercept
beta_time = [1., 1., 2.]

p_int_t2 = 10
p_int_t3 = 5

beta_time_int = hcat(
    vcat(zeros(p - p_int_t2), ones(p_int_t2)),
    vcat(zeros(p - p_int_t3), ones(p_int_t3))
)

data_dict = generate_time_interaction_model_data(
    n_individuals=n_individuals,
    n_time_points=n_time_points,
    p=p, p1=p1, p0=p0,
    beta_pool=Float32.([-1., -2., 1, 2]),
    obs_noise_sd=0.5,
    corr_factor=0.5,
    include_random_int=false, random_int_from_pool=true,
    beta_time=beta_time,
    beta_time_int=beta_time_int,
    random_seed=124,
    dtype=Float32
)

p_tot = p * n_time_points

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
        x, sigma=Float32.(ones(p, n_time_points) .* [1, 1, 1]')
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


model(theta_components) = Predictors.linear_time_model(
    theta_components;
    X=data_dict["Xfix"]
)
theta_c = get_parameters_axes(params_dict)[2]
model(theta_c)[1]
model(theta_c)[2]

DistributionsLogPdf.log_normal(data_dict["y"], model(theta_c)...)

# model log joint
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)
theta = Float32.(ones(params_dict["tot_params"]))
partial_log_joint(Float32.(ones(params_dict["tot_params"])))

# VI distribution
vi_dist(z::AbstractArray) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z_dim=params_dict["tot_params"]*2,
    n_iter=2000,
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
data_dict["beta_fixed"]

plt = density(beta_samples[p+1:2*p, :]', label=false)
beta_time_int

plt = density(beta_samples[2*p+1:p*3, :]', label=true)
beta_time_int

beta_time_samples = extract_parameter(
    prior="beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_time_samples', label=true)
beta_time

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

# ---------------------------------------------------------
# Predictions
selection0 = (selection .== 0)
beta_range = collect(params_dict["priors"]["beta_fixed"]["range"])

obs = 10
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
