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


n_individuals = 100
n_repetitions = 5
n_time_points = 4

p = 100
p1 = Int(p * 0.1)
p0 = p - p1
p_tot = p * n_time_points

p_int_t2 = 0
p_int_t3 = 0
p_int_t4 = 0

beta_time_int = hcat(
    vcat(zeros(p - p_int_t2), ones(p_int_t2)),
    vcat(zeros(p - p_int_t3), ones(p_int_t3)),
    vcat(zeros(p - p_int_t4), ones(p_int_t4))
)

# Intercept - Fixed effect
beta0 = 1.
# Intercept - Random effects
u0 = randn(n_individuals) * 5.

# Time dummy effect
beta_time = [beta0, 1., 2., 0.]
beta_time = beta_time .+ randn(n_time_points, n_repetitions) * 0.05

data_dict = generate_time_interaction_multiple_measurements_data(
    n_individuals=n_individuals,
    n_time_points=n_time_points,
    n_repeated_measures=n_repetitions,
    p=p, p1=p1, p0=p0,
    beta_pool=Float32.([-1., -2., 1, 2]),
    obs_noise_sd=0.5,
    corr_factor=0.5,
    include_random_int=true, random_int_from_pool=false,
    random_intercept_sd=2.,
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
for ii = 1:n_repetitions
    plot!(data_dict["y"][1, :, ii], label=false)
end
display(plt)

#
n_chains = 1
num_iter = 2000
MC_SAMPLES = 2000

params_dict = OrderedDict()

update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    dimension=(1, ),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(
        x
    )
)

# Random intercept beta0
update_parameters_dict(
    params_dict;
    name="sigma_beta0",
    dimension=(1, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(1)
    )
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    dimension=(n_individuals,),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::Float32) -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_individuals) * sigma)
    ),
    dependency=["sigma_beta0"]
)

# beta fixed
# hierarchical prior

# group prior on the variance of the half-cauchy
hyperprior_sigma = Float32.(ones(p, n_time_points))
hyperprior_sigma[:, 2:n_time_points] .= 0.1f0

update_parameters_dict(
    params_dict;
    name="sigma_beta_group",
    dimension=(p, n_time_points),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=hyperprior_sigma
    )
)
# update_parameters_dict(
#     params_dict;
#     name="beta_group",
#     dimension=(p, n_time_points),
#     log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(
#         x, sigma=sigma
#     ),
#     dependency=["sigma_beta_group"]
# )

# prior over repeated measurements
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    dimension=(p, n_time_points, n_repetitions),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(
        x,
        mu=Float32.(zeros(p, n_time_points, n_repetitions)),
        sigma=sigma .* Float32.(ones(p, n_time_points, n_repetitions))
    ),
    dependency=["sigma_beta_group"]
)

# beta time
update_parameters_dict(
    params_dict;
    name="sigma_beta_time",
    dimension=(1, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32(3)
    )
)
update_parameters_dict(
    params_dict;
    name="mu_beta_time",
    dimension=(n_time_points, ),
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_normal(
        x, sigma=Float32.(ones(n_time_points))
    )
)

update_parameters_dict(
    params_dict;
    name="beta_time",
    dimension=(n_time_points, n_repetitions),
    log_prob_fun=(x::AbstractArray{Float32}, mu::AbstractArray{Float32}, sigma::Float32) -> DistributionsLogPdf.log_normal(
        x, mu=mu, sigma=Float32.(ones(n_time_points)) .* sigma
    ),
    dependency=["mu_beta_time", "sigma_beta_time"]
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


model(theta_components, rep_index) = Predictors.linear_time_random_intercept_model(
    theta_components,
    rep_index,
    X=data_dict["Xfix"],
    n_individuals=n_individuals,
    n_time_points=n_time_points,
)

theta_c = get_parameters_axes(params_dict)[2]
model(theta_c, 1)[1]
model(theta_c, 1)[2]

# model log joint
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"],
    n_repeated_measures=n_repetitions
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

# hyperprior on beta - mu
beta_group_samples = extract_parameter(
    prior="sigma_beta_group",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
# beta baseline
plt = density(beta_group_samples', label=false)
data_dict["beta_fixed"][:, :, 1]

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

plt = density(beta_samples[2*p+1:p*3, :]', label=false)
beta_time_int

beta_time_samples = extract_parameter(
    prior="beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_time_samples', label=true)
beta_time

mu_beta_time_samples = extract_parameter(
    prior="mu_beta_time",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(mu_beta_time_samples', label=true)
beta_time


# beta0 random int
beta0_fixed = extract_parameter(
    prior="beta0_fixed",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta0_fixed', label=true)


# beta0 random int
beta0_random = extract_parameter(
    prior="beta0_random",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta0_random', label=true)

# beta0 random int
sigma_beta0 = extract_parameter(
    prior="sigma_beta0",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(sigma_beta0', label=true)

# sigma y
sigma_y = extract_parameter(
    prior="sigma_y",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(sigma_y', label=true)


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
    beta_true=vcat(mean(data_dict["beta_fixed"], dims=3)[:,:,1]...),
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
    # t_temp[beta_range[selection0]] .= 0.
    theta_components = ComponentArray(t_temp, theta_axes)

    lin_pred = Predictors.linear_time_random_intercept_model(
        theta_components;
        n_individuals=n_individuals,
        n_time_points=n_time_points,
        n_repetitions=n_repetitions    
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
