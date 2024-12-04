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
p = 4

# error terms for all time points
e = randn(n_individuals, n_time_points) * 0.5

# time effect dummies - first one is the baseline intercept
beta0 = 1.
beta_time = [beta0, 1., 0.]

# baseline
beta_baseline = [1., -1., 2., -1.]
X = Float32.(randn(n_individuals, p))

yt0 = beta0 .+ X * beta_baseline

# time 1
beta_int_t1 = [1., 1., 0., 0.]
yt1 = yt0 .+ beta_time[1] .+ X * beta_int_t1

# time 2
beta_int_t2 = [1., 0., 0., 0.]
yt2 = yt1 .+ beta_time[2] .+ X * beta_int_t2

# outocome
y = Float32.(hcat(yt0, yt1, yt2) .+ e)

p_tot = p * n_time_points

n_chains = 1
num_iter = 2000
MC_SAMPLES = 2000

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    dimension=(1,),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p, n_time_points),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(p, n_time_points) * 1)
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

function linear_time_model(
    theta_c::ComponentArray;
    Xfix::AbstractArray
    )
    n, p = size(Xfix)
    n_time = length(theta_c["beta_time"])

    # baseline
    mu_inc = [
        theta_c["beta_time"][tt] .+ Xfix * theta_c["beta_fixed"][:, tt] for tt = 1:n_time
    ]
    mu = [
        mu_inc[tt-1] .+ mu_inc[tt] for tt = 2:n_time
    ]

    sigma = [Float32.(ones(n)) .* theta_c["sigma_y"] for tt = 1:n_time]

    return (hcat(mu_inc[1], mu...), hcat(sigma...))
end

model(theta_components) = linear_time_model(
    theta_components;
    Xfix=X
)
theta_c = get_parameters_axes(params_dict)[2]
model(theta_c)[1]
model(theta_c)[2]

DistributionsLogPdf.log_normal(y, model(theta_c)...)

# model log joint
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=y
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

plot(res["elbo_trace"][100:end, :])
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
plt = density(beta_samples[1:p, :]', label=true)
beta_baseline

plt = density(beta_samples[p+1:2*p, :]', label=true)
beta_int_t1

plt = density(beta_samples[2*p+1:p*3, :]', label=true)
beta_int_t2


beta0_samples = extract_parameter(
    prior="beta0_fixed",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta0_samples', label=false)

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
    prior="beta",
    params_dict=params_dict
)
plt = density(rand(ms_dist, MC_SAMPLES)', label=false)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)

# distribution
boxplot(metrics.fdr_distribution, label="FDR")
boxplot!(metrics.tpr_distribution, label="TPR")

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


# range
metrics.metrics_mean
metrics.metrics_median
scatter(sort(unique(metrics.n_inclusion_per_mc)), metrics.fdr_range)
scatter!(sort(unique(metrics.n_inclusion_per_mc)), metrics.tpr_range)


plt = scatter_sel_matrix(metrics.inclusion_matrix, p0=p0)


# LASSO
using GLMNet

y_q = []
for yi in data_dict["y"]
    if yi == 1
        push!(y_q, "yes")
    else
        push!(y_q, "no")
    end
end

lasso_cv = glmnetcv(data_dict["X"], y_q)
lasso_cv.path
