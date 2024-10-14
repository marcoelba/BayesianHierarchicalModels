# test mixture model
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
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))



n_individuals = 300
n_group = 100

K = 3

mu = vcat(
    ones(n_group) * -1,
    ones(n_group) * 0,
    ones(n_group)
)
X = Float32.(ones(n_individuals))
y = Float32.(mu .+ randn(n_individuals) * 0.5)
histogram(y, bins=20)

n_chains = 1
num_iter = 1000
MC_SAMPLES=2000

params_dict = OrderedDict()

# Mixture prior distribution on random intercept
n_clusters = 5

# mixing probabilities
update_parameters_dict(
    params_dict;
    name="cluster_probs",
    dimension=(n_clusters, ),
    bij=StatsFuns.softmax,
    log_prob_fun=x::AbstractArray{<:Float32} -> Distributions.logpdf(
        Distributions.Dirichlet(n_clusters, 3f0), x
    )
)

update_parameters_dict(
    params_dict;
    name="beta0",
    dimension=(n_clusters, ),
    log_prob_fun=
        x::AbstractArray{<:Float32} -> DistributionsLogPdf.log_normal(
        x, Float32.(zeros(n_clusters)), Float32.(ones(n_clusters)) * 2f0
    )
)

# try
# theta_components # from utils
# DistributionsLogPdf.log_normal_mixture(
#     theta_components["beta0"],
#     theta_components["cluster_probs"],
#     theta_components["cluster_means"],
#     Float32.(ones(n_clusters)) .* theta_components["cluster_sigma"]
# )
# logpdf.(
#     Distributions.MixtureModel([
#         Normal(theta_components["cluster_means"][1], theta_components["cluster_sigma"]),
#         Normal(theta_components["cluster_means"][2], theta_components["cluster_sigma"]),
#         Normal(theta_components["cluster_means"][3], theta_components["cluster_sigma"])
#     ], theta_components["cluster_probs"]),
#     theta_components["beta0"]
# )


# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dimension=(1, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_half_cauchy(x, sigma=3f0)
)

theta_axes, _ = get_parameters_axes(params_dict)


# model predictions
function mixture_predictor(
    theta_c::ComponentArray;
    X::AbstractArray
    )

    mu = X * theta_c["beta0"]'
    sigma = Float32.(ones(size(mu))) .* theta_c["sigma_y"]

    return (theta_c["cluster_probs"]', mu, sigma)
end


model(theta_components) = mixture_predictor(
    theta_components;
    X=X
)

model(get_parameters_axes(params_dict)[2])[1]

DistributionsLogPdf.loglik_normal_mixture(
    y,
    model(get_parameters_axes(params_dict)[2])...
)

# model
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.loglik_normal_mixture,
    label=y
)
partial_log_joint(Float32.(randn(params_dict["tot_params"])))

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
    use_noisy_grads=false,
    n_cycles=1
)

plot(res["elbo_trace"])

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
    prior="beta0",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_samples', label=false)

probs_samples = extract_parameter(
    prior="cluster_probs",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(probs_samples', label=false)

sigma_samples = extract_parameter(
    prior="sigma_y",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(sigma_samples', label=false)

# posterior cluster inclusion probabilities
ind = 150
cluster_probs = posterior_cluster_prob(
    y_i=y[ind],
    x_i=X[ind, :],
    samples_posterior=samples_posterior,
    n_clusters=n_clusters,
    model=mixture_predictor,
    theta_axes=theta_axes
)
boxplot(cluster_probs')
