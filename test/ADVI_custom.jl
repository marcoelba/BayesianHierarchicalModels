# ADVI with custom variational distributions

using Optimisers
using Distributions
using DistributionsAD
using LogExpFunctions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "my_optimisers.jl"))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "training_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "bijectors_extension.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))

include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))



# data
n_individuals = 500
p = 1000
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

data_dict = generate_logistic_model_data(;
    n_individuals, class_threshold=0.5f0,
    p, p1, p0, beta_pool=Float32.([-2., 2]), obs_noise_sd=0.5, corr_factor=0.5,
    random_seed=124, dtype=Float32
)


" Model definition "

# Define priors and Variational Distributions
params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
    init_z=[0., 0.1]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(0.1, 0.1)
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 1., 1.),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=[]
)
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::Real) -> DistributionsLogPdf.log_normal(x, sigma=sigma .* tau),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.01, randn(p)*0.01),
    dependency=["sigma_beta", "tau_beta"]
)


# Constant Distribution
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=5.),
    dim_z=1,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_constant_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(1, ),
    random_variable=false,
    noisy_gradient=1
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, tau::Real) -> DistributionsLogPdf.log_half_cauchy(x, sigma=tau .* ones(p)),
    dim_z=p,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_constant_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(p)*0.1,
    dependency=["tau_beta"],
    random_variable=false,
    noisy_gradient=1
)

update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_normal(x),
    dim_z=p,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_constant_mv_normal(z; bij=identity),
    init_z=randn(p)*0.01,
    dependency=[]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> Distributions.logpdf(
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(1., 0.5)
)


# Variational Distributions
params_dict["priors"]
params_dict["vi_family_array"]
params_dict["ranges_z"]
params_dict["tot_vi_weights"]
params_dict["random_weights"]
params_dict["noisy_gradients"]

# get ONE VI distribution
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
q_dist_array = VariationalDistributions.get_variational_dist(z, params_dict["vi_family_array"], params_dict["ranges_z"])

# sample
VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=true)
theta = VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=false)
VariationalDistributions.rand_array(q_dist_array, from_base_dist=true)

# sample with log-jacobian
VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=params_dict["random_weights"])

# Entropy
for dist in q_dist_array
    println(VariationalDistributions.entropy(dist))
end


# Model prediction
function model(theta::AbstractArray; X::AbstractArray, prior_position=params_dict["tuple_prior_position"])
    mu = X * theta[prior_position[:beta]] .+ theta[prior_position[:beta0]]
    return (mu, )
end

pred = model(theta, X=data_dict["X"])

DistributionsLogPdf.log_bernoulli_from_logit(data_dict["y"], pred...)

# Non-Centred Model
function model(theta::AbstractArray; X::AbstractArray, prior_position=params_dict["tuple_prior_position"])
    beta_reg = theta[prior_position[:beta]] .* theta[prior_position[:sigma_beta]]
    mu = X * beta_reg .+ theta[prior_position[:beta0]]
    return (mu, )
end
pred = model(theta, X=data_dict["X"])


# joint prior
compute_logpdf_prior(theta; params_dict=params_dict)

elbo(z;
    y=data_dict["y"],
    X=data_dict["X"],
    ranges_z=params_dict["ranges_z"],
    vi_family_array=params_dict["vi_family_array"],
    random_weights=params_dict["random_weights"],
    model,
    log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_samples=10
)


# test loop
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MyOptimisers.DecayedADAGrad()
optimiser = Optimisers.RMSProp(0.01)
n_iter=1000

res = hybrid_training_loop(
    z=z,
    y=data_dict["y"],
    X=data_dict["X"],
    params_dict=params_dict,
    model=model,
    log_likelihood=DistributionsLogPdf.log_bernoulli_from_logit,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=n_iter,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=3,
    lr_schedule=ones(n_iter) .* 0.01
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

prior_position = params_dict["tuple_prior_position"]

beta = rand(q[prior_position[:beta]], 500)
density(beta', label=false)

scatter(q[prior_position[:beta]].dist.μ)

scatter(mean(rand(q[prior_position[:sigma_beta]], 500), dims=2))
beta = rand(q[prior_position[:beta]], 500) .* rand(q[prior_position[:sigma_beta]], 500)
density(beta', label=false)

# tau
mean(rand(q[prior_position[:tau_beta]], 100))


# Mirror statistic
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))

ms_dist = MirrorStatistic.posterior_ms_coefficients(q_beta)
ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta]].dist)
density(rand(ms_dist, 1000)')

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=1000,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)

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
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)
beta = rand(q[prior_position[:beta]], mc_samples) .* rand(q[prior_position[:sigma_beta]], mc_samples)

ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
inclusion_matrix = ms_coeffs .> opt_t
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    0.1
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)
