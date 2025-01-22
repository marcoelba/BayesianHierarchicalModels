# ADVI with custom variational distributions

using Zygote
using Optimisers
using Bijectors
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
n_individuals = 300
p = 100
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=134
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
    logpdf_prior=(x::AbstractArray, tau::Real) -> DistributionsLogPdf.log_half_cauchy(x, sigma=tau .* ones(p)),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=["tau_beta"]
)

# Constant Distribution
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=1,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_constant_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(0.1, )
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, tau::Real) -> DistributionsLogPdf.log_half_cauchy(x, sigma=tau .* ones(p)),
    dim_z=p,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_constant_mv_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=randn(p)*0.1,
    dependency=["tau_beta"]
)


update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.5, randn(p)*0.1),
    dependency=["sigma_beta"]
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
params_dict["priors"]["sigma_y"]["vi_family"]([1, -1])

params_dict["vi_family_array"]
params_dict["ranges_z"]
params_dict["tot_vi_weights"]

# get ONE VI distribution
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
q_dist_array = VariationalDistributions.get_variational_dist(z, params_dict["vi_family_array"], params_dict["ranges_z"])

# sample
VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=true)
theta = VariationalDistributions.rand_array(q_dist_array, reduce_to_vec=false)
VariationalDistributions.rand_array(q_dist_array, from_base_dist=true)

# sample with log-jacobian
VariationalDistributions.rand_with_logjacobian(q_dist_array)

# Entropy
for dist in q_dist_array
    println(VariationalDistributions.entropy(dist))
end


# Model prediction
function model(theta::AbstractArray; X::AbstractArray, prior_position=params_dict["tuple_prior_position"])
    mu = X * theta[prior_position[:beta]] .+ theta[prior_position[:beta0]]
    return (mu, theta[prior_position[:sigma_y]] .* ones(eltype(X), size(X, 1)))
end

pred = model(theta, X=data_dict["X"])

DistributionsLogPdf.log_normal(data_dict["y"], pred...)

# joint prior
compute_logpdf_prior(theta; params_dict=params_dict)

elbo(z;
    y=data_dict["y"],
    X=data_dict["X"],
    ranges_z=params_dict["ranges_z"],
    vi_family_array=params_dict["vi_family_array"],
    model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_samples=10
)


# test loop

z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MyOptimisers.DecayedADAGrad()

res = hybrid_training_loop(
    z=z,
    y=data_dict["y"],
    X=data_dict["X"],
    ranges_z=params_dict["ranges_z"],
    vi_family_array=params_dict["vi_family_array"],
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=2000,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=5
)

plot(res["loss_dict"]["z_trace"])
plot(res["loss_dict"]["loss"])
plot(res["loss_dict"]["loss"][500:end])

# Get VI distribution
res["best_iter_dict"]["best_iter"]
res["best_iter_dict"]["best_iter"]
res["best_iter_dict"]["best_z"]
res["best_iter_dict"]["final_z"]

q = VariationalDistributions.get_variational_dist(
    res["best_iter_dict"]["best_z"],
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)
theta = VariationalDistributions.rand_array(q; reduce_to_vec=false)

prior_position = params_dict["tuple_prior_position"]

beta = rand(q[prior_position[:beta]], 1000)
density(beta', label=false)

# lambda
rand(q[prior_position[:sigma_beta]])
# tau
rand(q[prior_position[:tau_beta]])

# Mirror statistic
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))

ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta]].dist)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=1000,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)


boxplot(metrics.fdr_distribution, label="FDR")
boxplot!(metrics.tpr_distribution, label="TPR")

plt = fdr_n_hist(metrics)

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
