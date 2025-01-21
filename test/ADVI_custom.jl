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



# data
X = randn(100, 3)
y = 1. .+ X * [1, 2, 1] .+ randn(100)
p = 3


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
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_normal(x),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity)
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
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp)
)


# Variational Distributions
params_dict["priors"]
params_dict["priors"]["sigma_y"]["vi_family"]([1, -1])

params_dict["vi_family_array"]
params_dict["ranges_z"]
params_dict["tot_vi_weights"]

# get ONE VI distribution
z = randn(params_dict["tot_vi_weights"])
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
function model(theta::AbstractArray; X::AbstractArray)
    mu = X * theta[2] .+ theta[1]
    return (mu, theta[3])
end

pred = model(theta, X=X)

DistributionsLogPdf.log_normal(y, pred...)


# joint prior
compute_logpdf_prior(theta; params_dict=params_dict)

elbo(z;
    y=y,
    X=X,
    ranges_z=params_dict["ranges_z"],
    vi_family_array=params_dict["vi_family_array"],
    model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_samples=10
)


# test loop

z = randn(params_dict["tot_vi_weights"]) * 0.2
optimiser = MyOptimisers.DecayedADAGrad()

res = hybrid_training_loop(
    z=z,
    y=y,
    X=X,
    ranges_z=params_dict["ranges_z"],
    vi_family_array=params_dict["vi_family_array"],
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=1000,
    optimiser=optimiser,
    save_all=true,
    use_noisy_grads=false,
    elbo_samples=3
)

plot(res["loss_dict"]["z_trace"])
plot(res["loss_dict"]["loss"])

# Get VI distribution
res["best_iter_dict"]["best_iter"]
res["best_iter_dict"]["best_z"]

q = VariationalDistributions.get_variational_dist(
    res["best_iter_dict"]["best_z"],
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)
theta = VariationalDistributions.rand_array(q; reduce_to_vec=false)
