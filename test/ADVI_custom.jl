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

DistributionsLogPdf.log_normal(y, pred..., T=Float64)


# joint prior
function log_prior(theta::AbstractArray)

    logprior = sum(DistributionsAD.logpdf.(DistributionsAD.Normal(0., 1.), theta[1])) +
        sum(DistributionsAD.logpdf(truncated(DistributionsAD.Normal(0., 1.), 0., Inf64), theta[2]))
    
    return logprior
end

elbo(z;
    y=y,
    X=X,
    ranges_z=params_dict["ranges_z"],
    q_family_array=params_dict["vi_family_array"],
    model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x->zero(1.),
    n_samples=10
)


# test loop

z = randn(dim_z) * 0.2

opt = MyOptimisers.DecayedADAGrad()

state = Optimisers.setup(opt, z)

n_iter = 1000
z_trace = zeros(n_iter, length(z))
loss = []
for iter = 1:n_iter
    println(iter)

    train_loss, grads = Zygote.withgradient(z) do zp
        elbo(
            zp;
            q_family_array=q_family_array,
            log_joint=log_joint,
            n_samples=5
        )
    end

    push!(loss, train_loss)
    # z update
    Optimisers.update!(state, z, grads[1])
    # @. z += opt.eta * grads[1]
    z_trace[iter, :] = z
end

plot(z_trace)
plot(loss)

# Get VI distribution
q = get_variational_dist(z, q_family_array, range_z)
theta = rand_array(q; from_base_dist=false, reduce_to_vec=false)

log_joint(theta; y=y, X=X)

elbo(
    z;
    q_family_array=q_family_array,
    log_joint=log_joint,
    n_samples=10
)
