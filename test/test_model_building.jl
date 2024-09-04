# Test model building

using OrderedCollections
using Distributions
using ComponentArrays, UnPack


abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))

# Normal model with parameters:
# beta0
# beta
# sigma_y

p = 5
n = 10
X = Float32.(randn(n, p))
y = Float32.(randn(n))

params_dict = OrderedDict()

update_parameters_dict(
    params_dict;
    name="beta0",
    size=1,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)

update_parameters_dict(
    params_dict;
    name="beta",
    size=p,
    log_prob_fun=x::AbstractArray{Float32} -> sum(DistributionsLogPdf.log_normal(x))
)

update_parameters_dict(
    params_dict;
    name="sigma_y",
    size=1,
    bij=exp,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)

params_dict["priors"]
params_dict["priors"]["beta0"]["log_prob"](1f0)
params_dict["priors"]["beta"]["log_prob"]([1f0, 0f0, 3f0, 1f0, 0f0])

# parameters extraction
priors_dict = params_dict["priors"]
theta_components = tuple(Symbol.(priors_dict.keys)...)
proto_array = ComponentArray(;
    [Symbol(pp) => ifelse(priors_dict[pp]["size"] > 1, ones(priors_dict[pp]["size"]), ones(priors_dict[pp]["size"])[1])  for pp in priors_dict.keys]...
)
theta_axes = getaxes(proto_array)

theta = Float32.(randn(params_dict["tot_params"]))

begin
    theta_components = ComponentArray(theta, theta_axes)
end

# predictions
predictions = Predictors.linear_model(
    theta_components,
    priors_dict=priors_dict,
    X=X
)

# log likelihood
DistributionsLogPdf.log_normal(y, predictions[1], predictions[2])
loglik = DistributionsLogPdf.log_normal(y, predictions...)

# log prior
log_prior = 0f0
for prior in keys(priors_dict)
    log_prior += priors_dict[prior]["log_prob"](
        priors_dict[prior]["bij"](theta_components[prior])
    )
end

log_prior + sum(loglik)

log_joint(
    theta;
    priors_dict=priors_dict,
    theta_axes=theta_axes,
    predictor=Predictors.linear_model,
    log_likelihood=DistributionsLogPdf.log_normal,
    input=X,
    label=y
)

partial_log_joint(theta) = log_joint(
    theta;
    priors_dict=priors_dict,
    theta_axes=theta_axes,
    predictor=Predictors.linear_model,
    log_likelihood=DistributionsLogPdf.log_normal,
    input=X,
    label=y
)
partial_log_joint(theta)

z = Float32.(randn(params_dict["tot_params"]*2))
vi_dist(z) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])
vi_dist(z)

res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z=z,
    n_iter=10,
    n_chains=1,
    samples_per_step=4
)

