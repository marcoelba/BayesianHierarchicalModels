# Test on linear model
using OrderedCollections
using Distributions
using ComponentArrays, UnPack

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 200
n_time_points = 5

p = 10
prop_non_zero = 0.5
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5
beta_time=Float32.([0, 2, 0, 0, 0])
beta_pool=Float32.([-1., 1])

data_dict = generate_mixed_model_data(;
    n_individuals=n_individuals, n_time_points=n_time_points,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor, beta_pool=beta_pool,
    include_random_int=true, random_int_from_pool=false, random_intercept_sd=0.5,
    include_random_time=true, beta_time=beta_time,
    include_random_slope=false, random_seed=12
)

n_chains = 2
num_iter = 2000

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0_fixed",
    size=1,
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)
update_parameters_dict(
    params_dict;
    name="sigma_beta0",
    size=n_individuals,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, 0.1f0), 0f0, Inf32),
        x
    ))
)
update_parameters_dict(
    params_dict;
    name="beta0_random",
    size=n_individuals,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    size=p,
    bij=StatsFuns.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, 1f0), 0f0, Inf32),
        x
    ))
)
update_parameters_dict(
    params_dict;
    name="beta_fixed",
    size=p,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma)
)

# beta time
update_parameters_dict(
    params_dict;
    name="sigma_beta_time",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)
update_parameters_dict(
    params_dict;
    name="beta_time",
    size=n_time_points,
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma)
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    size=1,
    bij=StatsFuns.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)

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

# model predictions
model(theta_components, priors_dict) = Predictors.linear_random_intercept_model(
    theta_components,
    priors_dict;
    Xfix=data_dict["Xfix"],
    n_time_points=n_time_points
)
preds = model(
    theta_components,
    priors_dict
)

# Likelihood
sum(DistributionsLogPdf.log_normal(data_dict["y"], preds[1], preds[2]))
sum(DistributionsLogPdf.log_normal(data_dict["y"], preds...))


# model
partial_log_joint(theta) = log_joint(
    theta;
    priors_dict=priors_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)
partial_log_joint(Float32.(randn(params_dict["tot_params"])))

# VI distribution
vi_dist(z::AbstractArray) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])
z_init = Float32.(randn(params_dict["tot_params"]*2))

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z=z_init,
    n_iter=num_iter,
    n_chains=1,
    samples_per_step=4
)
