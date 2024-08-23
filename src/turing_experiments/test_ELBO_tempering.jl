# test ELBO with tempering
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections
using ProgressMeter

using StatsFuns
using Bijectors
using DiffResults
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack
using ADTypes
using Flux
using Zygote
using Turing
using AdvancedVI

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))
include(joinpath(abs_project_path, "utils", "mixed_models_data_generation.jl"))
include(joinpath(abs_project_path, "utils", "decayed_ada_grad.jl"))
include(joinpath(abs_project_path, "mixed_models", "relaxed_bernoulli.jl"))

p = 50

data_dict = generate_logistic_model_data(
    n_individuals=30, obs_noise_sd=0.1f0,
    p=p, p1=10, p0=p-10, corr_factor=0.5,
    random_seed=1
)

data_dict["beta"]

# Prior distributions
params_dict = OrderedDict()
num_params = 0

params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
num_params += p
log_prior_beta_fixed(beta_fixed) = sum(Distributions.logpdf.(
    Distributions.Normal(0f0, 1f0),
    beta_fixed
))


# Likelihood
function likelihood(;
    beta_fixed::AbstractArray{Float32},
    Xfix::AbstractArray{Float32}
    )
    lin_pred = Xfix * beta_fixed
    arraydist([Distributions.BernoulliLogit(logitp) for logitp in lin_pred])
end

function log_likelihood(;
    y::AbstractArray,
    beta_fixed::AbstractArray{Float32},
    Xfix::AbstractArray{Float32}
    )
    Distributions.logpdf(likelihood(
        beta_fixed=beta_fixed,
        Xfix=Xfix
    ), y)        
end

# Joint
params_names = tuple(Symbol.(params_dict.keys)...)
proto_array = ComponentArray(;
    [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn32(params_dict[pp]["size"]), randn32(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
)
proto_axes = getaxes(proto_array)
num_params = length(proto_array)


function log_joint(theta_hat::AbstractArray{Float32}; X_batch::AbstractArray{Float32}, y_batch::AbstractArray, temp=1f0)
    begin
        params_names = ComponentArray(theta_hat, proto_axes)
    end

    beta_fixed = params_names.beta_fixed

    loglik = log_likelihood(
        y=y_batch,
        beta_fixed=beta_fixed,
        Xfix=X_batch
    )

    log_prior = log_prior_beta_fixed(beta_fixed)
    
    (loglik + log_prior) / temp
end

# Variational Distribution
dim_q = num_params * 2
half_dim_q = num_params

function getq(theta::AbstractArray{Float32})
    Distributions.MultivariateNormal(
        theta[1:half_dim_q],
        StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
    )
end



# Define objective
variational_objective = Turing.Variational.ELBO()
# Optimizer
optimizer = DecayedADAGrad()

# VI algorithm
alg = AdvancedVI.ADVI(100, 100, adtype=ADTypes.AutoZygote())

theta = randn32(dim_q)

diff_results = DiffResults.GradientResult(theta)

X_batch = data_dict["X"]
y_batch = data_dict["y"]

# 1. Compute gradient and objective value; results are stored in `diff_results`
batch_log_joint(theta_hat::AbstractArray{Float32}) = log_joint(
    theta_hat,
    X_batch=X_batch,
    y_batch=y_batch,
    temp=10f0
)

AdvancedVI.grad!(
    variational_objective,
    alg,
    getq,
    batch_log_joint,
    theta,
    diff_results,
    100
)

# # 2. Extract gradient from `diff_result`
gradient_step = DiffResults.gradient(diff_results)

# # 3. Apply optimizer, e.g. multiplying by step-size
diff_grad = apply!(optimizer, theta, gradient_step)

# 4. Update parameters
@. theta = theta - diff_grad

# with true beta values
theta[1:p] = data_dict["beta"]
theta[p+1:2*p] .= 0.1f0

elbo_best = AdvancedVI.elbo(
    alg,
    getq(theta),
    batch_log_joint,
    1000
)

# with random values (same var)
theta[1:p] = randn32(p)
theta[p+1:2*p] .= 0.1f0

elbo_random = AdvancedVI.elbo(
    alg,
    getq(theta),
    batch_log_joint,
    1000
)

# Fix all but 2
theta[1:p-2] = data_dict["beta"][1:p-2]
theta[p+1:2*p] .= 0.01f0

theta_vec = Float32.(range(-4, 4, length=100))
theta_hat = deepcopy(theta)
loss_vec = zeros32(100, 100)

for tt1 = 1:100
    theta_hat[p-1] = theta_vec[tt1]
    for tt2 = 1:100
        theta_hat[p] = theta_vec[tt2]

        loss = AdvancedVI.elbo(
            alg,
            getq(theta_hat),
            batch_log_joint,
            100
        )
        
        loss_vec[tt1, tt2] = loss
    end
end

heatmap(loss_vec)


# Fix all but 2
theta[3:p] = data_dict["beta"][3:p]
theta[p+1:2*p] .= 0.1f0

theta[p] = 0f0

theta_vec = Float32.(range(-2, 2, length=100))
theta_hat = deepcopy(theta)
loss_vec = zeros32(100, 100)

for tt1 = 1:100
    theta_hat[1] = theta_vec[tt1]
    for tt2 = 1:100
        theta_hat[2] = theta_vec[tt2]

        loss = AdvancedVI.elbo(
            alg,
            getq(theta_hat),
            batch_log_joint,
            100
        )
        
        loss_vec[tt1, tt2] = loss
    end
end

heatmap(loss_vec)
