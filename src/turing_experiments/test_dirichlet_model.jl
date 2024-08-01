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

include(joinpath("decayed_ada_grad.jl"))
include(joinpath("mixed_models_data_generation.jl"))
include(joinpath("mirror_statistic.jl"))
include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("../mixed_models/relaxed_bernoulli.jl"))
include(joinpath("plot_functions.jl"))
include(joinpath("../utils/classification_metrics.jl"))


# Define Gaussian mixture model.
w = [0.25, 0.5, 0.25]
μ = [-3.5, 0.5, 3.5]
mixturemodel = MixtureModel([Normal(μₖ, 0.5) for μₖ in μ], w)

# We draw the data points.
N = 100
x = rand(mixturemodel, N)
density(x)

@model function gmm_marginalized(x)
    K = 3
    μ ~ Bijectors.ordered(MvNormal(zeros(K), I))
    w ~ Dirichlet(K, 1.0)

    x ~ MixtureModel([Normal(μₖ, 1.) for μₖ in μ], w)
end

model = gmm_marginalized(x)

bij = Bijectors.bijector(model)
inv_bij = Bijectors.inverse(bij)

sampler = NUTS()
chains = sample(model, sampler, 2000)

plot(chains, legend=true)


# Simplex bijector
using LogExpFunctions

y = [1., -0.5]
x = zeros(3)

K = length(y) + 1
@assert K > 1 "x needs to be of length greater than 1"
T = eltype(y)
ϵ = Bijectors._eps(T)
z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
x[1] = Bijectors._clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
sum_tmp = zero(T)
for k in 2:(K - 1)
    z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
    sum_tmp += x[k - 1]
    x[k] = Bijectors._clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
end
sum_tmp += x[K - 1]
x[K] = Bijectors._clamp(one(T) - sum_tmp, 0, 1)
sum(x)


# Ordered bijector
yt = [-1., -2., -1.99]

function ordered_vector(y)
    L = length(y)
    cumsum(vcat(y[1], StatsFuns.softplus.(y[2:L])))
end
ordered_vector(yt)


function ordered_vector_matrix(y)
    L = size(y, 1)
    X = similar(y)
    for i = 1:size(y, 2)
        X[:, i] = cumsum(vcat(y[1, i], StatsFuns.softplus.(y[2:L, i])))
    end
    return X
end

# Manual

# priors
K = 3

# Likelihood
function log_likelihood(; y, mu, mix_probs)
    sum(
        Distributions.logpdf.(
            MixtureModel([Normal(μₖ, 1.) for μₖ in mu], mix_probs),
            y
        )
    )
end

log_likelihood(y=x, mu=μ, mix_probs=w)


function log_likelihood(;
    mix_probs, mu, y, sd=0.5
    )
    w_ext = transpose(mix_probs)
    mu = transpose(mu)

    xstd = -0.5f0 .* ((y .- mu) ./ sd).^2f0
    wstd = w_ext ./ (sqrt(2f0 .* Float32(pi)) .* sd)
    offset = maximum(xstd .* wstd, dims=2)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=2)
    sum(log.(s) .+ offset)
end

log_likelihood(y=x, mu=μ, mix_probs=w, sd=1.)

# Joint
num_params = 5

function log_joint(theta_hat)

    mix_probs = StatsFuns.softmax(vcat(theta_hat[1:2], 0.))
    clusters_mean = ordered_vector(theta_hat[3:5])
    # sd = StatsFuns.softplus(theta_hat[6])

    loglik = log_likelihood(
        y=x, mu=clusters_mean, mix_probs=mix_probs, sd=1.
    )

    log_prior = logpdf(MvNormal(zeros(K), I), clusters_mean) +
        logpdf(Dirichlet(K, 1.0), mix_probs)
        # logpdf(truncated(Normal(0, 0.5), 0, Inf), sd)
        
    loglik + log_prior
end
theta_hat = ones32(num_params) * 0.5f0
log_joint(theta_hat)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
dim_q = num_params * 2
half_dim_q = num_params

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_dim_q],
        StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
    )
end

q = getq(ones(num_params * 2))
rand(q)

num_steps = 1000
samples_per_step = 5

elbo_trace = zeros32(num_steps)
theta_trace = zeros32(num_steps, num_params)

# Define objective
variational_objective = Turing.Variational.ELBO()

# Optimizer
optimizer = DecayedADAGrad()

# VI algorithm
alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

# --- Train loop ---
converged = false
step = 1
theta = randn32(dim_q) * 0.5f0

prog = ProgressMeter.Progress(num_steps, 1)
diff_results = DiffResults.GradientResult(theta)

while (step ≤ num_steps) && !converged
    # 1. Compute gradient and objective value; results are stored in `diff_results`
    AdvancedVI.grad!(variational_objective, alg, getq, log_joint, theta, diff_results, samples_per_step)

    # # 2. Extract gradient from `diff_result`
    gradient_step = DiffResults.gradient(diff_results)

    # # 3. Apply optimizer, e.g. multiplying by step-size
    diff_grad = apply!(optimizer, theta, gradient_step)

    # 4. Update parameters
    @. theta = theta - diff_grad

    # 5. Do whatever analysis you want - Store ELBO value
    q_temp = getq(theta)
    sample_t = rand(q_temp)
    
    theta_trace[step, 1:2] = StatsFuns.softmax(sample_t[1:2])
    theta_trace[step, 3:5] = sample_t[3:5]
    # theta_trace[step, 6] = StatsFuns.softplus(sample_t[6])

    elbo_trace[step] = AdvancedVI.elbo(alg, q_temp, log_joint, samples_per_step)

    step += 1

    ProgressMeter.next!(prog)
end

q = getq(theta)

plot(elbo_trace, label="ELBO")
plot(elbo_trace[300:num_steps, :], label="ELBO")

plot(theta_trace)

samples = rand(q, 2000)

density(ordered_vector_matrix(samples[3:5, :])')

density(StatsFuns.softmax(vcat(samples[1:2, :], zeros(1, 2000)), dims=1)')
