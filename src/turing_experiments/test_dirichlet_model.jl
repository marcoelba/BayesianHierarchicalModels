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


# Define Gaussian mixture model
N = 300
p = 2
beta0_true = vcat(ones(100) * -2., ones(100) * 0., ones(100) * 2.)
beta_true = [1., -1.]

X = randn(N, p)

y = beta0_true .+ X * beta_true + randn(N)


@model function gmm_marginalized(y)
    K = 3

    mu ~ Bijectors.ordered(MvNormal(zeros(K), I))
    mix_probs ~ Dirichlet(K, 1.0)

    beta0 ~ filldist(
        MixtureModel([Normal(mu_k, 0.5) for mu_k in mu], mix_probs),
        N
    )

    beta ~ MultivariateNormal(zeros(p), 1.)

    y ~ MultivariateNormal(beta0 .+ X * beta, 1.)
end

model = gmm_marginalized(y)

bij = Bijectors.bijector(model)
inv_bij = Bijectors.inverse(bij)

sampler = NUTS()
chains = sample(model, sampler, 1000)

plot(chains[["mu[1]", "mu[2]", "mu[3]"]], legend=true)
plot(chains[["mix_probs[1]", "mix_probs[2]", "mix_probs[3]"]], legend=true)
plot(chains[["beta0[1]", "beta0[199]", "beta0[300]"]], legend=true)


# Simplex bijector
y = Float32.([-35, 45, -35, -35])
StatsFuns.softmax(vcat(y, 0f0))

using LogExpFunctions

x = zeros32(length(y) + 1)

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
function log_likelihood(; y, X, beta, beta0)
    logpdf(
        MultivariateNormal(X*beta .+ beta0, 1.), y
    )
end

# log_likelihood(y=y, X=X, beta=beta, beta0=beta0)


function logpdf_mixture_prior(;
    mix_probs, mu, y, sd=1.
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


# Joint
num_params = N + p + K + K - 1

n_beta0_from = K+K
n_beta0_to = n_beta0_from + N - 1
n_beta_from = n_beta0_to + 1
n_beta_to = n_beta_from + p - 1

function log_joint(theta_hat)

    mix_probs = StatsFuns.softmax(vcat(theta_hat[1:K-1], 0.))
    clusters_mean = ordered_vector(theta_hat[K:K-1+K])

    # sd = StatsFuns.softplus(theta_hat[6])

    beta0 = theta_hat[n_beta0_from:n_beta0_to]
    beta = theta_hat[n_beta_from:n_beta_to]

    loglik = log_likelihood(
        y=y, X=X, beta=beta, beta0=beta0
    )

    log_prior = logpdf(MvNormal(zeros(K), I), clusters_mean) +
        logpdf(Dirichlet(K, 1.0), mix_probs) +
        logpdf_mixture_prior(mix_probs=mix_probs, mu=clusters_mean, y=beta0, sd=0.5)
        # logpdf(truncated(Normal(0, 0.1), 0, Inf), sd)
        
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
theta_trace = zeros32(num_steps, num_params+1)

# Define objective
variational_objective = Turing.Variational.ELBO()

# Optimizer
optimizer = DecayedADAGrad()

# VI algorithm
alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

# --- Train loop ---
converged = false
step = 1
theta = randn32(dim_q) * 0.2f0

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

    theta_trace[step, 1:K-1] = q_temp.μ[1:K-1]
    theta_trace[step, K:K-1+K] = ordered_vector(q_temp.μ[K:K-1+K])
    theta_trace[step, n_beta0_from:n_beta0_to] = q_temp.μ[n_beta0_from:n_beta0_to]
    theta_trace[step, n_beta_from:n_beta_to] = q_temp.μ[n_beta_from:n_beta_to]

    elbo_trace[step] = AdvancedVI.elbo(alg, q_temp, log_joint, samples_per_step)

    step += 1

    ProgressMeter.next!(prog)
end

q = getq(theta)

plot(elbo_trace, label="ELBO")
plot(elbo_trace[300:num_steps, :], label="ELBO")

plot(theta_trace)

samples = rand(q, 2000)


density(StatsFuns.softmax(vcat(samples[1:K-1, :], zeros(1, 2000)), dims=1)')
density(ordered_vector_matrix(samples[K:K-1+K, :])')
density(samples[n_beta0_from:n_beta0_to, :]')
density(samples[n_beta_from:n_beta_to, :]')
