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

y = beta0_true .+ X * beta_true + randn(N)*0.5


function ordered_vector(y)
    L = length(y)
    cumsum(vcat(y[1], StatsFuns.softplus.(y[2:L])))
end

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
function log_likelihood(; y, X, beta, beta0, sigma_y)
    logpdf(
        MultivariateNormal(X*beta .+ beta0, sigma_y), y
    )
end


function logpdf_mixture_prior(y;
    mix_probs, mu, sd=1.
    )

    xstd = -0.5f0 .* ((y .- mu) ./ sd).^2f0
    wstd = mix_probs ./ (sqrt(2f0 .* Float32(pi)) .* sd)
    offset = maximum(xstd .* wstd)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd)
    sum(log.(s) .+ offset)
end

# Joint
mix_probs_from = 1
mix_probs_to = K - 1

clusters_mean_from = mix_probs_to + 1
clusters_mean_to = clusters_mean_from + K - 1

sd_clusters_from = clusters_mean_to + 1
sd_clusters_to = sd_clusters_from

sigma_y_from = sd_clusters_to + 1
sigma_y_to = sigma_y_from

beta0_from = sigma_y_to + 1
beta0_to = beta0_from + N - 1

beta_from = beta0_to + 1
beta_to = beta_from + p - 1

num_params = beta_to


function log_joint(theta_hat)

    mix_probs = StatsFuns.softmax(vcat(theta_hat[mix_probs_from:mix_probs_to], 0.))
    clusters_mean = ordered_vector(theta_hat[clusters_mean_from:clusters_mean_to])

    sd_cluster = StatsFuns.softplus(theta_hat[sd_clusters_from])

    sigma_y = StatsFuns.softplus(theta_hat[sigma_y_from])

    beta0 = theta_hat[beta0_from:beta0_to]
    beta = theta_hat[beta_from:beta_to]

    loglik = log_likelihood(
        y=y, X=X, beta=beta, beta0=beta0, sigma_y=sigma_y
    )

    log_prior = logpdf(MvNormal(zeros(K), I * 10.), clusters_mean) +
        logpdf(Dirichlet(K, 5.0), mix_probs) +
        sum(logpdf_mixture_prior.(beta0, mix_probs=mix_probs, mu=clusters_mean, sd=sd_cluster)) +
        logpdf(truncated(Normal(0, 0.1), 0, Inf), sd_cluster) + 
        logpdf(truncated(Normal(0, 1), 0, Inf), sigma_y)
        
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

num_steps = 2000
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

    # theta_trace[step, 1:K-1] = q_temp.μ[1:K-1]
    # theta_trace[step, K:K-1+K] = ordered_vector(q_temp.μ[K:K-1+K])
    # theta_trace[step, n_beta0_from:n_beta0_to] = q_temp.μ[n_beta0_from:n_beta0_to]
    # theta_trace[step, n_beta_from:n_beta_to] = q_temp.μ[n_beta_from:n_beta_to]

    elbo_trace[step] = AdvancedVI.elbo(alg, q_temp, log_joint, samples_per_step)

    step += 1

    ProgressMeter.next!(prog)
end

q = getq(theta)

plot(elbo_trace, label="ELBO")
plot(elbo_trace[300:num_steps, :], label="ELBO")

# plot(theta_trace)

samples = rand(q, 2000)


density(StatsFuns.softmax(vcat(samples[mix_probs_from:mix_probs_to, :], zeros(1, 2000)), dims=1)')
mean(StatsFuns.softmax(vcat(samples[mix_probs_from:mix_probs_to, :], zeros(1, 2000)), dims=1)', dims=1)

density(ordered_vector_matrix(samples[clusters_mean_from:clusters_mean_to, :])')
mean(ordered_vector_matrix(samples[clusters_mean_from:clusters_mean_to, :])', dims=1)

density(StatsFuns.softplus.(samples[sigma_y_from, :]))
density(StatsFuns.softplus.(samples[sd_clusters_from, :]))
density(samples[beta0_from:beta0_to, :]')
density(samples[beta_from:beta_to, :]')


function logpdf_mixture(;
    mix_probs, mu, y, sd
    )

    xstd = -0.5f0 .* ((y .- mu) ./ sd).^2f0
    wstd = mix_probs ./ (sqrt(2f0 .* Float32(pi)) .* sd)
    offset = maximum(xstd .* wstd)
    xe = exp.(xstd .- offset)
    s = xe .* wstd
    log.(s) .+ offset
end

clusters_mean = ordered_vector(q.μ[clusters_mean_from:clusters_mean_to])

clusters_mean = ordered_vector_matrix(samples[clusters_mean_from:clusters_mean_to, :])
beta0 = samples[beta0_from:beta0_to, :]
sd_clusters = StatsFuns.softplus.(samples[sd_clusters_from, :])
mix_probs = StatsFuns.softmax(vcat(samples[mix_probs_from:mix_probs_to, :], zeros(1, 2000)), dims=1)

samples_class = zeros(N, 2000)
modal_class = zeros(N)
for j = 1:N
    for mc = 1:2000
        log_probs = logpdf_mixture(
            mix_probs=mix_probs[:, mc],
            mu=clusters_mean[:, mc],
            y=beta0[j, mc],
            sd=sd_clusters[mc]
        )
        samples_class[j, mc] = rand(Categorical(StatsFuns.softmax(log_probs)))
    end
    modal_class[j] = mode(samples_class[j, :])
end

frequencies(modal_class)
frequencies(samples_class[3, :])


