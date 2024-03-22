# Spike and Slab model with Concrete Distribution
using Flux
using Zygote
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra


# Simulate some simple regression data
n = 100
p = 10
p1 = 3

Random.seed!(32143)
# First coefficient is the intercept
beta_true = vcat(zeros(p - p1), rand([1.5, -1.], p1))

sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)*0.5

X_train = Float32.(transpose(X))
y_train = Float32.(transpose(y))


# Model definition
struct SpikeSlab{M <: AbstractMatrix, P_W, P_G, P_P, B}
    weight::M
    gamma::M
    mix_prob::M
    prior_weight::P_W
    prior_gamma::P_G
    prior_probs::P_P
    bias::B
    function SpikeSlab(
        W::M,
        gamma::M,
        mix_prob::M,
        prior_weight::P_W,
        prior_gamma::P_G,
        prior_probs::P_P,   
        bias::B = true
    ) where {M <: AbstractMatrix, P_W, P_G, P_P, B<:Union{Bool, AbstractArray}}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, P_W, P_G, P_P, typeof(b)}(W, gamma, mix_prob, prior_weight, prior_gamma, prior_probs, b)
    end
end

SpikeSlab(
    (in, out)::Pair{<:Integer, <:Integer};
    prior_weight,
    prior_gamma,
    prior_probs,
    init_weights=init_weights,
    init_gamma=init_gamma,
    init_mix_probs=init_mix_probs,
    bias=true
) = SpikeSlab(
    init_weights(out, in),
    init_gamma(out, in),
    init_mix_probs(out, in),
    prior_weight,
    prior_gamma,
    prior_probs,
    bias
)
Flux.@functor SpikeSlab

function (m::SpikeSlab)(x::AbstractVecOrMat)
    gamma_sigmoid = Flux.sigmoid_fast(m.gamma)
    beta = m.weight .* gamma_sigmoid
    mix_prob_sigmoid = Flux.sigmoid.(m.mix_prob)

    y_pred = beta * x .+ m.bias

    neg_logprior = -sum(m.prior_weight(beta, gamma_sigmoid)) -
        sum(m.prior_gamma(gamma_sigmoid, mix_prob_sigmoid)) -
        sum(m.prior_probs(mix_prob_sigmoid))

    return (y_pred, neg_logprior)
end


"""
    log-pdf Spike and Slab distribution with a point mass at 0
"""
function pdf_gaussian(x::Array{Float32}, sd::Float32=1f0)
    xstd = x ./ sd
    1f0 / (sqrt(2f0*pi) * sd) .* exp.(-0.5f0 * (xstd .* xstd))
end


function logpdf_spike_slab(
    x::Array{Float32},
    w::Array{Float32};
    spike_tol::Float32=Float32(1e-8),
    sd_slab::Float32=5f0
    )

    log.(w .* pdf_gaussian(x, sd_slab) .+ (1f0 .- w) .+ spike_tol)
end
logpdf_spike_slab([2f0], [0.9f0], sd_slab=1f0)
logpdf_spike_slab([2f0], [0.01f0], sd_slab=1f0)

logpdf_spike_slab([0.1f0], [0.9f0], sd_slab=1f0)
logpdf_spike_slab([0.1f0], [0.01f0], sd_slab=1f0)


plot(truncated(Normal(0, 0.25), 0, 1))
plot!(truncated(Normal(1, 0.25), 0, 1))

cdf(Normal(0, 0.25), 1) - cdf(Normal(0, 0.25), 0)
cdf(Normal(1, 0.25), 1) - cdf(Normal(1, 0.25), 0)

function logpdf_truncated_mixture_normal(
    x::Matrix{Float32};
    w::Vector{Float32}=Float32.([0.5, 0.5]),
    mu::Vector{Float32}=Float32.([0, 1]),
    sd::Vector{Float32}=Float32.([0.25, 0.25]),
    t::Vector{Float32}=Float32.([0.5, 0.5])
    )
    xstd = -0.5f0 .* ((x .- mu) ./ sd).^2f0
    wstd = w ./ (sqrt(2f0 .* pi) .* sd) ./ t
    offset = maximum(xstd .* wstd, dims=1)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=1)
    log.(s) .+ offset
end


function logpdf_unif(x::Matrix{Float32})
    -0f0
end


# Bernoulli(alpha) ~= CD(ni, delta, alpha)
# ni ~ Unif(0, 1)

function logpdf_concrete_distribution(
    x::Array{Float32}, prob::Array{Float32}
    )
    x .* log.(prob) .+ (1f0 .- x) .* log.(1f0 .- prob)
end
logpdf_concrete_distribution([0f0, 1f0], [0.5f0, 0.1f0])
logpdf_concrete_distribution([0.1f0, 0.9f0], [0.5f0, 0.1f0])

function logpdf_normal_prior(x::Array{Float32}, mu::Float32=0f0, sd::Float32=5f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end


function init_weights(in, out)
    0.01f0 * randn32(in, out)
end

function init_gamma(in, out)
    0.1f0 * randn32(in, out)
end

function init_mix_probs(in, out)
    0.1f0 * randn32(in, out)
end


function polynomial_decay(t::Int64; a::Float32=0.05f0, b::Float32=0.01f0, gamma::Float32=0.2f0)
    a * (b + t)^(-gamma)
end
plot(range(1, 10000), polynomial_decay.(range(1, 10000)))


function logpdf_loss(x::Array{Float32}, mu::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end


# Define model
model = SpikeSlab(
    (p => 1);
    prior_weight=logpdf_spike_slab,
    prior_gamma=logpdf_concrete_distribution,
    prior_probs=logpdf_unif,
    init_weights=init_weights,
    init_gamma=init_gamma,
    init_mix_probs=init_mix_probs,
    bias=true
)

optim = Flux.Descent(0.01f0)
use_sgld = true

# Train loop
n_iter = 30000

optim = Flux.setup(optim, model)

train_loss = Float32[]
val_loss = Float32[]

weights_mat = zeros32(n_iter, p)
gamma_mat = zeros32(n_iter, p)
mix_probs_mat = zeros32(n_iter, p)

mix_prob_sigmoid = ones32(size(model.gamma)...) .* 0.5f0


for epoch in 1:n_iter

    loss, grads = Flux.withgradient(model) do m
        # Evaluate model and loss inside gradient context:
        model_pred = m(X_train)
        neg_loglik = -sum(logpdf_loss(y_train, model_pred[1]))

        # gamma_sigmoid = Flux.sigmoid.(m.gamma)
        # mix_prob_sigmoid = Flux.sigmoid.(m.mix_prob)
        # neg_logprior = -sum(logpdf_spike_slab(m.weight .* gamma_sigmoid, gamma_sigmoid)) -
        #     sum(logpdf_concrete_distribution(gamma_sigmoid, mix_prob_sigmoid))
        #     sum(logpdf_truncated_mixture_normal(mix_prob_sigmoid))

        neg_loglik + model_pred[2]
    end

    # Update learning rate with Polynomial Stepsize decay
    lt = polynomial_decay(epoch)
    # optim.eta = 0.5f0 * lt
    Flux.adjust!(optim, 0.5f0 * lt)

    Flux.update!(optim, model, grads[1])

    if use_sgld
        for par in Flux.params(model)
            par .-= lt .* randn32(size(par)...)
        end
    end

    push!(train_loss, loss)  # logging, outside gradient context

    weights_mat[epoch, :] = model.weight
    gamma_mat[epoch, :] = model.gamma
    mix_probs_mat[epoch, :] = model.mix_prob

end


plot(train_loss)
plot(train_loss[100:n_iter])

beta = weights_mat .* Flux.sigmoid.(gamma_mat)
plot(weights_mat)
plot(beta)

plot(Flux.sigmoid.(gamma_mat))
plot(gamma_mat)
plot(Flux.sigmoid.(mix_probs_mat))

plot(beta[:, 1])
density(beta[5000:n_iter, 1])
density(beta[5000:n_iter, p])

gamma_posterior = Flux.sigmoid.(gamma_mat)[10000:n_iter, :]
mean(gamma_posterior, dims=1) .> mean(mean(gamma_posterior, dims=1))
