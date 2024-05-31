# SGLD with continuous Spike and Slab
using Flux
using Zygote
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra


# Simulate some simple regression data
n = 500
p = 100
p1 = Int(p * 0.5)
p0 = p - p1

Random.seed!(32143)
# First coefficient is the intercept
beta_true = vcat(zeros(p - p1), rand([1, -1.], p1))

sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)

X_train = Float32.(transpose(X))
y_train = Float32.(transpose(y))


struct SpikeSlab{M <: AbstractMatrix, P, F, B, L}
    weight::M
    alpha::M
    prior_weight::P
    prior_alpha::F
    bias::B
    lambda::L
    function SpikeSlab(
        W::M,
        alpha::M,
        prior_weight::P,
        prior_alpha::F,
        bias::B = true,
        lambda::L = 0.01f0
    ) where {M <: AbstractMatrix, P, F, B<:Union{Bool, AbstractArray}, L}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, P, F, typeof(b), L}(W, alpha, prior_weight, prior_alpha, b, lambda)
    end
end

SpikeSlab(
    (in, out)::Pair{<:Integer, <:Integer};
    init_weights=init_weights,
    init_alpha=init_alpha,
    prior_weight,
    prior_alpha,
    bias=true,
    lambda=0.01f0
) = SpikeSlab(
    init_weights(out, in),
    init_alpha(out, in),
    prior_weight,
    prior_alpha,
    bias,
    lambda
)
Flux.@functor SpikeSlab

function (m::SpikeSlab)(x::AbstractVecOrMat)
    random_gumbell = Distributions.rand(Distributions.Gumbel{Float32}(0f0, 1f0), size(m.alpha))

    alpha_sigmoid = Flux.sigmoid_fast((m.alpha .+ random_gumbell) ./ m.lambda)

    weight = m.weight .* alpha_sigmoid

    y_pred = weight * x .+ m.bias

    prior_neglogpdf = -sum(m.prior_weight(m.weight, alpha_sigmoid))

    return (y_pred, prior_neglogpdf)
end

function init_weights(in, out)
    0.01f0 * randn32(in, out)
end
function init_alpha(in, out)
    0.1f0 * randn32(in, out) .- 1f0
end

function polynomial_decay(t::Int64; a::Float32=0.01f0, b::Float32=0.01f0, gamma::Float32=0.35f0)
    a * (b + t)^(-gamma)
end
plot(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.1f0))
plot!(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.01f0))


function logpdf_loss(x::Array{Float32}, mu::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end


# Prior mixture with varying sd_spike
function logpdf_mixture_normal(
    x::Matrix{Float32},
    w::Matrix{Float32};
    mu::Vector{Float32}=Float32.([0, 0]),
    sd::Vector{Float32}=Float32.([0.001, 1])
    )

    w_ext = vcat(w, 1f0 .- w)
    xstd = -0.5f0 .* ((x .- mu) ./ sd).^2f0
    # wstd = w_ext ./ (sqrt(2f0 .* pi) .* sd)
    wstd = w_ext
    offset = maximum(xstd .* wstd, dims=1)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=1)
    log.(s) .+ offset
end

function logpdf_mixture_normal_delta0(
    x::Matrix{Float32},
    w::Matrix{Float32};
    sd::Float32=1f0
    )

    log.(w .* exp.(-0.5f0 .* (x ./ sd).^2f0) .+ (1f0 .- w) .+ Float32(1e-16))
end

logpdf_mixture_normal([1f0;; 0f0], [0.8f0;; 0.5f0])

logpdf_mixture_normal_delta0([0.01f0;; 0.01f0], [0.5f0;; 0.001f0])


# Define model
model = SpikeSlab(
    (p => 1);
    init_weights=init_weights,
    init_alpha=init_alpha,
    prior_weight=logpdf_mixture_normal_delta0,
    prior_alpha=Distributions.Uniform(0f0, 1f0),
    bias=true
)

optim = Flux.Descent(0.01f0)
use_sgld = true

# Train loop
n_iter = 20000

optim = Flux.setup(optim, model)

train_loss = Float32[]
val_loss = Float32[]

weights_mat = zeros32(n_iter, p)
alpha_mat = zeros32(n_iter, p)


for epoch in 1:n_iter

    loss, grads = Flux.withgradient(model) do m
        # Evaluate model and loss inside gradient context:
        y_pred, neg_logprior = m(X_train)
        neg_loglik = -sum(logpdf_loss(y_train, y_pred))

        neg_loglik + neg_logprior
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

    alpha_sigmoid = Flux.sigmoid_fast(model.alpha / model.lambda)
    weights_mat[epoch, :] = model.weight .* alpha_sigmoid
    alpha_mat[epoch, :] = alpha_sigmoid
end


plot(train_loss)
plot(train_loss[500:n_iter])

plot(weights_mat)
plot(weights_mat[500:n_iter, :])

plot(weights_mat[500:n_iter, 1])
density(weights_mat[500:n_iter, 1])

plot(weights_mat[500:n_iter, p])
density(weights_mat[500:n_iter, p])
sqrt.(var(weights_mat[500:n_iter, p]))

# alpha
plot(alpha_mat)
plot(alpha_mat[500:n_iter, :])

sum(mean(alpha_mat[Int(n_iter/2):n_iter, :] .> 0.5, dims=1) .> 0.5)

plot(weights_mat[500:n_iter, 2])
density(weights_mat[500:n_iter, 2])
