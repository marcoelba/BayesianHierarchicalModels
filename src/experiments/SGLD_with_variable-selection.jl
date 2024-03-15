# SGLD with variable selection
using Flux
using Zygote
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra


# Simulate some simple regression data
n = 300
p = 300

prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)

Random.seed!(32143)
beta_true = vcat(zeros(p - p1), rand([1, -1.], p1))
sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)

X_train = Float32.(transpose(X))
y_train = Float32.(transpose(y))


# Non-Centred Horseshoe model
struct NC_HS{M <: AbstractMatrix, P_W, P_S, B}
    weight::M
    scale::M
    prior_weight::P_W
    prior_scale::P_S
    bias::B
    function NC_HS(
        W::M,
        S::M,
        prior_weight::P_W,
        prior_scale::P_S,
        bias::B = true
    ) where {M <: AbstractMatrix,  P_W, P_S, B<:Union{Bool, AbstractArray}}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M,  P_W, P_S, typeof(b)}(W, S, prior_weight, prior_scale, b)
    end
end

NC_HS(
    (in, out)::Pair{<:Integer, <:Integer};
    prior_weight,
    prior_scale,
    init_weights=init_weights,
    init_scales=init_scales,
    bias=true
) = NC_HS(
    init_weights(out, in),
    init_scales(out, in),
    prior_weight,
    prior_scale,
    bias
)
Flux.@functor NC_HS

function (m::NC_HS)(x::AbstractVecOrMat)
    # scale must be positive
    scale_t = Flux.softplus.(m.scale)
    beta = m.weight .* scale_t

    y_pred = beta * x .+ m.bias
    neg_log_penalty = -sum(m.prior_weight(m.weight)) - sum(m.prior_scale(scale_t))

    return (y_pred, neg_log_penalty)
end


function init_weights(in, out)
    0.01f0 * randn32(in, out)
end

function init_scales(in, out)
    0.1f0 * randn32(in, out)
end

function logpdf_truncated_cauchy(x::Array{Float32}, t::Float32=0.5f0)
    -log.(pi .* (1f0 .+ x.^2)) .- log.(t)
end

function logpdf_cauchy(x::Array{Float32})
    -log.(pi .* (1f0 .+ x.^2))
end

function logpdf_normal_prior(x::Array{Float32}, mu::Float32=0f0, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end

function polynomial_decay(t::Int64; a::Float32=0.01f0, b::Float32=0.01f0, gamma::Float32=0.35f0)
    a * (b + t)^(-gamma)
end
plot(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.1f0))
plot!(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.01f0))


function logpdf_normal(x::Array{Float32}, mu::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end


# Define model
model = NC_HS(
    (p => 1);
    prior_weight=logpdf_normal_prior,
    prior_scale=logpdf_truncated_cauchy
)
Flux.params(model)

optim = Flux.Descent(0.01f0)
use_sgld = false

# Train loop
n_iter = 10000

optim = Flux.setup(optim, model)

train_loss = Float32[]
val_loss = Float32[]

weights_mat = zeros32(n_iter, p)


for epoch in 1:n_iter

    loss, grads = Flux.withgradient(model) do m
        # Evaluate model and loss inside gradient context:
        y_pred = m(X_train)
        neg_loglik = -sum(logpdf_loss(y_train, y_pred))
        neg_logprior = -sum(logpdf_prior(m.weight))

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

    weights_mat[epoch, :] = model.weight
end

plot(train_loss)
plot(train_loss[100:n_iter])

plot(weights_mat)

plot(weights_mat[:, 1])
density(weights_mat[5000:n_iter, 1])
