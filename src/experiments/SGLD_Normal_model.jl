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
p = 10
p1 = 5
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


struct SpikeSlab{M <: AbstractMatrix, B, P <: Distributions.Distribution}
    weight::M
    bias::B
    prior_weight::P
    function SpikeSlab(
        W::M,
        bias::B = true,
        prior_weight::P = Distributions.Normal(0f0, 5f0)
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}, P}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b), P}(W, b, prior_weight)
    end
end

SpikeSlab(
    (in, out)::Pair{<:Integer, <:Integer};
    init_weights=init_weights,
    bias=true,
    prior_weight=Distributions.Normal(0f0, 5f0)
) = SpikeSlab(
    init_weights(out, in),
    bias,
    prior_weight
)
Flux.@functor SpikeSlab

function (m::SpikeSlab)(x::AbstractVecOrMat)
    y_pred = m.weight * x .+ m.bias
    return y_pred
end

function init_weights(in, out)
    0.01f0 * randn32(in, out)
end

function polynomial_decay(t::Int64; a::Float32=0.01f0, b::Float32=0.01f0, gamma::Float32=0.35f0)
    a * (b + t)^(-gamma)
end
plot(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.1f0))
plot!(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.01f0))


function logpdf_loss(x::Array{Float32}, mu::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end

function logpdf_prior(x::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* (x ./ sd).^2f0
end

prior_weight = Distributions.Normal(0f0, 5f0)


# Define model
model = SpikeSlab(
    (p => 1);
    init_weights=init_weights,
    bias=true,
    prior_weight=prior_weight
)

optim = Flux.Descent(0.01f0)
use_sgld = true

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
        neg_logprior = -sum(Distributions.logpdf.(prior_weight, m.weight))
        # neg_logprior = -sum(logpdf_prior(m.weight))

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
plot(weights_mat[100:n_iter, :])

plot(weights_mat[500:n_iter, 1])
density(weights_mat[500:n_iter, 1])

plot(weights_mat[500:n_iter, p])
density(weights_mat[500:n_iter, p])
