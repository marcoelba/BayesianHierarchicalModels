# SGLD
using Flux
using Zygote
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra


# Simulate some simple regression data
n = 100
p = 5

Random.seed!(32143)
# First coefficient is the intercept
beta_true = rand([1.5, -1.], p)

sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)*0.5

X_train = Float32.(transpose(X))
y_train = Float32.(transpose(y))


struct LinModel{M <: AbstractMatrix, B}
    weight::M
    bias::B
    function LinModel(
        W::M,
        bias::B = true
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b)}(W, b)
    end
end

LinModel(
    (in, out)::Pair{<:Integer, <:Integer};
    init_weights=init_weights,
    bias=true
) = LinModel(
    init_weights(out, in),
    bias
)
Flux.@functor LinModel

function (m::LinModel)(x::AbstractVecOrMat)
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


# Define model
model = LinModel(
    (p => 1);
    init_weights=init_weights,
    bias=true
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
