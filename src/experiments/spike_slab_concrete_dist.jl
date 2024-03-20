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
struct SpikeSlab{M <: AbstractMatrix, B}
    weight::M
    gamma::M
    bias::B
    function SpikeSlab(
        W::M,
        gamma::M,
        bias::B = true
    ) where {M <: AbstractMatrix, B<:Union{Bool, AbstractArray}}
        b = Flux.create_bias(W, bias, size(W, 1))
        new{M, typeof(b)}(W, gamma, b)
    end
end

SpikeSlab(
    (in, out)::Pair{<:Integer, <:Integer};
    init_weights=init_weights,
    init_gamma=init_gamma,
    bias=true
) = SpikeSlab(
    init_weights(out, in),
    init_gamma(out, in),
    bias
)
Flux.@functor SpikeSlab

function (m::SpikeSlab)(x::AbstractVecOrMat)
    y_pred = m.weight * x .+ m.bias
    return y_pred
end


"""
    log-pdf Spike and Slab distribution with a point mass at 0
"""
function pdf_gaussian(x::Array{Float32}, sd::Float32=1f0)
    xstd = x ./ sd
    1f0 / (sqrt(2f0*pi) * sd) .* exp.(-0.5f0 * (xstd .* xstd))
end
pdf_gaussian([1f0, -3f0], 0.5f0)


function logpdf_spike_slab(
    x::Array{Float32},
    w::Array{Float32};
    spike_tol::Float32=Float32(1e-10),
    sd_slab::Float32=1f0
    )

    log.(w .* pdf_gaussian(x, sd_slab) .+ (1f0 .- w) .+ spike_tol)
end
logpdf_spike_slab([0.1f0], [0.1f0], sd_slab=1f0)


function init_weights(in, out)
    0.01f0 * randn32(in, out)
end

function init_gamma(in, out)
    0.1f0 * randn32(in, out)
end


function polynomial_decay(t::Int64; a::Float32=0.01f0, b::Float32=0.01f0, gamma::Float32=0.35f0)
    a * (b + t)^(-gamma)
end
plot(range(1, 1000), polynomial_decay.(range(1, 1000), gamma=0.5f0, a=0.1f0))


function logpdf_loss(x::Array{Float32}, mu::Array{Float32}, sd::Float32=1f0)
    -0.5f0 .* log(2f0*pi) - log(sd) .- 0.5f0 .* ((x .- mu) ./ sd).^2f0
end


# Define model
model = SpikeSlab(
    (p => 1);
    init_weights=init_weights,
    init_gamma=init_gamma,
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
gamma_mat = zeros32(n_iter, p)


for epoch in 1:n_iter

    loss, grads = Flux.withgradient(model) do m
        # Evaluate model and loss inside gradient context:
        y_pred = m(X_train)
        neg_loglik = -sum(logpdf_loss(y_train, y_pred))
        gamma_t = Flux.sigmoid.(m.gamma)
        neg_logprior = -sum(logpdf_spike_slab(m.weight, gamma_t))

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
    gamma_mat[epoch, :] = model.gamma

end

plot(train_loss)
plot(train_loss[100:n_iter])

plot(weights_mat)
plot(Flux.sigmoid.(gamma_mat))

plot(weights_mat[:, 1])
density(weights_mat[5000:n_iter, 1])
density(weights_mat[5000:n_iter, p])
