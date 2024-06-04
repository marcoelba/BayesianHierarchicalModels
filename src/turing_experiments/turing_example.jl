# Bayesian models in Turing
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames


" Example Linear Regression "
n = 100
p = 4

# First coefficient is the intercept
true_beta = [1.5, 1., -1., -1.5]
sigma_y = 1.

X = zeros(Float64, n, p)

Random.seed!(32143)

X[:, 1] .= 1.
X_dist = Distributions.Normal(0., sigma_y)
X[:, 2:p] = Random.rand(X_dist, (n, p-1))
# Get y = X * beta + err ~ N(0, 1)
y = X * true_beta + sigma_y * Random.rand(Distributions.Normal(), n)

# define function using Turing syntax
@model function lin_model(y, X)
    # Variance
    s2 ~ InverseGamma(2, 3)
    # beta (reg coefficients)
    beta ~ Turing.MultivariateNormal(zeros(p), 1.)
    mu = X * beta

    return y ~ MultivariateNormal(mu, s2)
end

model1 = lin_model(y, X)
nuts_lm = sample(lin_model(y, X), NUTS(0.65), 1000)

nuts_chains = DataFrames.DataFrame(nuts_lm)

plot(nuts_chains[!, "beta[1]"])
plot!(nuts_chains[!, "beta[2]"])
plot!(nuts_chains[!, "beta[3]"])
plot!(nuts_chains[!, "beta[4]"])


# ADVI
advi = ADVI(10, 1000)
model = lin_model(y, X)
q = vi(model, advi)

samples = rand(q, 1000)
size(samples)

histogram(samples[1, :])
histogram!(samples[2, :])
histogram!(samples[3, :])
histogram!(samples[4, :])


# -----------------------------------------
# Same inference via AdvancedVI library
# -----------------------------------------
using AdvancedVI
using StatsFuns

# Prior distribution
prior_beta = Distributions.MultivariateNormal(ones(p))
log_prior_beta(beta) = Distributions.logpdf(prior_beta, beta)

prior_sigma_y = Distributions.truncated(Normal(0., 5.), 0., Inf)
log_prior_sigma(sigma_y) = Distributions.logpdf(prior_sigma_y, sigma_y)

# Likelihood
likelihood(X, beta, sigma_y) = Distributions.MultivariateNormal(X * beta, ones(n) * sigma_y)
log_likelihood(y, X, beta, sigma_y) = sum(Distributions.logpdf(likelihood(X, beta, sigma_y), y))
likelihood(X, true_beta, 2.)

# Joint
function log_joint(theta_hat)
    beta = theta_hat[1:p]
    sigma_y = StatsFuns.softplus(theta_hat[p+1])
    log_likelihood(y, X, beta, sigma_y) + log_prior_beta(beta) + log_prior_sigma(sigma_y)
end
log_joint([1. ,0., 1., 0., -1.])

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
num_params = (p + 1) * 2
half_num_params = Int(num_params / 2)

function getq(theta)
    Distributions.MultivariateNormal(
        theta[1:half_num_params],
        StatsFuns.softplus.(theta[half_num_params+1:half_num_params*2])
    )
end

getq([1., 2., 0., -1., 1., 2., 0., -1., 1., -1.])

# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(num_params*2))

# sigma hat
StatsFuns.softplus(q.μ[p+1])
# beta hat
q.μ[1:p]

# Check the ELBO
AdvancedVI.elbo(advi, q, log_joint, 1000)


# -------------------------------------------------------------------------
# Define the mapping for a lower dimensional factored multivariate Normal
using Bijectors
# bijector transfrom FROM the latent space TO the REAL line
using AdvancedVI
using StatsFuns
using ComponentArrays, UnPack

# base distribution
base_dist = Distributions.MultivariateNormal(zeros(p), ones(p))

# BIJECTORS taken from Turing model
model = turing_model(y, X)

# # Turing ADVI
# advi = ADVI(10, 1000)
# q = vi(model, advi)

proto_arr = ComponentArrays.ComponentArray(; L=zeros(p, p), m=zeros(p))
proto_axes = ComponentArrays.getaxes(proto_arr)
num_params = length(proto_arr)


function getq(theta)
    L, m = begin
        UnPack.@unpack L, m = ComponentArrays.ComponentArray(theta, proto_axes)
        LinearAlgebra.LowerTriangular(L), m
    end
    # The diagonal of the covariance matrix must be positive
    D = Diagonal(diag(L))
    Dplus = Diagonal(StatsFuns.softplus.(diag(L)))
    A = L - D + Dplus

    transformed_dist = m + A * base_dist
    
    return transformed_dist
end

theta = ones(num_params)
getq(theta)

advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_params))

# check the covariance matrix
Sigma_hat = q_full.Σ
heatmap(Sigma_hat)


# -------------------------------------------------
# Lower dimensional Normal factorisation
# -------------------------------------------------
proto_arr = ComponentArrays.ComponentArray(; L=zeros(p, 2), d=zeros(p), m=zeros(p))
proto_axes = ComponentArrays.getaxes(proto_arr)
num_params = length(proto_arr)

function getq(theta)
    L, d, m = begin
        UnPack.@unpack L, d, m = ComponentArrays.ComponentArray(theta, proto_axes)
        L, d, m
    end
    # The diagonal of the covariance matrix must be positive
    A = L * L'
    Lo = LinearAlgebra.LowerTriangular(A)
    D = Diagonal(StatsFuns.softplus.(d))
    S = Lo + D

    transformed_dist = m + S * base_dist
    
    return transformed_dist
end

theta = ones(num_params)
getq(theta)

advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q_full = vi(log_joint, advi, getq, randn(num_params))

# check the covariance matrix
Sigma_hat = q_full.Σ
heatmap(Sigma_hat)
