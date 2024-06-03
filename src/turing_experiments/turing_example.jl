# Bayesian models in Turing
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames


# Following Turing tutorial at https://turing.ml/v0.22/docs/using-turing/guide

# define function using Turing syntax
@model function gdemo(y)
    # Variance
    s² ~ InverseGamma(2, 3)
    # Mean
    m ~ Normal(0, sqrt(s²))

    return y ~ Normal(m, sqrt(s²))
end

nuts_sample = sample(gdemo([2., 2., 2.45, 1.9, 1.95, 1.99, 2.1, 2.]), NUTS(0.65), 1000)


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

# Prior distribution
function prior(beta)
    Distributions.logpdf(Distributions.MultivariateNormal(ones(p)), beta)
end

# Likelihood
function likelihood(y, X, beta)
    sum(
        Distributions.logpdf(Distributions.MultivariateNormal(X * beta, ones(n)), y)
    )
end

# Joint
function log_joint(beta)
    likelihood(y, X, beta) + prior(beta)
end

log_joint(true_beta)

# Variational Distribution
# Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):
using StatsFuns

# Here MeanField approximation
# theta is the parameter vector in the unconstrained space
function getq(theta)
    Distributions.MultivariateNormal(theta[1:p], StatsFuns.softplus.(theta[p+1:p*2]))
end

getq([1., 2., 0., -1., 1., 2., 0., -1.])

# Chose the VI algorithm
advi = AdvancedVI.ADVI(10, 10_000)
# vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
q = vi(log_joint, advi, getq, randn(p*2))

# Check the ELBO
AdvancedVI.elbo(advi, q, log_joint, 1000)


# Define the mapping for a lower dimensional factored multivariate Normal
using Bijectors

# base distribution
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(p), ones(p))

# bijector transfrom FROM the latent space TO the REAL line
using AdvancedVI
using StatsFuns
using ComponentArrays, UnPack

proto_arr = ComponentArrays.ComponentArray(; L=zeros(p, p), b=zeros(p))
proto_axes = ComponentArrays.getaxes(proto_arr)
num_params = length(proto_arr)


function getq(theta)
    L, b = begin
        UnPack.@unpack L, b = ComponentArrays.ComponentArray(theta, proto_axes)
        LinearAlgebra.LowerTriangular(L), b
    end
    # The diagonal of the covariance matrix must be positive
    D = Diagonal(diag(L))
    Dplus = StatsFuns.softplus.(diag(L))
    A = L - D + Dplus



    d = Int(length(theta) / 2)
    A = @inbounds theta[1:d]
    b = @inbounds theta[(d + 1):(2 * d)]

    b = to_constrained ∘
        Bijectors.Shift(b; dim=Val(1)) ∘
        Bijectors.Scale(StatsFuns.softplus(A); dim=Val(1))
    
        return Turing.transformed(base_dist, b)
end
