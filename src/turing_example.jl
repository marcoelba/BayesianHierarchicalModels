# Bayesian models in Turing

using Pkg
Pkg.status()

using Turing
using Distributions
using StatsPlots
using Random
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
