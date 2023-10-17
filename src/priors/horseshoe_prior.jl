# Horseshoe prior
using Pkg
Pkg.status()

using Turing
using LinearAlgebra
using StatsPlots
using Random
using DataFrames
using Distributions

include("./utils/posterior_inference.jl")


@model function lm_horseshoe_prior(y, X)
    p = size(X)[2]

    # Variance:
    sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    
    # coefficients:
    half_cauchy = Turing.truncated(Turing.Cauchy(0, 1); lower=0.)
    tau ~ half_cauchy
    lambda ~ Turing.filldist(half_cauchy, p)
    beta_int ~ Turing.Normal(0, 5)
    beta ~ Turing.MvNormal(Diagonal((lambda .* tau).^2))
    
    # Likelihood
    mu = beta_int .+ X * beta
    y ~ Turing.MultivariateNormal(mu, sigma2_y)

end


# generate some data from linear Regression
n = 100
p = 10
X = Random.rand(Normal(0, 1), n, p)
beta_true = [1., -1, 1., -1., 0., 0., 0., 0., 0., 0.]
y = 1. .+ X * beta_true + Random.rand(Normal(0, 0.5), n)


# HMC-NUTS sampler
hmc_samples = sample(lm_horseshoe_prior(y, X), NUTS(0.65), 1000)
df_samples = DataFrames.DataFrame(hmc_samples)

plot(df_samples[!, "tau"])
plot!(df_samples[!, "beta[1]"])
plot!(df_samples[!, "beta[2]"])
plot!(df_samples[!, "beta[5]"])

posterior_inference.get_parameters_interval(hmc_samples, interval_type="HPD")


