# Test on GPA data
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections
using CSV

using AdvancedVI
using StatsFuns
using Bijectors
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack


gpa = CSV.read("/home/marco_ocbe/Downloads/data/data/gpa.csv", DataFrame)
# occasion is the time index
gpa_time = gpa.occasion
gpa_idx = gpa.student
y = gpa.gpa

n_individuals = length(unique(gpa.student))
n_per_ind = length(unique(gpa.occasion))


# define function using Turing syntax
@model function longitudinal_model(y, X, idx; n_individuals=length(unique(idx)))
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Intercept
    beta0_fixed ~ Turing.Normal(0., 5.)

    sigma_beta0 ~ truncated(Normal(0., 2.), 0., Inf)

    beta0_random ~ Turing.filldist(Turing.Normal(0., sigma_beta0), n_individuals)
    
    # Covariates
    # beta_fixed ~ Turing.MultivariateNormal(zeros(p), 5.)
    # beta_time ~ Turing.Normal(0., 5.)

    # sigma_beta ~ truncated(Normal(0., 5.), 0., Inf)

    # beta_random ~ Turing.filldist(
    #     Turing.Normal(0., sigma_beta), n_individuals, d
    # )

    # mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ sum(Xrand .* beta_random, dims=2)[:, 1]

    # mu = beta0_fixed .+ beta0_random .+ Xfix * beta_fixed
    # y ~ Turing.filldist(
    #     Turing.MultivariateNormal(mu, ones(n_individuals) .* sigma_y),
    #     n_per_ind
    # )

    # time intercept
    mu = beta0_fixed .+ beta0_random[idx]

    y ~ Turing.MultivariateNormal(mu, sigma_y)

end


model = longitudinal_model(y, NaN, gpa_idx)
nuts_lm = sample(model, NUTS(0.65), 2000)


# Wide format
new_y = zeros(n_individuals, n_per_ind)
for ii in range(1, n_individuals)
    new_y[ii, :] = y[gpa_idx .== ii]
end

gpa_time
Xtime = zeros(n_individuals, n_per_ind)
for tt in range(1, n_per_ind)
    Xtime[:, tt] .= tt
end


@model function wide_model(y, Xtime)
    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Intercept
    beta0_fixed ~ Turing.Normal(0., 5.)

    sigma_beta0 ~ truncated(Normal(0., 2.), 0., Inf)

    beta0_random ~ Turing.filldist(Turing.Normal(0., sigma_beta0), n_individuals)

    sigma_beta_time ~ truncated(Normal(0., 2.), 0., Inf)
    # beta_time ~ Turing.filldist(Turing.Normal(0., sigma_beta_time), n_per_ind)
    beta_time ~ Turing.Normal(0., 2.)

    # time intercept
    mu = beta0_fixed .+ beta0_random

    y ~ Turing.arraydist([
        Turing.MultivariateNormal(mu .+ tt * beta_time, sigma_y) for tt in range(1, n_per_ind)
    ])

end


model2 = wide_model(new_y, NaN)
nuts_lm2 = sample(model2, NUTS(0.65), 2000)

plot(nuts_lm2[[
    "beta_time",
]])


model_vi = wide_model(new_y,NaN)

# ADVI
advi = ADVI(5, 1000)
q = vi(model_vi, advi)

samples = rand(q, 1000)
size(samples)
tot_params = size(samples)[1]

histogram(samples[1, :], label="sigma")

histogram(samples[2, :], label="beta0")
histogram(samples[3, :], label="sigma beta0")

histogram(samples[tot_params, :], label="beta time")
