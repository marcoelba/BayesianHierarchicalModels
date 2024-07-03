# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing

include(joinpath("mirror_statistic.jl"))
include(joinpath("../utils/classification_metrics.jl"))


# Generate posterior samples from Gaussian distributions
p = 100
p0 = 90
p0_wrong = 5
p0_right = p0 - p0_wrong
p1 = 10

n = 10000

Random.seed!(343)
post_dist_null = Normal(0., 0.05)
post_dist_wrong_null = Normal(0, 0.05)
post_dist_active = Normal(1., 0.1)

samples_null = rand(post_dist_null, (p0_right, n)) .+ randn(p0_right)*0.05
samples_wrong_null = rand(post_dist_wrong_null, (p0_wrong, n)) .+ randn(p0_wrong)*0.05
samples_active = rand(post_dist_active, (p1, n)) .+ randn(p1)*0.1

beta_posterior = vcat(samples_null, samples_wrong_null, samples_active)
gamma_posterior = vcat(ones(p0_right)*0.1, ones(p0_wrong)*0.1, ones(p1)*0.9)

plt = plot()
for pp in range(1, p)
    plt = density!(beta_posterior[pp, :], label=false)
end
display(plt)


# Get Mirror Stat posterior coefficients
ms_posterior = posterior_ms_coefficients(beta_posterior)
ms_posterior = posterior_ms_coefficients(beta_posterior .* gamma_posterior)

plt = plot()
for pp in range(1, p)
    plt = density!(ms_posterior[pp, :], label=false)
end
display(plt)


fdr_target = 0.1

# Get inclusion for each MC sample
inclusion_posterior = zeros(size(ms_posterior))
opt_t_posterior = zeros(size(inclusion_posterior, 2))

for mc in range(1, size(inclusion_posterior, 2))
    opt_t_mc = get_t(ms_posterior[:, mc], fdr_q=fdr_target)
    opt_t_posterior[mc] = opt_t_mc
    inclusion_posterior[:, mc] = ms_posterior[:, mc] .> opt_t_mc
end

histogram(opt_t_posterior)
std(opt_t_posterior)

scatter(mean(inclusion_posterior, dims=2))
sum(mean(inclusion_posterior, dims=2) .> 0.5)


plt = plot()
for pp in range(1, p)
    plt = density!(ms_posterior[pp, :], label=false, color="grey")
end
display(plt)

plt = density!(opt_t_posterior, color="red", label="opt t")
plt = vline!(mean(opt_t_posterior, dims=1), color="red", label=false)


sum(mean(ms_posterior, dims=2) .> mean(opt_t_posterior))


mean(samples_wrong_null[1, :] .< 0)


# Loop like below
p0 = 900
p1 = 100
p = p0 + p1

mc_samples = 1000

Random.seed!(35)

posterior_mean_null = randn(p0) * 0. .+ 0.
posterior_mean_active = randn(p1) * 0.1 .+ 1.5
posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = randn(p0) * 0.01 .+ 0.1
posterior_std_active = randn(p1) * 0.01 .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(posterior_mean, posterior_std)
])

# null posteriors
plt = plot()
for pp = 1:p0
    plt = density!(rand(Normal(posterior_mean_null[pp], posterior_std_null[pp]), mc_samples), label=false)
end
display(plt)

# non-null
plt = plot()
for pp = p0:p
    plt = density!(rand(Normal(posterior_mean[pp], posterior_std[pp]), mc_samples), label=false)
end
display(plt)

output = zeros(mc_samples)

for nn = 1:mc_samples
    beta_1 = rand(posterior)
    beta_2 = rand(posterior)

    # scatter(beta_1, label="beta 1")
    # scatter!(beta_2, label="beta 2")

    mirror_coeffs = abs.(beta_1 .+ beta_2) .- 
        abs.(beta_1 .- beta_2)

    scatter!(mirror_coeffs, label="MC")

    opt_t = get_t(mirror_coeffs; fdr_q=0.05)
    output[nn] = sum(mirror_coeffs .> opt_t)
end

mean(output)
std(output)

histogram(output)

# Simulation with two point estimates (like LASSO + OLS)
p0 = 900
p1 = 100

Random.seed!(35)

output = zeros(1000)
for nn = 1:1000
    # beta_1 = vcat(randn(p1)*0.1 .+ 1, randn(p0) * 0.1 .+ 0.1)
    # beta_2 = vcat(randn(p1)*0.1 .+ 1, randn(p0) * 0.1 .+ 0.1)

    beta_1 = vcat(randn(p1)*0.1 .+ 1, rand(truncated(Normal(0, 0.1), -0.3, 0.2), p0))
    beta_2 = vcat(randn(p1)*0.1 .+ 1, rand(truncated(Normal(0, 0.1), -0.3, 0.2), p0))

    scatter(beta_1, label="beta 1")
    scatter!(beta_2, label="beta 2")

    mirror_coeffs = abs.(beta_1 .+ beta_2) .- 
        abs.(beta_1 .- beta_2)

    scatter!(mirror_coeffs, label="MC")

    opt_t = get_t(mirror_coeffs; fdr_q=0.05)
    output[nn] = sum(mirror_coeffs .> opt_t)
end

mean(output)
std(output)

histogram(output)


gamma_dist = truncated(Normal(0, 0.1), -0.3, 0.2)
mean(gamma_dist)
mean(rand(gamma_dist, 10000))
density(rand(gamma_dist, 10000))