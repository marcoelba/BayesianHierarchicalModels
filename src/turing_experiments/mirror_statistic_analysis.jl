# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra

include(joinpath("mirror_statistic.jl"))
include(joinpath("../utils/classification_metrics.jl"))


# Generate posterior samples from Gaussian distributions
p = 100
p0 = 90
p1 = 10

n = 10000

post_dist_null = Normal(0., 0.1)
post_dist_wrong_null = Normal(0.1, 0.01)
post_dist_active = Normal(1., 0.1)

samples_null = rand(post_dist_null, (p0 - 5, n))
samples_wrong_null = rand(post_dist_wrong_null, (5, n))
samples_active = rand(post_dist_active, (p1, n))

beta_posterior = vcat(samples_null, samples_wrong_null, samples_active)

plt = plot()
for pp in range(1, p)
    plt = density!(beta_posterior[pp, :], label=false)
end
display(plt)


# Get Mirror Stat posterior coefficients
ms_posterior = posterior_ms_coefficients(beta_posterior)

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
