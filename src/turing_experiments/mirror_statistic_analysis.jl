# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing

include(joinpath("mirror_statistic.jl"))
include(joinpath("../utils/classification_metrics.jl"))

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
label_files = "ms_analysis"


# Bayesian FDR
p0 = 900
p1 = 100
p = p0 + p1
true_coef = vcat(zeros(p0), ones(p1))

mc_samples = 1000
fdr_target = 0.1

Random.seed!(35)

posterior_mean_null = randn(p0) * 0.01
posterior_mean_active = randn(p1) * 0.1 .+ 1.
posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = randn(p0) * 0.01 .+ 0.1
posterior_std_active = randn(p1) * 0.01 .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)


posterior_gamma = vcat(
    ones(p0) * 0.01,
    ones(p1) * 0.9
)

posterior_gamma = vcat(
    ones(p0 + 10) * 0.01,
    ones(p1 - 10) * 0.9
)

weighted_posterior_mean = posterior_mean .* posterior_gamma
scatter(weighted_posterior_mean)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(weighted_posterior_mean, posterior_std)
])

# null posteriors
plt = plot()
for pp = 1:p0
    plt = density!(rand(Normal(weighted_posterior_mean[pp], posterior_std_null[pp]), 1000), label=false)
end
display(plt)

# non-null
plt = plot()
for pp = p0:p
    plt = density!(rand(Normal(weighted_posterior_mean[pp], posterior_std[pp]), mc_samples), label=false)
end
display(plt)

output = zeros(mc_samples)
fdr = []
tpr = []
selection_matrix = zeros(p, mc_samples)
optimal_t = []
mirror_coefficients = zeros(p, mc_samples)


for nn = 1:mc_samples
    beta_1 = rand(posterior)
    beta_2 = rand(posterior)

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)
    mirror_coefficients[:, nn] = mirror_coeffs

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    push!(optimal_t, opt_t)
    output[nn] = sum(mirror_coeffs .> opt_t)
    selection_matrix[:, nn] = (mirror_coeffs .> opt_t) * 1

    metrics = classification_metrics.wrapper_metrics(
        true_coef .> 0.,
        mirror_coeffs .> opt_t
    )
    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)

end

mean(output)
std(output)

mean(fdr)
mode(fdr)

histogram(output)
histogram(fdr)

histogram(optimal_t)

mean_selection_matrix = mean(selection_matrix, dims=2)[:, 1]
plt = scatter(mean_selection_matrix, label=false)
xlabel!("Regression Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))


classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    mean_selection_matrix .> 0.5
)

sort_indeces = sortperm(mean_selection_matrix, rev=true)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(mode(output))]] .= 1.
classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    sel_vec .> 0.
)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(mean(output)))]] .= 1.
classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    sel_vec .> 0.
)


range_included = Int.(sort(unique(output)))
fdr_vec = []
tpr_vec = []

for (ii, jj) in enumerate(range_included)
    sel_vec = zeros(p)
    sel_vec[sort_indeces[1:jj]] .= 1.
    met = classification_metrics.wrapper_metrics(
        true_coef .> 0.,
        sel_vec .> 0.
    )
    push!(fdr_vec, met.fdr)
    push!(tpr_vec, met.tpr)
end
scatter(fdr_vec)


plt = plot()
for jj = 1:p
    density!(mirror_coefficients[jj, :], label=false)
end
display(plt)


# -----------------------------------------------------------
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