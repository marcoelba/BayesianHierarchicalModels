# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "utils", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "classification_metrics.jl"))

label_files = "ms_analysis"

function mean_folded_normal(mu, sigma)
    sigma * sqrt(2/pi) * exp(-0.5 *(mu/sigma)^2) + mu * (1 - 2*cdf(Normal(), -(mu/sigma)))
end

function var_folded_normal(mu, sigma)
    mu^2 + sigma^2 - mean_folded_normal(mu, sigma)^2
end



# Bayesian FDR
p0 = 900
p1 = 100
p = p0 + p1
true_coef = vcat(zeros(p0), ones(p1))

mc_samples = 1000
fdr_target = 0.1
fp = 0
fn = 0

Random.seed!(35)

posterior_mean_null = vcat(randn(p0 - fp) * 0.05, rand([-0.1, 0.1], fp))
posterior_mean_active = vcat(randn(p1 - fn) * 0.05 .+ 1., randn(fn) * 0.01)

posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = abs.(randn(p0) * 0.01) .+ 0.1
posterior_std_active = abs.(randn(p1) * 0.2) .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)

scatter(posterior_mean)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(posterior_mean, posterior_std)
])

# null posteriors
plt = plot()
for pp = 1:p0
    plt = density!(rand(Normal(posterior_mean[pp], posterior_std_null[pp]), 1000), label=false)
end
display(plt)

# non-null
plt = plot()
for pp = p0+1:p
    plt = density!(rand(Normal(weighted_posterior_mean[pp], posterior_std[pp]), mc_samples), label=false)
end
display(plt)

# mix
plt = density(
    rand(Normal(weighted_posterior_mean[p0], posterior_std[p0]), mc_samples),
    label="Null", color="gray"
)
plt = density!(
    rand(Normal(weighted_posterior_mean[p], posterior_std[p]), mc_samples),
    label="Active", color="red", labelfontsize=20
)
for pp in range(p0-10, p0-1)
    plt = density!(
        rand(Normal(weighted_posterior_mean[pp], posterior_std[pp]), mc_samples),
        label=false, color="gray"
    )
end
for pp in range(p0+1, p0+10)
    plt = density!(
        rand(Normal(weighted_posterior_mean[pp], posterior_std[pp]), mc_samples),
        label=false, color="red"
    )
end
display(plt)
title!("Posterior Distributions", titlefontsize=15)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_posterior_dists.pdf"))


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
mean(fdr)
mode(fdr)

plt_n = histogram(output, label="# inc. covs")
vline!([mean(output)], color="red", label="Mean #", linewidth=5)
plt_fdr = histogram(fdr, label="FDR")
vline!([mean(fdr)], color="red", label="Mean FDR", linewidth=5)

plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_bayesian_fdr_n.pdf"))


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

selection = mean_selection_matrix .> 0.5
extra = Int(round(sum(selection) * (0.1 / 0.9)))
selection[
    sort_indeces[sum(selection) + 1:sum(selection) + extra]
] .= 1
classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    selection .> 0
)

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
scatter(range_included, fdr_vec)


# MS Distribution
mean_vec = mean_folded_normal.(posterior_mean, posterior_std) .- 
    mean_folded_normal.(0., posterior_std)
var_vec = var_folded_normal.(posterior_mean, posterior_std) .+ 
    var_folded_normal.(0., posterior_std)

ms_dist_vec = arraydist([
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
])


output = zeros(mc_samples)
fdr = []
tpr = []
selection_matrix = zeros(p, mc_samples)
optimal_t = []
mirror_coefficients = zeros(p, mc_samples)

for nn = 1:mc_samples

    mirror_coeffs = rand(ms_dist_vec)
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
mean(fdr)
mode(fdr)

plt_n = histogram(output, label="# inc. covs")
vline!([mean(output)], color="red", label="Mean #", linewidth=5)
plt_fdr = histogram(fdr, label="FDR")
vline!([mean(fdr)], color="red", label="Mean FDR", linewidth=5)

plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_bayesian_fdr_n.pdf"))


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

selection = mean_selection_matrix .> 0.5
extra = Int(round(sum(selection) * (0.1 / 0.9)))
selection[
    sort_indeces[sum(selection) + 1:sum(selection) + extra]
] .= 1

classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    selection .> 0
)

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

plt = plot()
for jj = 1:p
    density!(mirror_coefficients[jj, :], label=false)
end
display(plt)
vline!([mean(optimal_t)], label=false, linewidth=5)



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


###
x = Normal(1., 2)

y1 = abs.(rand(x, 1000) .+ rand(x, 1000))
y2 = abs.(rand(x, 1000) .- rand(x, 1000))
y = y1 .- y2

density(rand(x, 1000), label="Coef")
density!(y, label="MS")
density!(y1, label="+")
density!(y2, label="-")

