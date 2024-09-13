# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "utils", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "classification_metrics.jl"))

label_files = "ms_analysis_partial_identification"

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

fdr_target = 0.1
fp = 0
fn = 50

Random.seed!(35)

posterior_mean_null = vcat(randn(p0 - fp) * 0.0, rand([-0.1, 0.1], fp))
posterior_mean_active = vcat(randn(p1 - fn) * 0.05 .+ 1., randn(fn) * 0.1)

posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = abs.(randn(p0) * 0.01) .+ 0.1
posterior_std_active = abs.(randn(p1) * 0.05) .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)

scatter(posterior_mean)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(posterior_mean, posterior_std)
])

mc_samples = 2000

Random.seed!(123)
plt = scatter(rand(posterior), label=false, markersize=3)
xlabel!("Coefficients", labelfontsize=15)
vspan!(plt, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_posterior.pdf"))


# null posteriors
plt = plot()
for pp = 1:p0
    plt = density!(rand(Normal(posterior_mean[pp], posterior_std_null[pp]), 1000), label=false)
end
display(plt)

# non-null
plt = plot()
for pp = p0+1:p
    plt = density!(rand(Normal(posterior_mean[pp], posterior_std[pp]), mc_samples), label=false)
end
display(plt)


# MS Distribution
mean_vec = mean_folded_normal.(posterior_mean, posterior_std) .- 
    mean_folded_normal.(0., posterior_std)
var_vec = var_folded_normal.(posterior_mean, posterior_std) .+ 
    var_folded_normal.(0., posterior_std)

ms_dist_vec = arraydist([
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
])

scatter(mean_vec)
scatter(rand(ms_dist_vec))

mc_samples = 2000
# no MC loop
mirror_coefficients = rand(ms_dist_vec, mc_samples)
opt_t = 0
t = 0
for t in range(0., maximum(mirror_coefficients), length=2000)
    n_left_tail = sum(mirror_coefficients .< -t)
    n_right_tail = sum(mirror_coefficients .> t)
    n_right_tail = ifelse(n_right_tail .> 0, n_right_tail, 1)

    fdp = n_left_tail ./ n_right_tail

    if fdp .<= fdr_target
        opt_t = t
        break
    end
end

inclusion_matrix = mirror_coefficients .> opt_t
n_inclusion_per_mc = sum(inclusion_matrix, dims=1)[1,:]
histogram(n_inclusion_per_mc)
average_inclusion_number = Int(round(mean(n_inclusion_per_mc)))

n_inclusion_per_coef = sum(inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:,1]
scatter(n_inclusion_per_coef)

plt = scatter(mean_inclusion_per_coef, label=false)
xlabel!("Regression Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

sum(mean_inclusion_per_coef .> 0.1)

sort_indices = sortperm(n_inclusion_per_coef, rev=true)

selection = zeros(p)
selection[sort_indices[1:average_inclusion_number]] .= 1
metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection .> 0
)

fdr = []
for n in range(minimum(n_inclusion_per_mc), maximum(n_inclusion_per_mc))
    selection = zeros(p)
    selection[sort_indices[1:n]] .= 1
    metrics = classification_metrics.wrapper_metrics(
        true_coef .> 0.,
        selection .> 0
    )
    push!(fdr, metrics.fdr)
end
boxplot(fdr)

fdr = []
tpr = []
for mc in eachcol(inclusion_matrix)
    metrics = classification_metrics.wrapper_metrics(
        true_coef .> 0.,
        mc .> 0
    )
    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)
end
boxplot(fdr)
boxplot!(tpr)
mean(fdr)
mean(tpr)

# histograms
plt_n = histogram(n_inclusion_per_mc, label="# inc. covs")
vline!([mean(n_inclusion_per_mc)], color="red", label="Mean #", linewidth=5)
plt_fdr = histogram(fdr, label="FDR")
vline!([mean(fdr)], color="red", label="Mean FDR", linewidth=5)

plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_bayesian_fdr_n.pdf"))


# -------------- MC loop -------------
output = zeros(mc_samples)
fdr = []
tpr = []
selection_matrix = zeros(p, mc_samples)
optimal_t = []
mirror_coefficients = zeros(p, mc_samples)

for nn = 1:mc_samples

    mirror_coeffs = rand(ms_dist_vec)
    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    mirror_coefficients[:, nn] = mirror_coeffs
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

histogram(optimal_t)

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

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(median(output)))]] .= 1.
classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    sel_vec .> 0.
)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(minimum(output)))]] .= 1.
classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    sel_vec .> 0.
)

sel_vec = zeros(p)
sel_vec[sort_indeces[1:Int(round(maximum(output)))]] .= 1.
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

