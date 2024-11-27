# Mirror Statistic In Bayesian Statistics
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing
using StatsFuns
using StatsBase

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
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
fp = 30
fn = 0

Random.seed!(35)

posterior_mean_null = vcat(randn(p0) * 0.05, rand([-0.2, 0.2], 0))
posterior_mean_active = vcat(randn(p1 - fn) * 0.05 .+ rand([-2., 2.], p1-fn), randn(fn) * 0.1 .+ 0.2)

posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = vcat(abs.(randn(p0 - fp) * 0.05) .+ 0.1, abs.(rand([0.5], fp)))
posterior_std_active = abs.(randn(p1) * 0.05) .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)

scatter(posterior_mean)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(posterior_mean, posterior_std)
])


Random.seed!(123)
plt_posterior = scatter(rand(posterior), label=false, markersize=3)
xlabel!("Coefficients", labelfontsize=15)
vspan!(plt_posterior, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt_posterior)
savefig(plt_posterior, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_posterior.pdf"))

# Distribution with MS transformation
mean_vec = mean_folded_normal.(posterior_mean, posterior_std) .- 
    mean_folded_normal.(0., posterior_std)
var_vec = var_folded_normal.(posterior_mean, posterior_std) .+ 
    var_folded_normal.(0., posterior_std)

ms_vec = [
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
]
ms_dist_vec = arraydist(ms_vec)

scatter(mean_vec)
scatter(rand(ms_dist_vec))

Random.seed!(324)
plt_ms = scatter(rand(ms_dist_vec), label=false, markersize=3)
xlabel!("MS Coefficients", labelfontsize=15)
vspan!(plt_ms, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_ms)
savefig(plt_ms, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_W.pdf"))

plt = plot(plt_posterior, plt_ms)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_posterior_to_W.pdf"))


analytic = false

# Using the FDR criterion from MS
fdr_target = 0.1
mc_samples = 2000
# no MC loop

if analytic
    mirror_coefficients = rand(ms_dist_vec, mc_samples)
else
    beta_1 = rand(posterior, mc_samples)
    beta_2 = rand(posterior, mc_samples)
    mirror_coefficients = MirrorStatistic.mirror_statistic(beta_1, beta_2)
    plt_ms = scatter(mirror_coefficients[:, 1], label=false, markersize=3)
end

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
average_inclusion_number = Int(round(mean(n_inclusion_per_mc)))

plt_n = histogram(n_inclusion_per_mc, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([average_inclusion_number], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

n_inclusion_per_coef = sum(inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection
)

plt_probs = scatter(
    findall((1 .- mean_inclusion_per_coef) .> c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .> c_opt)],
    label=false, markersize=3,
)
scatter!(
    findall((1 .- mean_inclusion_per_coef) .<= c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .<= c_opt)],
    label="Selected", markersize=5,
)
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_and_probs.pdf"))


# Density of selected covariates
Random.seed!(123)
plt = density(rand(posterior, mc_samples)[selection, :]', label=false)
xlabel!("Coefficients", labelfontsize=15)
vspan!(plt, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_posterior.pdf"))


fp_prob = 1. .- mean_inclusion_per_coef
fdp = []
tau_range = range(0., 1., length=100)
for c in tau_range
    lower_than_c = mean_inclusion_per_coef .> c
    push!(fdp, sum(fp_prob .* lower_than_c) / sum(lower_than_c))
end
scatter(tau_range, fdp)
vline!([1-c_opt])
hline!([fdr_target])

# distribution
fdr = []
tpr = []
for mc in eachcol(inclusion_matrix)
    metrics = classification_metrics.wrapper_metrics(
        true_coef .!= 0.,
        mc .> 0
    )
    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)
end

histogram(fdr, label=false)


"""
Less clearly identified coefficients
"""
label_files = "higher_var"

p0 = 900
p1 = 100
p = p0 + p1
true_coef = vcat(zeros(p0), ones(p1))

fdr_target = 0.1
fp = 0
fn = 0

Random.seed!(35)

posterior_mean_null = vcat(randn(p0) * 0.1, rand([-0.2, 0.2], 0))
posterior_mean_active = vcat(
    randn(p1 - fn) * 0.1 .+ rand([-1., 1.], p1-fn),
    randn(fn) * 0.1 .+ 0.2
)

posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = vcat(abs.(randn(p0 - fp) * 0.05) .+ 0.1, abs.(rand([0.5], fp)))
posterior_std_active = abs.(randn(p1) * 0.05) .+ 0.1
posterior_std = vcat(posterior_std_null, posterior_std_active)

scatter(posterior_mean)

posterior = arraydist([
    Normal(mu, sd) for (mu, sd) in zip(posterior_mean, posterior_std)
])


Random.seed!(123)
plt_posterior = scatter(rand(posterior), label=false, markersize=3)
xlabel!("Coefficients", labelfontsize=15)
vspan!(plt_posterior, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt_posterior)
savefig(plt_posterior, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_posterior.pdf"))

# Distribution with MS transformation
mean_vec = mean_folded_normal.(posterior_mean, posterior_std) .- 
    mean_folded_normal.(0., posterior_std)
var_vec = var_folded_normal.(posterior_mean, posterior_std) .+ 
    var_folded_normal.(0., posterior_std)

ms_vec = [
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
]
ms_dist_vec = arraydist(ms_vec)

scatter(mean_vec)
scatter(rand(ms_dist_vec))

Random.seed!(324)
plt_ms = scatter(rand(ms_dist_vec), label=false, markersize=3)
xlabel!("MS Coefficients", labelfontsize=15)
vspan!(plt_ms, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_ms)
savefig(plt_ms, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_sample_W.pdf"))

plt = plot(plt_posterior, plt_ms)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_posterior_to_W.pdf"))


# Using the FDR criterion from MS
fdr_target = 0.1
mc_samples = 2000
# no MC loop

if analytic
    mirror_coefficients = rand(ms_dist_vec, mc_samples)
else
    beta_1 = rand(posterior, mc_samples)
    beta_2 = rand(posterior, mc_samples)
    mirror_coefficients = MirrorStatistic.mirror_statistic(beta_1, beta_2)
    plt_ms = scatter(mirror_coefficients[:, 1], label=false, markersize=3)
end

opt_t = 0
t = 0
for t in range(0., maximum(mirror_coefficients), length=1000)
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
average_inclusion_number = Int(round(mean(n_inclusion_per_mc)))

plt_n = histogram(n_inclusion_per_mc, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([average_inclusion_number], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

n_inclusion_per_coef = sum(inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:,1]
scatter(mean_inclusion_per_coef)
fp_prob = 1 .- mean_inclusion_per_coef

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection
)

sum(fp_prob[selection]) / sum(selection)
sum(sort(fp_prob[selection], rev=true)[1:12])

sum(fp_prob[selection] ./ sum(selection))


plt_probs = scatter(
    findall((1 .- mean_inclusion_per_coef) .> c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .> c_opt)],
    label=false, markersize=3,
)
scatter!(
    findall((1 .- mean_inclusion_per_coef) .<= c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .<= c_opt)],
    label="Selected", markersize=5,
)
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_and_probs.pdf"))


# Density of selected covariates
Random.seed!(123)
posterior_samples = rand(posterior, mc_samples)
true_positives = findall((true_coef .!= 0) .* selection)
false_positives = findall((true_coef .== 0) .* selection)
blue_col = cgrad(:blues, 100, categorical=true)
red_col = cgrad(:reds, length(false_positives), categorical=true)

plt = plot()
for (jj, pos) in enumerate(true_positives)
    density!(
        posterior_samples[pos, :],
        label=false,
        color=blue_col[jj]
    )
end
for (jj, pos) in enumerate(false_positives)
    density!(
        posterior_samples[pos, :],
        label=false,
        color=red_col[jj]
    )
end
xlabel!("Density Selected Coefficients", labelfontsize=15)
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_density_selected_coefs.pdf"))

# posterior inference false positives
posterior_samples[false_positives, :]
for jj in false_positives
    quantile(posterior_samples[jj, :], 0.025)
end

# posterior inference excluded coefficients
true_negatives = findall((true_coef .== 0) .* (selection .== 0))
credible_int = []
ci_selection = []
for jj in true_negatives
    ci = quantile(posterior_samples[jj, :], [0.025, 0.975])
    push!(credible_int, ci)
    push!(ci_selection, (ci[1] * ci[2]) > 0)
end
sum(ci_selection)

"""
    Cutoff selection on probabilities
"""
fn = 0
mean_inclusion_per_coef = vcat(
    rand(Uniform(0., 0.01), p0),
    rand(Uniform(0.9, 1.), p1 - fn),
    rand(Uniform(0., 0.1), fn)
)
scatter(mean_inclusion_per_coef)
fp_prob = 1 .- mean_inclusion_per_coef

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    selection
)

sum(fp_prob[selection]) / sum(selection)


"""
    Summary other than the mean
"""
fdr_target = 0.1
mc_samples = 2000
# no MC loop
mirror_coefficients = rand(ms_dist_vec, mc_samples)
opt_t = 0
t = 0
for t in range(0., maximum(mirror_coefficients), length=1000)
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
inclusion_probs = mean(inclusion_matrix, dims=2)[:,1]
inclusion_probs = inclusion_probs.^5
scatter(inclusion_probs)
fp_prob = 1 .- inclusion_probs

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    inclusion_probs,
    fdr_target
)
sum((1 .- inclusion_probs) .<= c_opt)

metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection
)

sum(fp_prob[selection]) / sum(selection)

inclusion_sums = sum(inclusion_matrix, dims=2)[:,1]
posterior_probs = Distributions.Beta.(
    inclusion_sums .+ 1000,
    mc_samples .- inclusion_sums .+ 1
)
post_inclusion_probs = mean.(posterior_probs)

scatter(inclusion_probs)
scatter!(post_inclusion_probs)

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    post_inclusion_probs,
    fdr_target
)
sum((1 .- post_inclusion_probs) .<= c_opt)
