# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using Turing
using StatsFuns

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
fp = 10
fn = 0

Random.seed!(35)

posterior_mean_null = vcat(randn(p0 - fp) * 0.05, rand([-0.2, 0.2], fp))
posterior_mean_active = vcat(randn(p1 - fn) * 0.05 .+ rand([-2., 2.], p1-fn), randn(fn) * 0.1 .+ 0.5)

posterior_mean = vcat(posterior_mean_null, posterior_mean_active)

posterior_std_null = vcat(abs.(randn(p0 - fp) * 0.05) .+ 0.1, abs.(rand([0.2], fp)))
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

ms_vec = [
    Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
]
ms_dist_vec = arraydist(ms_vec)

scatter(mean_vec)
scatter(rand(ms_dist_vec))

#
t = 0.3

function get_probs_w(ms_vec, t)
    probs_t = []
    for j_dist in ms_vec
        push!(probs_t, 1 - cdf(j_dist, t))
    end
    return probs_t        
end
probs_t = get_probs_w(ms_vec, t)
scatter(probs_t)

function get_fdp(probs_t, tau)
    sum((1 .- probs_t) .* (probs_t .> tau)) / sum(probs_t .> tau)
end
get_fdp(probs_t, t)

function get_probs_fdp(ms_vec, x)
    probs_t = get_probs_w(ms_vec, x[1])
    get_fdp(probs_t, x[2])
end


ms_samples = rand(ms_dist_vec)
Dp = ms_samples .> t
Dm = ms_samples .< -t
sum(Dm)

# d(t)(w > t) * (1 - r)
fp_est = sum(Dp .* (1 .- probs_t))
tot_d = sum(Dp)
fdp = fp_est / tot_d


function fdp_estimate(ms_samples, t)
    probs_t = []
    for j_dist in ms_vec
        push!(probs_t, 1 - cdf(j_dist, t))
    end
    
    Dp = ms_samples .> t
    
    # d(t)(w > t) * (1 - r)
    fp_est = sum(Dp .* (1 .- probs_t))
    tot_d = sum(Dp)
    fdp = fp_est / tot_d
    
    return fdp
end

function mc_fdp_estimate(ms_samples, t)
    fdr_MC_t = []
    for m in eachcol(ms_samples)
        push!(fdr_MC_t, fdp_estimate(m, t))
    end
    return fdr_MC_t
end

ms_samples = rand(ms_dist_vec, 1)
fdr_MC_t = mc_fdp_estimate(ms_samples, t)
mean(fdr_MC_t)
histogram(fdr_MC_t)

fdr = []
for t in range(minimum(abs.(ms_samples)), maximum(abs.(ms_samples)), length=100)
    push!(fdr, mean(mc_fdp_estimate(ms_samples, t)))
end
scatter(fdr)

opt_t = 0
for t in range(minimum(abs.(ms_samples)), maximum(abs.(ms_samples)), length=100)
    if mean(mc_fdp_estimate(ms_samples, t)) < 0.1
        opt_t = t
        break
    end
end
mean(mc_fdp_estimate(ms_samples, opt_t))

sum(ms_samples .> opt_t)
probs_t = []
for j_dist in ms_vec
    push!(probs_t, 1 - cdf(j_dist, opt_t))
end
scatter(probs_t)

metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    ms_samples[:, 1] .> opt_t
)


# -------------------------------------
# Using the FDR criterion from MS
fdr_target = 0.1
mc_samples = 1000
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
scatter(mean_inclusion_per_coef)


plt = scatter(mean_inclusion_per_coef, label=false)
xlabel!("Regression Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients");
display(plt)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(mean_inclusion_per_coef, fdr_target)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection
)


# cutoff at the average
sort_indices = sortperm(n_inclusion_per_coef, rev=true)

selection = zeros(p)
selection[sort_indices[1:average_inclusion_number]] .= 1
metrics = classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection .> 0
)

# range
fdr = []
tpr = []
for n in range(minimum(n_inclusion_per_mc), maximum(n_inclusion_per_mc))
    selection = zeros(p)
    selection[sort_indices[1:n]] .= 1
    metrics = classification_metrics.wrapper_metrics(
        true_coef .> 0.,
        selection .> 0
    )
    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)
end
boxplot(fdr)

mean((fdr .- fdr_target).^2)


# distribution
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

density(fdr, fill=true)
mean((fdr .- fdr_target).^2)


# cumulative inclusion
dimension_subsets = sum(inclusion_matrix, dims=1)
relative_inclusion_freq = mean(inclusion_matrix ./ dimension_subsets, dims=2)[:, 1]
relative_inclusion_freq = mean(inclusion_matrix, dims=2)[:, 1]

sort_relative_inclusion_freq = sort(relative_inclusion_freq, rev=false)
sort_cumsum = cumsum(sort_relative_inclusion_freq)
plot(sort_cumsum)
hline!([fdr_target])
# * mean(dimension_subsets)

cutoff = 0
for jj = Int.(range(p, 1, length=p))
    if sort_cumsum[jj] <= fdr_target
        cutoff = jj
        break
    end
end

min_inclusion_freq=sort_relative_inclusion_freq[cutoff]
range_p = collect(range(1, p))
scatter(relative_inclusion_freq, markersize=3, label="Sorted relative inclusion freq")
hline!([min_inclusion_freq], label="Cutoff inclusion", linewidth=2)
sum(relative_inclusion_freq .> min_inclusion_freq)

excluded = relative_inclusion_freq .< min_inclusion_freq
included = relative_inclusion_freq .>= min_inclusion_freq

plt = scatter(range_p[excluded], relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

selection = relative_inclusion_freq .>= min_inclusion_freq
metrics_relative = classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    selection
)


# histograms
plt_n = histogram(n_inclusion_per_mc, label="# inc. covs")
vline!([mean(n_inclusion_per_mc)], color="red", label="Mean #", linewidth=5)
plt_fdr = histogram(fdr, label="FDR")
vline!([mean(fdr)], color="red", label="Mean FDR", linewidth=5)

plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_bayesian_fdr_n.pdf"))


# -------------- MC loop -------------
fdr_target = 0.1
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
mean(tpr)

# cumulative inclusion
dimension_subsets = sum(selection_matrix, dims=1)
relative_inclusion_freq = mean(selection_matrix ./ dimension_subsets, dims=2)[:, 1]
relative_inclusion_freq = mean(selection_matrix ./ mean(dimension_subsets), dims=2)[:, 1]

sort_relative_inclusion_freq = sort(relative_inclusion_freq)
sort_indices = sortperm(relative_inclusion_freq, rev=false)
sort_cumsum = cumsum(sort_relative_inclusion_freq)

cutoff = 0
for jj = 1:p
    if sort_cumsum[jj] > fdr_target
        cutoff = jj - 1
        break
    end
end

cutoff
sort_cumsum[cutoff]

min_inclusion_freq = sort_relative_inclusion_freq[cutoff]

sum(sort_indices .>= cutoff)
sum(relative_inclusion_freq .>= min_inclusion_freq)

plt = scatter(relative_inclusion_freq)
hline!([min_inclusion_freq])
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

excluded = relative_inclusion_freq .<= min_inclusion_freq
included = relative_inclusion_freq .> min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

selection = relative_inclusion_freq .> sort_relative_inclusion_freq[cutoff]

classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection .> 0
)


histogram(optimal_t)

plt_n = histogram(output, label="# inc. covs")
vline!([mean(output)], color="red", label="Mean #", linewidth=5)
plt_fdr = histogram(fdr, label="FDR")
vline!([mean(fdr)], color="red", label="Mean FDR", linewidth=5)

plt = plot(plt_fdr, plt_n)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_bayesian_fdr_n.pdf"))


mean_selection_matrix = mean(selection_matrix, dims=2)[:, 1]
sum(mean_selection_matrix[mean_selection_matrix .> 0.5])
mean(mean_selection_matrix[mean_selection_matrix .<= 0.5])

plt = scatter(mean_selection_matrix, label=false)
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
xlabel!("Regression Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
savefig(plt, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

scatter(sort_indices[excluded], mean_selection_matrix[excluded])
scatter!(sort_indices[included], mean_selection_matrix[included])
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
xlabel!("Regression Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)

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

plt = histogram(optimal_t, normalize=true)
for jj = 1:p
    density!(mirror_coefficients[jj, :], label=false)
end
display(plt)

plt = histogram(optimal_t, normalize=true)
density!(mirror_coefficients[1, :], label=false)
display(plt)

plt = ecdfplot(Float64.(optimal_t), label="t", linewidth=3)
for jj = p0+1:p
    ecdfplot!(mirror_coefficients[jj, :], label=false)
end
display(plt)

plt = ecdfplot(Float64.(optimal_t), label="t", linewidth=3)
for jj = 1:100
    ecdfplot!(mirror_coefficients[jj, :], label=false)
end
display(plt)



# ----------------------------------------------------------
# -------------- Full Monte Carlo approach -------------
beta_1 = rand(posterior, mc_samples*2)
beta_2 = rand(posterior, mc_samples*2)

c_sum = beta_1 .+ beta_2
c_diff = beta_1 .- beta_2

abs_c_sum = abs.(c_sum)
abs_c_diff = abs.(c_diff)
cor(c_sum[1, :], c_diff[1, :])
cor(c_sum[p, :], c_diff[p, :])
cor(abs_c_sum[1, :], abs_c_diff[1, :])
cor(abs_c_sum[p, :], abs_c_diff[p, :])

scatter(mean(c_sum, dims=2))
scatter(mean(c_diff, dims=2))

scatter(mean(abs_c_diff, dims=2))

mirror_coeffs = abs_c_sum .- abs_c_diff
scatter(mirror_coeffs[1, :])
scatter(mirror_coeffs[:, 1])

density(mirror_coeffs[1, :])
density!(rand(ms_dist_vec, mc_samples*2)[1, :])

density(mirror_coeffs[p, :])
density!(rand(ms_dist_vec, mc_samples*2)[p, :])

# quantiles
quantile(mirror_coeffs[1, :], [0.05, 0.95])
quantile(mirror_coeffs[2, :], [0.05, 0.95])


mirror_coeffs = rand(ms_dist_vec, mc_samples*2)


# No loop
opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
selection_matrix = (mirror_coeffs .> opt_t) * 1

mean_selection_matrix = mean(selection_matrix, dims=2)[:,1]
sum_selection_matrix = sum(selection_matrix, dims=1)
relative_probs = mean(selection_matrix ./ sum_selection_matrix, dims=2)[:,1]

histogram(sum_selection_matrix')
mean(sum_selection_matrix')

scatter(mean_selection_matrix)

metrics = classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    mean_selection_matrix .> 0.5
)

fp_prob = 1 .- mean_selection_matrix

function fdr_estimate(fp_prob, fdr_target=0.1)
    c_opt = 0.
    for c in sort(fp_prob, rev=true)
        lower_than_c = fp_prob .<= c
        if (sum(fp_prob[lower_than_c]) / sum(lower_than_c)) <= fdr_target
            c_opt = c
            break
        end
    end
    return c_opt
end

sum(fp_prob .<= c_opt)
classification_metrics.wrapper_metrics(
    true_coef .!= 0.,
    fp_prob .<= c_opt
)

# ------------------------------------------------------

fdr_target = 0.1
output = zeros(mc_samples)
fdr = []
tpr = []
selection_matrix = zeros(p, mc_samples)
optimal_t = []
mirror_coefficients = zeros(p, mc_samples)

for nn = 1:mc_samples
    beta_1 = rand(posterior)
    beta_2 = rand(posterior)

    mirror_coeffs = abs.(beta_1 .+ beta_2) .- abs.(beta_1 .- beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
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
mean(tpr)


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


# MDS
fp = 11
dimension_subsets = p1 + fp
inclusion_probs = vcat(ones(p0 - fp) * 0.01, ones(fp) * 0.1, ones(p1) * 0.99)
relative_inclusion_freq = inclusion_probs ./ dimension_subsets

sum(relative_inclusion_freq[1:p0])
sum(relative_inclusion_freq[p0+1:p])

sum((inclusion_probs ./ fdr_target)[1:p0+1])
sum((inclusion_probs ./ (1-fdr_target))[p0:p])

sort_relative_inclusion_freq = sort(relative_inclusion_freq, rev=false)
sort_cumsum = cumsum(sort_relative_inclusion_freq)
plot(sort_cumsum)
hline!([fdr_target])

sum(sort_relative_inclusion_freq[1:p0])
sum(sort_relative_inclusion_freq[1:p0-fp])
sum(sort_relative_inclusion_freq[p0+1:p])
sum(sort_relative_inclusion_freq[p0-fp:p])

cutoff = 0
for jj = 1:p
    if sort_cumsum[jj] > fdr_target
        cutoff = jj - 1
        break
    end
end

cutoff
sort_cumsum[cutoff]
sort_cumsum[cutoff+1]

min_inclusion_freq = sort_relative_inclusion_freq[cutoff]

sum(relative_inclusion_freq .>= min_inclusion_freq)

plt = scatter(relative_inclusion_freq)
hline!([min_inclusion_freq])
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

excluded = relative_inclusion_freq .< min_inclusion_freq
included = relative_inclusion_freq .>= min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

selection = relative_inclusion_freq .>= sort_relative_inclusion_freq[cutoff]

classification_metrics.wrapper_metrics(
    true_coef .> 0.,
    selection .> 0
)
