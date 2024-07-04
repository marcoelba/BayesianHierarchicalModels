# Mirror Statistic analysis
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections
using ProgressMeter

using StatsFuns
using Bijectors
using DiffResults
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack
using ADTypes
using Flux
using Zygote
using Turing
using AdvancedVI

include(joinpath("decayed_ada_grad.jl"))
include(joinpath("mixed_models_data_generation.jl"))
include(joinpath("mirror_statistic.jl"))
include(joinpath("gaussian_spike_slab.jl"))
include(joinpath("relaxed_bernoulli.jl"))
include(joinpath("plot_functions.jl"))
include(joinpath("../utils/classification_metrics.jl"))
include(joinpath("../utils/variable_selection_plus_inference.jl"))


n = 100
p = 100
p0 = 90
p1 = 10

half_1 = 1:Int(n/2)
half_2 = Int(n/2+1):n

# First coefficient is the intercept
true_beta = vcat(zeros(p0), ones(p1))
sigma_y = 1.
X_dist = Distributions.Normal(0., 1.)

Random.seed!(32143)

mc_samples = 1000
fdr_target = 0.1

fdr_distribution = zeros(mc_samples)
tpr_distribution = zeros(mc_samples)

for nn = 1:mc_samples
    X = Random.rand(X_dist, (n, p))
    # Get y = X * beta + err ~ N(0, 1)
    y = 1. .+ X * true_beta + sigma_y * Random.rand(Distributions.Normal(), n)

    # DS
    y1 = y[half_1]
    y2 = y[half_2]
    X1 = X[half_1, :]
    X2 = X[half_2, :]

    # LASSO + OLS
    ds_run = variable_selection_plus_inference.lasso_plus_ols(
        X1=X1,
        X2=X2,
        y1=y1,
        y2=y2,
        add_intercept=true
    )

    beta_1 = ds_run.lasso_coef
    beta_2 = ds_run.lm_coef

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_q=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = classification_metrics.wrapper_metrics(
        true_beta .!= 0.,
        mirror_coeffs .> opt_t
    )
    
    fdr_distribution[nn] = metrics.fdr
    tpr_distribution[nn] = metrics.tpr
end

mean(fdr_distribution)
mean(tpr_distribution)

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

fdr_plot = histogram(fdr_distribution, label="FDR", normalize=:probability)
savefig(fdr_plot, joinpath(abs_project_path, "results", "ms_analysis", "fdr.pdf"))

tpr_plot = histogram(tpr_distribution, label="TPR", normalize=:probability)
savefig(tpr_plot, joinpath(abs_project_path, "results", "ms_analysis", "tpr.pdf"))

