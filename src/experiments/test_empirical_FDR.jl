# Simulation of empirical FDR for frequentist and Bayesian models
using Pkg
Pkg.status()

using GLM
using Distributions
using DataFrames
using Random
using StatsPlots


function false_discovery_rate(;
    true_coef::Union{Vector{Float64}, BitVector},
    estimated_coef::Union{Vector{Float64}, BitVector}
    )

    sum_coef = true_coef + estimated_coef
    TP = sum(sum_coef .== 2.)
    FP = sum((sum_coef .== 1.) .& (estimated_coef .== 1.))

    tot_predicted_positive = TP + FP

    if tot_predicted_positive > 0
        FDR = FP / tot_predicted_positive
    else
        FDR = 0.
        # println("Warning: 0 Positive predictions")
    end

    return FDR
end


# pvalue significance level
pvalue_sign_level = 0.05


# Generate data from a linear regression
Random.seed!(32345)
n = 200
p = 100
p_zero = 95

x_dist = Distributions.Normal()

beta_true = rand([-1., 1.], p)
beta_zero = range(1, p_zero)
beta_true[beta_zero] .= 0.
true_non_zero = beta_true .!= 0.


n_replications = 1000
fdr_vec = zeros(n_replications)

for rep in range(1, n_replications)
    X = rand(x_dist, n, p)
    y = X * beta_true + rand(x_dist, n)

    # Estimate classic Linear Regression
    lm1 = GLM.lm(X, y)
    lm1_pvalues = coeftable(lm1).cols[4]

    fdr_vec[rep] = false_discovery_rate(true_coef=true_non_zero, estimated_coef=lm1_pvalues .< pvalue_sign_level)
end

mean(fdr_vec)
histogram(fdr_vec)

1 - (1 - pvalue_sign_level)^p_zero
1 - (1 - pvalue_sign_level/p)^p_zero

