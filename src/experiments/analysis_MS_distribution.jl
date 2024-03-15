# analysis of Mirror Stat distribution for a conjugate normal model
using GLM
using Distributions
using DataFrames
using Random
using StatsPlots
using LinearAlgebra

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))
include(joinpath(abs_project_path, "utils", "classification_metrics.jl"))


function mirror_stat_global(beta_post; aggregation=mean, spread_function="quantile", quant=0.1)
    q_half = quant / 2
    point_estimate = abs(aggregation(beta_post))
    if spread_function == "quantile"
        variability = abs(
            quantile(beta_post, 1 - q_half) - 
            quantile(beta_post, q_half)
        ) / 2
    elseif spread_function == "std"
        variability = 2 * std(beta_post)
    end
    point_estimate - variability
end

function get_t(mirror_coeffs; fdr_q=0.1)
    
    optimal_t = 0
    t = 0
    
    for t in sort(abs.(mirror_coeffs))
        n_left_tail = sum(mirror_coeffs .< -t)
        n_right_tail = sum(mirror_coeffs .> t)
        n_right_tail = ifelse(n_right_tail > 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail
    
        if fdp <= fdr_q
            optimal_t = t
            break
        end
    end

    return optimal_t
end


# Generate some data
n = 100
p = 100
n_zero_coef = 95

Random.seed!(32143)
# First coefficient is the intercept
beta_true = Random.rand([1.5, 1.], p)
which_zero = range(1, n_zero_coef)
beta_true[which_zero] .= 0.

sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)


# Conjugate Gaussian Posterior distribution on regression coefficients

# stats from y
sigma_y = 1.

# Prior on Beta
Sigma_0 = LinearAlgebra.Diagonal(ones(p)) .* 1
Omega_0 = LinearAlgebra.inv(Sigma_0)
mu_0 = zeros(p)
b_0 = Omega_0 * mu_0

# Posterior
Omega_n = Omega_0 + X'*X / sigma_y
Sigma_n = inv(Omega_n)
b_n = b_0 + Sigma_n * X' * y / sigma_y

histogram(b_n, bins=100)
vline!([b_n[96]])
histogram(2*b_n, bins=10)

beta_posterior_dist = Normal.(b_n, sqrt.(diag(Sigma_n)))

# Get analytical values for MS
ms_coeffs = abs.(b_n) - sqrt.(diag(Sigma_n))
ms_coeffs = abs.(b_n) - 2 * sqrt.(diag(Sigma_n))
ms_coeffs = (abs.(b_n) - sqrt.(diag(Sigma_n))) .* vcat(ones(n_zero_coef)*0.1, ones(p-n_zero_coef))


histogram(ms_coeffs, bins=10)
vline!([0], width=2)

# Calculate FDR/TPR and the MS cutoff
optimal_t = get_t(ms_coeffs; fdr_q=0.1)
sum(ms_coeffs .< -optimal_t)
sum(ms_coeffs .> optimal_t)

classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    ms_coeffs .> optimal_t
)


# MS with posterior MC samples
mc_samples = 1000
ms_coeffs_mc = zeros(p)
for jj in 1:p
    beta_jj_mc = rand(beta_posterior_dist[jj], mc_samples)
    ms_coeffs_mc[jj] = mirror_stat_global(
        beta_jj_mc;
        aggregation=median,
        spread_function="quantile"
    )    
end
histogram(ms_coeffs_mc, bins=10)
vline!([0], width=2)

# Calculate FDR/TPR and the MS cutoff
optimal_t = get_t(ms_coeffs_mc; fdr_q=0.1)
classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    ms_coeffs_mc .> optimal_t
)


# With randomisation and MS
using RandMirror

# single run
rand_ms = RandMirror.randomisation_ds.real_data_rand_ms(
    y=y,
    X=X,
    gamma=1.,
    fdr_level=0.1,
    alpha_lasso=1.
)
classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    rand_ms["selected_ms_coef"]
)

histogram(rand_ms["ms_coef"][rand_ms["ms_coef"] .!= 0], bins=30)
rand_ms["optimal_t"]
sum(rand_ms["ms_coef"] .> rand_ms["optimal_t"])
sum(rand_ms["ms_coef"] .< -rand_ms["optimal_t"])

n_replica = 100
ms_matrix = zeros(n_replica, p)
for simu in 1:n_replica
    rand_ms = RandMirror.randomisation_ds.real_data_rand_ms(
        y=y,
        X=X,
        gamma=1.,
        fdr_level=0.1,
        alpha_lasso=1.
    )
    ms_matrix[simu, :] = rand_ms["ms_coef"]
end

histogram(mean(ms_matrix, dims=1)[1, :], bins=10)
vline!([0], width=2)
sum(mean(ms_matrix, dims=1)[1, :] .> 1)
