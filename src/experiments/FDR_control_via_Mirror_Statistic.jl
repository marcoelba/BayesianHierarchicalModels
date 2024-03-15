# FDR control via Mirror Statistic in Bayesian Models

using GLM
using Distributions
using DataFrames
using Random
using StatsPlots
using Turing
using LinearAlgebra
using DynamicHMC

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))
include(joinpath(abs_project_path, "utils", "posterior_inference.jl"))
include(joinpath(abs_project_path, "utils", "classification_metrics.jl"))
include(joinpath(abs_project_path, "models", "hierarchical_models.jl"))


n = 100
p = 100
n_zero_coef = 90

Random.seed!(32143)
# First coefficient is the intercept
beta_true = Random.rand([1.5, 1.], p)
which_zero = range(1, n_zero_coef)
beta_true[which_zero] .= 0.

sigma_y = 1.

X_dist = Distributions.Normal(0., 1.)
X = Random.rand(X_dist, (n, p))
y = 1. .+ X * beta_true + sigma_y * Random.rand(Distributions.Normal(), n)


# define function using Turing syntax
@model function lin_model(y, X)
    # Variance
    s2 ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    # beta (reg coefficients)
    beta_int ~ Turing.Normal(0., 10)
    beta ~ Turing.MultivariateNormal(zeros(p), 2.)
    mu = beta_int .+ X * beta

    return y ~ MultivariateNormal(mu, s2)
end


@model function lm_horseshoe_prior(y, X)
    p = size(X)[2]

    # Variance:
    sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    
    # coefficients:
    half_cauchy = Turing.truncated(Turing.Cauchy(0, 3); lower=0.)
    lambda ~ Turing.filldist(half_cauchy, p)
    beta ~ Turing.MvNormal(Diagonal((lambda)))

    beta_int ~ Turing.Normal(0, 5)

    # Likelihood
    mu = beta_int .+ X * beta
    y ~ Turing.MultivariateNormal(mu, sigma2_y)

end


dynamic_nuts = externalsampler(DynamicHMC.NUTS())

n_iter = 3000
nuts_lm = sample(
    lin_model(y, X),
    SGLD(stepsize=PolynomialStepsize(0.001)),
    n_iter
)
nuts_chains = DataFrames.DataFrame(nuts_lm)

df_hpd = posterior_inference.get_parameters_interval(
    nuts_lm;
    interval_type="CI",
    alpha_prob=0.01
)

autocor(nuts_chains[:, "beta[1]"])
density(nuts_chains[:, "beta[1]"])

autocor(nuts_chains[:, "beta[100]"], range(0, 100))
density(nuts_chains[:, "beta[100]"])


sigma2 = mean(nuts_chains[:, "sigma2_y"])
k_vec = []
for jj in 1:p
    lamba2 = mean(nuts_chains[:, "lambda[$(jj)]"])
    push!(k_vec, 1 - 1 / (1 + lamba2 * 3. * n / sigma2))
end
histogram(k_vec, bins=30)


# POSTERIOR ERROR PROBABILITIES (PEP)
ecdfplot(nuts_chains[:, "beta[1]"])
sum(nuts_chains[:, "beta[1]"] .> 0) / length(nuts_chains[:, "beta[1]"])

pep_vec = []
for jj in 1:p
    beta_j = nuts_chains[:, "beta[$(jj)]"]
    mean_beta_j = mean(beta_j)
    if mean_beta_j > 0
        pep_j = sum(beta_j .< 0) / length(beta_j)
    else
        pep_j = sum(beta_j .> 0) / length(beta_j)
    end
    push!(pep_vec, pep_j)
end

pep_vec = (pep_vec .- minimum(pep_vec)) / (maximum(pep_vec) - minimum(pep_vec))
histogram(pep_vec, bins=30)
pep_mat = hcat(1:p, pep_vec)


sum(sort(pep_mat[:, 2])[1:10])
mean(sort(pep_mat[:, 2])[1:10])

sum(sort(pep_mat[:, 2])[1:15])
mean(sort(pep_mat[:, 2])[1:15])
5 / 15

density(nuts_chains[:, "beta[$(46)]"])
sum(nuts_chains[:, "beta[$(1)]"] .< 0)
quantile(nuts_chains[:, "beta[$(1)]"], [0.222, 0.8])

pep_mat[pep_mat[:, 2] .< 0.05, :]


# PEP on simulation
n_mc = 1000
beta_posteriors = hcat(
    randn(n_mc, 45) * 0.1,
    -0.1 .+ randn(n_mc, 45) * 0.1,
    2 .+ randn(n_mc, 5) * 0.5,
    -2 .+ randn(n_mc, 5) * 0.5
)
density(beta_posteriors[:, 1])
density(beta_posteriors[:, 46])
density(beta_posteriors[:, 96])

pep_vec = []
for jj in 1:p
    beta_j = beta_posteriors[:, jj]
    mean_beta_j = mean(beta_j)
    if mean_beta_j > 0
        pep_j = sum(beta_j .< 0) / length(beta_j)
    else
        pep_j = sum(beta_j .> 0) / length(beta_j)
    end
    push!(pep_vec, pep_j)
end

histogram(pep_vec, bins=30)
pep_vec = (pep_vec .- minimum(pep_vec)) / (maximum(pep_vec) - minimum(pep_vec))
histogram(pep_vec, bins=30)

pep_mat = hcat(1:p, pep_vec)

sum(sort(pep_vec)[1:10])
mean(sort(pep_vec)[1:10])

sum(sort(pep_vec)[1:15])
mean(sort(pep_vec)[1:15])
5 / 15

pep_mat[pep_vec .< 0.1, :]

cumsum(sort(pep_vec)[1:20])


# Mirror Statistic
function get_t(mirror_coeffs; fdr_q=0.1)
    
    optimal_t = 0
    t = 0
    
    for t in sort(abs.(mirror_coeffs))
        n_left_tail = sum(mirror_coeffs .<= -t)
        n_right_tail = sum(mirror_coeffs .>= t)
        n_right_tail = ifelse(n_right_tail > 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail
    
        if fdp <= fdr_q
            optimal_t = t
            break
        end
    end

    return optimal_t
end


function mirror_stat_global(beta_post, q; aggregation=mean, spread_function="quantile")
    q_half = q / 2
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


function mirror_stat(b1, b2)
    abs(b1 + b2) - abs(b1 - b2)
end

# --------------------------------------
# simulate posterior distributions
p_test = 100
p1 = 10
n_mc = 500
half_dist = Int(n_mc / 2)
mirror_coeffs = zeros(p_test, half_dist)

# zero coeffs
beta_posteriors = hcat(
    randn(n_mc, 45) * 0.1,
    -0.2 .+ randn(n_mc, 45) * 0.1,
    2 .+ randn(n_mc, 5) * 0.5,
    -2 .+ randn(n_mc, 5) * 0.5
)
density(beta_posteriors[:, 1])
density(beta_posteriors[:, 46])

density(beta_posteriors[:, 91])
density(beta_posteriors[:, 96])


####
jj = 46
sign_vec = (-1).^range(1, length(beta_posteriors[:, jj]))
2 * abs(mean(beta_posteriors[:, jj])) - abs(mean(sign_vec .* beta_posteriors[:, jj]))

2 * abs(mean(beta_posteriors[:, jj])) - 
    abs(maximum(beta_posteriors[:, jj]) - minimum(beta_posteriors[:, jj]))

abs(mean(beta_posteriors[:, jj])) - 
    abs(quantile(beta_posteriors[:, jj], 0.975) - quantile(beta_posteriors[:, jj], 0.025)) / 2

abs(mean(beta_posteriors[:, jj])) - 
    abs(quantile(beta_posteriors[:, jj], 0.95) - quantile(beta_posteriors[:, jj], 0.05)) / 2

abs(mean(beta_posteriors[:, jj])) - 2 * std(beta_posteriors[:, jj])

mu = 0.05
sd = 0.1
x = Normal(mu, sd)
plot(x)
vline!([mu - 2*sd, mu + 2*sd])
vline!(quantile(x, [0.025, 0.975]))
####


# pairwise MS
for jj in 1:p_test

    left_tail = beta_posteriors[1:half_dist, jj]
    right_tail = beta_posteriors[half_dist + 1:half_dist * 2, jj]

    for cc in 1:half_dist
        mirror_coeffs[jj, cc] = mirror_stat(
            left_tail[cc],
            right_tail[cc]
        )
    end
end

histogram(mirror_coeffs[:, 1])

opt_t = get_t(mirror_coeffs[:, 50]; fdr_q=0.1)
sum(mirror_coeffs[:, 10] .> opt_t)
classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    mirror_coeffs[:, 10] .> opt_t
)


mean_mirror_coeffs = mean(mirror_coeffs, dims=2)[:, 1]
histogram(mean_mirror_coeffs, bins=30)

jj = 91
density(beta_posteriors[:, jj])
density!(mirror_coeffs[jj, :], bins=10)
vline!([mean(mirror_coeffs[jj, :])])

jj = 1
density(beta_posteriors[:, jj])
density!(mirror_coeffs[jj, :], bins=10)
vline!([mean(mirror_coeffs[jj, :])])

jj = 46
density(beta_posteriors[:, jj])
density!(mirror_coeffs[jj, :], bins=10)
vline!([mean(mirror_coeffs[jj, :])])
std(beta_posteriors[:, jj])
maximum(beta_posteriors[:, jj]) - minimum(beta_posteriors[:, jj])


# Calculate FDR
optimal_t = get_t(mean_mirror_coeffs; fdr_q=0.1)
sum(mean_mirror_coeffs .> optimal_t)

classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    mean_mirror_coeffs .> optimal_t
)


# --------------------------------------------------------
# Global MS
mean_mirror_coeffs = zeros(p)
for jj in 1:p
    mean_mirror_coeffs[jj] = mirror_stat_global(
        beta_posteriors[:, jj], 0.1;
        aggregation=median, spread_function="std"
    )
end

histogram(mean_mirror_coeffs, bins=40)
histogram(mean_mirror_coeffs[mean_mirror_coeffs .< 1])

# Calculate FDR
optimal_t = get_t(mean_mirror_coeffs, fdr_q = 0.1)
sum(mean_mirror_coeffs .> optimal_t)

classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    mean_mirror_coeffs .> optimal_t
)


# -----------------------------------------------
# On the real data
# MS with global statistic
mean_mirror_coeffs = zeros(p)
subsample = collect(range(1, 1000, step=20))

for jj in 1:p
    mean_mirror_coeffs[jj] = mirror_stat_global(
        nuts_chains[subsample, "beta[$(jj)]"], 0.2
    )
end

# some cheks
histogram(mean_mirror_coeffs, bins=100)
vline!([0])

sum(mean_mirror_coeffs .> 0)

jj = 1
density(nuts_chains[:, "beta[$(jj)]"])
vline!([mean_mirror_coeffs[jj]])

jj = 95
density(nuts_chains[:, "beta[$(jj)]"])
vline!([mean_mirror_coeffs[jj]])

mean_mirror_coeffs[91:100]

# Calculate FDR TPR
optimal_t = get_t(mean_mirror_coeffs, fdr_q=0.1)
sum(mean_mirror_coeffs .> optimal_t)

classification_metrics.wrapper_metrics(
    beta_true .!= 0.,
    mean_mirror_coeffs .> optimal_t
)

# RandMirror
using RandMirror

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

# last 10 coeffs
rand_ms["ms_coef"][91:100]
mean_mirror_coeffs[91:100]

ms_matrix = zeros(50, p)
for simu in 1:50
    rand_ms = RandMirror.randomisation_ds.real_data_rand_ms(
        y=y,
        X=X,
        gamma=1.,
        fdr_level=0.1,
        alpha_lasso=1.
    )
    ms_matrix[simu, :] = rand_ms["ms_coef"]
end

histogram(ms_matrix[:, 50], bins=10)
mean(ms_matrix[:, 50])
mean_mirror_coeffs[50]


