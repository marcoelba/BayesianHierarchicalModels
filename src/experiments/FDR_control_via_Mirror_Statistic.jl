# FDR control via Mirror Statistic in Bayesian Models

using GLM
using Distributions
using DataFrames
using Random
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))
include(joinpath(abs_project_path, "utils", "posterior_inference.jl"))
include(joinpath(abs_project_path, "models", "hierarchical_models.jl"))


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
    s2 ~ InverseGamma(2, 3)
    # beta (reg coefficients)
    beta_int ~ Turing.Normal(0., 2.5)
    beta ~ Turing.MultivariateNormal(zeros(p), 1.)
    mu = beta_int .+ X * beta

    return y ~ MultivariateNormal(mu, s2)
end

nuts_lm = sample(lin_model(y, X), NUTS(0.65), 1000)
nuts_chains = DataFrames.DataFrame(nuts_lm)

df_hpd = posterior_inference.get_parameters_interval(
    nuts_lm;
    interval_type="HPD",
    alpha_prob=0.05
)
beta_est = df_hpd[3:size(df_hpd)[1], "Significative"] .!= ""

false_discovery_rate(
    true_coef=beta_true .!= 0.,
    estimated_coef=beta_est
)

# Mirror Statistic
function mirror_stat(beta_posterior_array)

    sign.(beta1 .* beta2) .* (abs.(beta1) .+ abs.(beta2))
end

function optimal_threshold(;mirror_coef, fdr_q)

    optimal_t = 0
    t = 0
    for t in range(0, maximum(mirror_coef), length=100)
        n_left_tail = sum(mirror_coef .< -t)
        n_right_tail = sum(mirror_coef .> t)
        n_right_tail = ifelse(n_right_tail > 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail

        if fdp <= fdr_q
            optimal_t = t
            break
        end
    end

    return optimal_t
end


fdr_q = 0.1
col_names = DataFrames.names(nuts_chains)
beta_cols = [occursin("beta", a) & (a != "beta_int") for a in col_names]
beta_posterior_array = nuts_chains[:, beta_cols]

average_sign = mean.(eachcol(sign.(beta_posterior_array)))
posterior_mean = mean.(eachcol(abs.(beta_posterior_array)))
fdr_sign = copy(average_sign)
fdr_sign[abs.(average_sign) .>= (1 - fdr_q)] .= 1

ms = fdr_sign .* posterior_mean
histogram(ms, bins=30)

opt_t = optimal_threshold(mirror_coef=ms, fdr_q=fdr_q)
false_discovery_rate(
    true_coef=beta_true .!= 0.,
    estimated_coef=ms .> opt_t
)

df_beta_hpd = df_hpd[3:size(df_hpd)[1], :]
df_beta_hpd[ms .> opt_t, :]
