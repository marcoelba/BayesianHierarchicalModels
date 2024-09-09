# Mirror Statistic

module MirrorStatistic

using Distributions
using OrderedCollections
using LinearAlgebra
using Turing: arraydist


function mean_folded_normal(mu, sigma)
    sigma * sqrt(2/pi) * exp(-0.5 *(mu/sigma)^2) + mu * (1 - 2*cdf(Normal(), -(mu/sigma)))
end

function var_folded_normal(mu, sigma)
    mu^2 + sigma^2 - mean_folded_normal(mu, sigma)^2
end

function posterior_ms_coefficients(;vi_posterior::Distributions.Distribution, prior::String, params_dict::OrderedDict)

    # get mean and std of the multivariate normal on the regression coefficients
    mu = params(vi_posterior)[1][params_dict["priors"][prior]["range"]]
    sigma = sqrt.(diag(params(vi_posterior)[2])[params_dict["priors"][prior]["range"]])
    
    # MS Distribution
    mean_vec = MirrorStatistic.mean_folded_normal.(mu, sigma) .- 
        MirrorStatistic.mean_folded_normal.(0., sigma)

    var_vec = MirrorStatistic.var_folded_normal.(mu, sigma) .+ 
        MirrorStatistic.var_folded_normal.(0., sigma)

    ms_dist_vec = arraydist([
        Distributions.Normal(mean_ms, sqrt(var_ms)) for (mean_ms, var_ms) in zip(mean_vec, var_vec)
    ])

    return ms_dist_vec
end

function get_t(mirror_coeffs; fdr_target)
    
    optimal_t = 0
    t = 0
    
    for t in sort(abs.(mirror_coeffs))
        n_left_tail = sum(mirror_coeffs .< -t)
        n_right_tail = sum(mirror_coeffs .> t)
        n_right_tail = ifelse(n_right_tail .> 0, n_right_tail, 1)
    
        fdp = n_left_tail / n_right_tail
    
        if fdp <= fdr_target
            optimal_t = t
            break
        end
    end
    return optimal_t    
end

function mirror_statistic(theta_1, theta_2)
    abs.(theta_1 .+ theta_2) .- abs.(theta_1 .- theta_2)
end

function fdr_distribution(;ms_dist_vec, mc_samples::Int64, beta_true, fdr_target::Real=0.1)

    fdr_distribution = zeros(mc_samples)
    tpr_distribution = zeros(mc_samples)
    n_selected_distribution = zeros(mc_samples)
    selection_matrix = zeros(length(beta_true), mc_samples)

    for nn = 1:mc_samples

        # mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)
        mirror_coeffs = rand(ms_dist_vec)
        opt_t = get_t(mirror_coeffs; fdr_target=fdr_target)
        n_selected = sum(mirror_coeffs .> opt_t)

        n_selected = sum(mirror_coeffs .> opt_t)
        n_selected_distribution[nn] = n_selected
        selection_matrix[:, nn] = (mirror_coeffs .> opt_t) * 1

        metrics = wrapper_metrics(
            beta_true .!= 0.,
            mirror_coeffs .> opt_t
        )
        
        fdr_distribution[nn] = metrics.fdr
        tpr_distribution[nn] = metrics.tpr
    end

    return (
        fdr_distribution=fdr_distribution,
        tpr_distribution=tpr_distribution,
        n_selected_distribution=n_selected_distribution,
        selection_matrix=selection_matrix
    )
end

"""
false_discovery_rate(;
    true_coef::BitVector,
    estimated_coef::BitVector
    )

    Calculate the False Discovery Rate from a given set of coefficients
"""
function false_discovery_rate(;
    true_coef::Union{Array{Real}, BitVector},
    estimated_coef::Union{Array{Real}, BitVector}
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

"""
    Calculate the True Positive Rate (aka Sensitivity, Recall, Hit Rate):
    TP / (TP + FN)

    true_positive_rate(;
        true_coef::Union{Vector{Float64}, BitVector},
        estimated_coef::Union{Vector{Float64}, BitVector}
    )
    # Arguments
    - `true_coef::BitVector`: boolean vector of true coefficients, '1' refers to a coef != 0 and '0' otherwise.
    - `estimated_coef::BitVector`: boolean vector of estimated coefficients, '1' refers to a coef != 0 and '0' otherwise.
"""
function true_positive_rate(;
    true_coef::Union{Array{Real}, BitVector},
    estimated_coef::Union{Array{Real}, BitVector}
    )

    sum_coef = true_coef + estimated_coef
    TP = sum(sum_coef .== 2.)
    FN = sum((sum_coef .== 1.) .& (true_coef .== 1))

    TPR = TP / (TP + FN)

    return TPR
end

"""
Compute collection of metrics
"""
function wrapper_metrics(true_coef, pred_coef)
    # FDR
    fdr = false_discovery_rate(
        true_coef=true_coef,
        estimated_coef=pred_coef
    )

    # TPR
    tpr = true_positive_rate(
        true_coef=true_coef,
        estimated_coef=pred_coef
    )

    return (fdr=fdr, tpr=tpr)

end
 
end