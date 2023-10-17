# Hierarchical Models (aka Mixed Models in frequentist statistics)

module hierarchical_models

    import Turing
    import Distributions
    import StatsPlots
    import Random
    import DataFrames

    """
        hierarchical_model_normal_prior(y::Matrix{Float64}, n_groups::Int64, size_groups::Vector{Int64})
    """
    function hierarchical_mean_model_normal_prior(y::Matrix{Float64}, n_groups::Int64, size_groups::Vector{Int64})
        # --- Prior distributions:
        # variance y
        sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        # std dev beta group specific
        sigma_mu_group ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        
        # overall mean
        mu ~ Turing.Normal(0., 5.)
        # group means
        mu_group ~ Turing.MultivariateNormal(fill(mu, n_groups), sigma_mu_group)

        # --- Likelihood:
        for l in range(1, n_groups)
            y[:, l] ~ Turing.MultivariateNormal(fill(mu_group[l], size_groups[l]), sigma2_y)
        end

    end


    """
        hierarchical_lm_normal_prior(y::Matrix{Float64}, X_global::Array{Float64, 3}, X_groups::Array{Float64, 3}, n_groups::Int64)
    """
    @model function hierarchical_lm_normal_prior(y::Matrix{Float64}, X_global::Array{Float64, 3}, X_groups::Array{Float64, 3}, n_groups::Int64)
        p_global = size(X_global)[3]
        p_groups = size(X_groups)[3]
        # container for beta group specific
        beta_groups = zeros(n_groups, p_groups)
    
        # --- Priors:
        # variances y
        sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        # std dev beta group specific
        sigma_beta_group ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        
        # beta global effect:
        beta_global ~ Turing.MultivariateNormal(zeros(p_global), sqrt(5.))
        # beta group specific effects
        dist_beta_group = Turing.MultivariateNormal(zeros(p_groups), sigma_beta_group)

        # --- Likelihood
        for l in range(1, n_groups)
            beta_groups[l, :] ~ dist_beta_group
            mu_l = X_groups[l, :, :] * beta_groups[l, :] + X_global[l, :, :] * beta_global
            y[:, l] ~ Turing.MultivariateNormal(mu_l, sigma2_y)
        end
    
    end

    
    """
        hierarchical_lm_hs_prior(y::Matrix{Float64}, X_global::Array{Float64, 3}, X_groups::Array{Float64, 3}, n_groups::Int64)
    """
    @model function hierarchical_lm_hs_prior(y::Matrix{Float64}, X_global::Array{Float64, 3}, X_groups::Array{Float64, 3}, n_groups::Int64)
        p_global = size(X_global)[3]
        p_groups = size(X_groups)[3]
        # container for beta group specific
        beta_groups = zeros(n_groups, p_groups)

        # --- Priors:
        # variances y
        sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        # std dev beta group specific
        sigma_beta_group ~ Turing.TruncatedNormal(1., 1., 0., Inf)
        
        # beta global effect:
        beta_global ~ Turing.MultivariateNormal(zeros(p_global), sqrt(5.))
        # beta group specific effects
        dist_beta_group = Turing.MultivariateNormal(zeros(p_groups), sigma_beta_group)

        # --- Likelihood
        for l in range(1, n_groups)
            beta_groups[l, :] ~ dist_beta_group
            mu_l = X_groups[l, :, :] * beta_groups[l, :] + X_global[l, :, :] * beta_global
            y[:, l] ~ Turing.MultivariateNormal(mu_l, sigma2_y)
        end

    end

end


