# Mixed Models data generation
using Distributions
using Random
using LinearAlgebra


function generate_mixed_model_data(;
    n_individuals, n_time_points,
    p, p1, p0, beta_pool=[-1., -2., 1, 2], obs_noise_sd=1.,
    include_random_int=true, random_intercept_sd=0.3,
    include_random_time=true, random_time_sd=0.5,
    include_random_slope=false, p_random_covs=0, random_slope_sd=0.5
    )

    data_dict = Dict()

    Random.seed!(234)
    # > FIXED EFFETCS <
    # Fixed effects, baseline covariates (DO NOT change over time)
    Xfix = Random.randn(n_individuals, p)
    data_dict["Xfix"] = Xfix

    # Fixed Coeffcients
    beta_fixed = vcat(zeros(p0), Random.rand(beta_pool, p1))
    beta0_fixed = 1.

    data_dict["beta_fixed"] = beta_fixed
    data_dict["beta0_fixed"] = beta0_fixed
    
    # > RANDOM EFFETCS <
    
    # Random Intercept (one per individual)
    if include_random_int
        beta0_random = Random.randn(n_individuals) .* random_intercept_sd
        data_dict["beta0_random"] = beta0_random
    end
    
    # Random Time effect
    if include_random_time
        beta_time = Random.randn(n_time_points) .* random_time_sd
        data_dict["beta_time"] = beta_time
    end

    # Beta Coefficients (only random deviations)
    if include_random_slope
        Xrand = Xfix[:, (p - p_random_covs + 1):p]
        beta_random = Random.randn(n_individuals, d) * random_slope_sd
        data_dict["beta_random"] = beta_random
    end
    
    # Outcome
    y = zeros(n_individuals, n_time_points)
    
    # only fixed effects and random intercept
    for ii in range(1, n_individuals)
        y[ii, :] .= beta0_fixed .+ Xfix[ii, :]' * beta_fixed

        if include_random_int
            y[ii, :] = y[ii, :] .+ beta0_random[ii]
        end

        if include_random_time
            y[ii, :] = y[ii, :] .+ beta_time
        end
    
        if include_random_slope
            y[ii, :] = y[ii, :] .+ Xrand[ii, :] * beta_random[ii, :]
        end

        # obs noise
        y[ii, :] += Random.randn(n_time_points) * obs_noise_sd
    end
    data_dict["y"] = y

    return data_dict
end
