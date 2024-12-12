# Mixed Models data generation
using Distributions
using Random
using LinearAlgebra
using ToeplitzMatrices
using StatsFuns
using Turing: arraydist


function generate_mixed_model_data(;
    n_individuals, n_time_points, beta0_fixed=0,
    p, p1, p0, beta_pool=Float32.([-1., -2., 1, 2]), obs_noise_sd=1., corr_factor=0.5,
    include_random_int=true, random_int_from_pool=true, random_intercept_sd=0.3, beta0_pool=Float32.([-2, -1.5, -0.5, 0, 0.5, 1.5, 2]),
    include_random_time=true, beta_time=Float32.(range(0, 1., length=n_time_points)),
    include_random_slope=false, p_random_covs=0, random_slope_sd=0.5,
    random_seed=124, dtype=Float32
    )

    data_dict = Dict()

    Random.seed!(random_seed)

    # > FIXED EFFETCS <
    # Fixed effects, baseline covariates (DO NOT change over time)
    # block covariance matrix
    cor_coefs_0 = vcat([1.], [corr_factor * (p0 - ll) / (p0 - 1) for ll in range(1, p0-1)])
    cor_coefs_1 = vcat([1.], [corr_factor * (p1 - ll) / (p1 - 1) for ll in range(1, p1-1)])
    cov_matrix_0 = Array(Toeplitz(cor_coefs_0, cor_coefs_0))
    cov_matrix_1 = Array(Toeplitz(cor_coefs_1, cor_coefs_1))

    Xfix_0 = rand(MultivariateNormal(cov_matrix_0), n_individuals)
    Xfix_1 = rand(MultivariateNormal(cov_matrix_1), n_individuals)
    Xfix = transpose(vcat(Xfix_0, Xfix_1))

    data_dict["Xfix"] = dtype.(Xfix)

    # Fixed Coeffcients
    beta_fixed = dtype.(vcat(zeros(p0), Random.rand(beta_pool, p1)))
    beta0_fixed = dtype.(beta0_fixed)

    data_dict["beta_fixed"] = beta_fixed
    data_dict["beta0_fixed"] = beta0_fixed
    
    # > RANDOM EFFETCS <
    
    # Random Intercept (one per individual)
    if include_random_int
        if random_int_from_pool
            beta0_random = sample(beta0_pool, n_individuals, replace=true)
        else
            beta0_random = dtype.(Random.randn(n_individuals) .* random_intercept_sd)
        end
        data_dict["beta0_random"] = beta0_random
    end
    
    # Random Time effect
    if include_random_time
        data_dict["beta_time"] = beta_time
    end

    # Beta Coefficients (only random deviations)
    if include_random_slope
        Xrand = dtype.(Xfix[:, (p - p_random_covs + 1):p])
        beta_random = dtype.(Random.randn(n_individuals, d) * random_slope_sd)
        data_dict["beta_random"] = beta_random
    end
    
    # Outcome
    y = dtype.(zeros(n_individuals, n_time_points))
    
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
        y[ii, :] += dtype.(Random.randn(n_time_points) * obs_noise_sd)
    end
    data_dict["y"] = y

    return data_dict
end


function generate_linear_model_data(;
    n_individuals,
    p, p1, p0, beta_pool=Float32.([-1., -2., 1, 2]), obs_noise_sd=1., corr_factor=0.5,
    random_seed=124, dtype=Float32
    )

    data_dict = Dict()

    Random.seed!(random_seed)

    # > FIXED EFFETCS <
    # Fixed effects, baseline covariates (DO NOT change over time)
    # block covariance matrix
    cor_coefs_0 = vcat([1.], [corr_factor * (p0 - ll) / (p0 - 1) for ll in range(1, p0-1)])
    cor_coefs_1 = vcat([1.], [corr_factor * (p1 - ll) / (p1 - 1) for ll in range(1, p1-1)])
    cov_matrix_0 = Array(Toeplitz(cor_coefs_0, cor_coefs_0))
    cov_matrix_1 = Array(Toeplitz(cor_coefs_1, cor_coefs_1))

    Xfix_0 = rand(MultivariateNormal(cov_matrix_0), n_individuals)
    Xfix_1 = rand(MultivariateNormal(cov_matrix_1), n_individuals)
    Xfix = transpose(vcat(Xfix_0, Xfix_1))

    data_dict["X"] = dtype.(Xfix)

    # Fixed Coeffcients
    beta_fixed = dtype.(vcat(zeros(p0), Random.rand(beta_pool, p1)))
    beta0_fixed = dtype.(1.)

    data_dict["beta"] = beta_fixed
    data_dict["beta0"] = beta0_fixed
    
    # Outcome
    y = Xfix * beta_fixed .+ beta0_fixed + randn(n_individuals) .* obs_noise_sd
    data_dict["y"] = dtype.(y)

    return data_dict
end


function generate_logistic_model_data(;
    n_individuals, class_threshold=0.5f0,
    p, p1, p0, beta_pool=Float32.([-1., -2., 1, 2]), obs_noise_sd=0.1, corr_factor=0.5,
    random_seed=124, dtype=Float32
    )

    data_dict = Dict()

    Random.seed!(random_seed)

    # block covariance matrix
    cor_coefs_0 = vcat([1.], [corr_factor * (p0 - ll) / (p0 - 1) for ll in range(1, p0-1)])
    cor_coefs_1 = vcat([1.], [corr_factor * (p1 - ll) / (p1 - 1) for ll in range(1, p1-1)])
    cov_matrix_0 = Array(Toeplitz(cor_coefs_0, cor_coefs_0))
    cov_matrix_1 = Array(Toeplitz(cor_coefs_1, cor_coefs_1))

    Xfix_0 = rand(MultivariateNormal(cov_matrix_0), n_individuals)
    Xfix_1 = rand(MultivariateNormal(cov_matrix_1), n_individuals)
    Xfix = transpose(vcat(Xfix_0, Xfix_1))

    data_dict["X"] = dtype.(Xfix)

    # Fixed Coeffcients
    beta_fixed = dtype.(vcat(zeros(p0), Random.rand(beta_pool, p1)))
    beta0_fixed = dtype.(1.)

    data_dict["beta"] = beta_fixed
    data_dict["beta0"] = beta0_fixed
    
    # Outcome
    # lin_pred = Xfix * beta_fixed .+ beta0_fixed .+ randn(n_individuals) .* obs_noise_sd

    lin_pred = Xfix * beta_fixed .+ beta0_fixed
    y_dist = arraydist([Distributions.Bernoulli(StatsFuns.logistic(logitp)) for logitp in lin_pred])
    y = rand(y_dist)
    # y = StatsFuns.logistic.(lin_pred) .> class_threshold
    data_dict["y"] = Int.(y)
    data_dict["y_logit"] = dtype.(lin_pred)

    return data_dict
end


function generate_time_interaction_model_data(;
    n_individuals, n_time_points,
    p, p1, p0, beta_pool=Float32.([-1., -2., 1, 2]),
    obs_noise_sd=1., corr_factor=0.5,
    beta_time=Float32.(range(0, 1., length=n_time_points)),
    beta_time_int=zeros(p, n_time_points - 1),
    random_seed=124, dtype=Float32
    )

    data_dict = Dict()

    Random.seed!(random_seed)

    # > FIXED EFFETCS <
    # Fixed effects
    # block covariance matrix
    cor_coefs_0 = vcat([1.], [corr_factor * (p0 - ll) / (p0 - 1) for ll in range(1, p0-1)])
    cor_coefs_1 = vcat([1.], [corr_factor * (p1 - ll) / (p1 - 1) for ll in range(1, p1-1)])
    cov_matrix_0 = Array(Toeplitz(cor_coefs_0, cor_coefs_0))
    cov_matrix_1 = Array(Toeplitz(cor_coefs_1, cor_coefs_1))

    Xfix_0 = rand(MultivariateNormal(cov_matrix_0), n_individuals)
    Xfix_1 = rand(MultivariateNormal(cov_matrix_1), n_individuals)
    Xfix = transpose(vcat(Xfix_0, Xfix_1))

    data_dict["Xfix"] = dtype.(Xfix)

    # Fixed Regression Coeffcients
    beta_baseline = dtype.(vcat(zeros(p0), Random.rand(beta_pool, p1)))

    beta_time_int = dtype.(beta_time_int)

    beta_fixed = hcat(beta_baseline, beta_time_int)

    data_dict["beta_fixed"] = beta_fixed
        
    # Time effect - first element is the baseline intercept
    data_dict["beta_time"] = beta_time
    
    # Outcome
    mu_inc = [
        beta_time[tt] .+ Xfix * beta_fixed[:, tt] for tt = 1:n_time_points
    ]    
    mu = cumsum(reduce(hcat, mu_inc), dims=2)

    y = dtype.(mu .+ Random.randn(n_individuals, n_time_points) * obs_noise_sd)    
    data_dict["y"] = y

    return data_dict
end


function generate_time_interaction_multiple_measurements_data(;
    n_individuals, n_time_points, n_measurements,
    p, p1, p0, beta_pool=Float32.([-1., -2., 1, 2]),
    obs_noise_sd=1., corr_factor=0.5,
    include_random_int=false, random_int_from_pool=false,
    random_intercept_sd=0.3, beta0_pool=Float32.([-2, -1.5, -0.5, 0, 0.5, 1.5, 2]),
    beta_time=Float32.(range(0, 1., length=n_time_points)),
    beta_time_int=zeros(p, n_time_points - 1),
    random_seed=124, dtype=Float32
    )

    data_dict = Dict()

    Random.seed!(random_seed)

    # > FIXED EFFETCS <
    # Fixed effects
    # block covariance matrix
    cor_coefs_0 = vcat([1.], [corr_factor * (p0 - ll) / (p0 - 1) for ll in range(1, p0-1)])
    cor_coefs_1 = vcat([1.], [corr_factor * (p1 - ll) / (p1 - 1) for ll in range(1, p1-1)])
    cov_matrix_0 = Array(Toeplitz(cor_coefs_0, cor_coefs_0))
    cov_matrix_1 = Array(Toeplitz(cor_coefs_1, cor_coefs_1))

    Xfix_0 = rand(MultivariateNormal(cov_matrix_0), n_individuals)
    Xfix_1 = rand(MultivariateNormal(cov_matrix_1), n_individuals)
    Xfix = transpose(vcat(Xfix_0, Xfix_1))

    data_dict["Xfix"] = dtype.(Xfix)

    # Fixed Regression Coeffcients
    beta_baseline = dtype.(vcat(zeros(p0), Random.rand(beta_pool, p1)))

    beta_time_int = dtype.(beta_time_int)

    beta_fixed = hcat(beta_baseline, beta_time_int)

    data_dict["beta_fixed"] = beta_fixed
    
    # > RANDOM EFFETCS <    
    # Random Intercept (one per individual)
    if include_random_int
        if random_int_from_pool
            beta0_random = sample(beta0_pool, n_individuals, replace=true)
        else
            beta0_random = dtype.(Random.randn(n_individuals) .* random_intercept_sd)
        end
        data_dict["beta0_random"] = beta0_random
    end
    
    # Time effect - first element is the baseline intercept
    data_dict["beta_time"] = beta_time
    
    # Outcome
    mu_inc = [
        beta_time[tt] .+ Xfix * beta_fixed[:, tt] for tt = 1:n_time_points
    ]
    if include_random_int
        mu_inc[1] = mu_inc[1] .+ data_dict["beta0_random"]
    end
    
    mu = cumsum(reduce(hcat, mu_inc), dims=2)

    y = dtype.(mu .+ Random.randn(n_individuals, n_time_points) * obs_noise_sd)    
    data_dict["y"] = y

    return data_dict
end

