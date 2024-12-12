# Model prediction functions


module Predictors
using ComponentArrays

function linear_model(
    theta_c::ComponentArray;
    X::AbstractArray,
    )
    n = size(X, 1)

    mu = theta_c["beta0"] .+ X * theta_c["beta"]
    sigma = Float32.(ones(n)) .* theta_c["sigma_y"]

    return (mu, sigma)
end


function linear_predictor(
    theta_c::ComponentArray;
    X::AbstractArray,
    )

    mu = theta_c["beta0"] .+ X * theta_c["beta"]

    return (mu,)
end


function linear_random_intercept_model(
    theta_c::ComponentArray;
    Xfix::AbstractArray
    )
    n = size(Xfix, 1)
    n_time_points = length(theta_c["beta_time"])

    mu = [theta_c["beta0_fixed"] .+ theta_c["beta0_random"] .+ Xfix * theta_c["beta_fixed"] .+ theta_c["beta_time"][tt] for tt = 1:n_time_points]
    sigma = [Float32.(ones(n)) .* theta_c["sigma_y"] for tt = 1:n_time_points]

    return (hcat(mu...), hcat(sigma...))
end


function random_intercept_model(
    theta_c::ComponentArray;
    n_individuals,
    n_repetitions
    )

    mu = [theta_c["beta0_fixed"] .+ theta_c["beta0_random"] for rep = 1:n_repetitions]
    sigma = Float32.(ones(n_individuals, n_repetitions)) .* theta_c["sigma_y"]

    return (reduce(hcat, mu), sigma)
end


function linear_time_model(
    theta_c::ComponentArray;
    X::AbstractArray
    )
    n, p = size(X)
    n_time = length(theta_c["beta_time"])

    # baseline
    mu_inc = [
        theta_c["beta_time"][tt] .+ X * theta_c["beta_fixed"][:, tt] for tt = 1:n_time
    ]

    mu = cumsum(reduce(hcat, mu_inc), dims=2)
    sigma = reduce(hcat, [Float32.(ones(n)) .* theta_c["sigma_y"] for tt = 1:n_time])

    return (mu, sigma)
end


function linear_time_random_int_model(
    theta_c::ComponentArray;
    X::AbstractArray
    )
    n, p = size(X)
    n_time = length(theta_c["beta_time"])

    # baseline
    mu_baseline = theta_c["beta_time"][1] .+ X * theta_c["beta_fixed"][:, 1] .+ theta_c["beta0_random"]
    mu_inc = [
        theta_c["beta_time"][tt] .+ X * theta_c["beta_fixed"][:, tt] for tt = 2:n_time
    ]
    
    mu_matrix = reduce(hcat, [mu_baseline, reduce(hcat, mu_inc)])

    mu = cumsum(mu_matrix, dims=2)
    sigma = reduce(hcat, [Float32.(ones(n)) .* theta_c["sigma_y"] for tt = 1:n_time])

    return (mu, sigma)
end

end
