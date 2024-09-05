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

end
