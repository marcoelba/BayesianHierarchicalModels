# Model prediction functions


module Predictors

function linear_model(
    theta_components;
    priors_dict,
    X::AbstractArray,
    )
    n = size(X, 1)

    beta0 = priors_dict["beta0"]["bij"](theta_components["beta0"])
    beta = priors_dict["beta"]["bij"](theta_components["beta"])
    sigma_y = priors_dict["sigma_y"]["bij"](theta_components["sigma_y"])

    mu = beta0 .+ X * beta
    sigma = Float32.(ones(n)) .* sigma_y

    return (mu, sigma)
end


function linear_random_intercept_model(
    theta_components,
    priors_dict;
    Xfix::AbstractArray,
    n_time_points::Int64
    )
    n = size(Xfix, 1)

    sigma_y = priors_dict["sigma_y"]["bij"](theta_components["sigma_y"])

    beta_fixed = priors_dict["beta_fixed"]["bij"].(theta_components["beta_fixed"])

    beta0_fixed = priors_dict["beta0_fixed"]["bij"].(theta_components["beta0_fixed"])
    beta0_random = priors_dict["beta0_random"]["bij"].(theta_components["beta0_random"])

    beta_time = priors_dict["beta_time"]["bij"].(theta_components["beta_time"])

    mu = [beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ beta_time[tt] for tt = 1:n_time_points]
    sigma = [Float32.(ones(n)) .* sigma_y for tt = 1:n_time_points]

    return (hcat(mu...), hcat(sigma...))
end

end
