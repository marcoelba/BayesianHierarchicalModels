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

end
