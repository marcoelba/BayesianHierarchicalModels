# Extract parameters from posterior samples

function get_param(posterior_samples, param, params_dict)
    from = params_dict[param]["from"]
    to = params_dict[param]["to"]
    n_chains = length(posterior_samples)

    param_samples = []
    for chain = 1:n_chains
        push!(param_samples, posterior_samples[chain][from:to, :])
    end

    return param_samples
end
