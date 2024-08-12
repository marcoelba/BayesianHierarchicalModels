# Plot functions
function posterior_summary(samples, param, param_dict; fun)
    fun(
        param_dict[param]["bij"].(
        samples[params_dict[param]["from"]:params_dict[param]["to"], :]
        ),
        dims=2
    )
end


function histogram_posterior(samples, param, param_dict; plot_label=true)
    from = param_dict[param]["from"]
    to = param_dict[param]["to"]

    label = false
    if plot_label
        label = "$(param)_1"
    end
    plt = histogram(param_dict[param]["bij"].(samples[from, :]), label=label)

    if (to - from) > 1
        for pp in range(from+1, to)
            if plot_label
                label = "$(param)_$(pp)"
            end
            histogram!(param_dict[param]["bij"].(samples[pp, :]), label=label)
        end
    end
    display(plt)
end


function density_posterior(posterior_samples, param, params_dict; plot_label=true, broadcast_input=true)
    from = params_dict[param]["from"]
    to = params_dict[param]["to"]
    mc_samples = size(posterior_samples[1], 2)
    n_chains = length(posterior_samples)
    
    label = false

    plt = plot()
    for chain = 1:n_chains
        samples = posterior_samples[chain][from:to, :]

        if broadcast_input
            n_params = size(params_dict[param]["bij"].(samples[:, 1]), 1)
        else
            n_params = size(params_dict[param]["bij"](samples[:, 1]), 1)
        end
        t_samples = zeros32(n_params, mc_samples)

        # Transform with bijector
        if broadcast_input
            t_samples = params_dict[param]["bij"].(samples)
        else
            for mc_sample = 1:mc_samples
                t_samples[:, mc_sample] = params_dict[param]["bij"](samples[:, mc_sample])
            end
        end

        if plot_label
            if n_params > 1
                label = "$(param)_1_chain_$(chain)"
            else
                label = "$(param)_chain_$(chain)"
            end
        end

        density!(t_samples[1, :], label=label)

        if n_params > 1
            for pp in range(2, n_params)
                if plot_label
                    label = "$(param)_$(pp)_chain_$(chain)"
                end
                
                density!(t_samples[pp, :], label=label)
            end
        end
    
    end
    return plt
end
