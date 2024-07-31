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


function density_posterior(posterior_samples, param, params_dict; plot_label=true)
    from = params_dict[param]["from"]
    to = params_dict[param]["to"]
    n_chains = length(posterior_samples)

    label = false

    plt = plot()
    for chain = 1:n_chains
        if plot_label
            if (to - from) > 1
                label = "$(param)_1_chain_$(chain)"
            else
                label = "$(param)_chain_$(chain)"
            end
        end
    
        density!(params_dict[param]["bij"].(posterior_samples[chain][from, :]), label=label)

        if (to - from) > 1
            for pp in range(from+1, to)
                if plot_label
                    label = "$(param)_$(pp)_chain_$(chain)"
                end
                density!(params_dict[param]["bij"].(posterior_samples[chain][pp, :]), label=label)
            end
        end
    
    end
    display(plt)
end
