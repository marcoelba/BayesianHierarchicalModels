# Plot functions
function posterior_summary(samples, param, param_dict; fun)
    fun(
        param_dict[param]["bij"].(
        samples[params_dict[param]["from"]:params_dict[param]["to"], :]
        ),
        dims=2
    )
end


function hist_posterior(samples, param, param_dict; plot_label=true)
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
