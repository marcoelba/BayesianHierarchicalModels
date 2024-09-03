# Model definition

using OrderedCollections
using StatsFuns
using ComponentArrays, UnPack


function update_parameters_dict(params_dict::OrderedDict; name::String, size::Int64, bij=Base.identity, log_prob_fun)

    if !("priors" in keys(params_dict))
        # create sub-dictionary if first call
        params_dict["priors"] = OrderedDict()  
    end

    parameter_already_included = false
    if name in keys(params_dict["priors"])
        @warn "Prior <$(name)> overwritten"
        parameter_already_included = true
    end

    # If first call
    if !("tot_params" in keys(params_dict))
        params_dict["tot_params"] = 0
        from = 0
        to = 0
    end

    if !(parameter_already_included)
        from = params_dict["tot_params"] + 1
        to = params_dict["tot_params"] + size
        params_dict["tot_params"] = params_dict["tot_params"] + size
    else
        from = params_dict["priors"][name]["from"]
        to = params_dict["priors"][name]["to"]
        params_dict["tot_params"] = params_dict["tot_params"]
    end

    new_prior = OrderedDict(
        "size" => (size),
        "bij" => bij,
        "from" => from,
        "to" => to,
        "log_prob" => log_prob_fun
    )

    params_dict["priors"][name] = new_prior

    return params_dict
end


function log_joint(;theta, priors_dict, theta_axes, predictor, log_likelihood, input, label)
    
    # parameters extraction
    begin
        theta_components = ComponentArray(theta, theta_axes)
    end

    predictions = predictor(
        theta_components,
        priors_dict=priors_dict,
        X=input
    )

    loglik = sum(log_likelihood(label, predictions...))

    # log prior
    log_prior = 0f0
    for prior in keys(priors_dict)
        log_prior += priors_dict[prior]["log_prob"](
            priors_dict[prior]["bij"](theta_components[prior])
        )
    end

    loglik + log_prior

end
