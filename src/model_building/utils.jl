# Model definition

using OrderedCollections
using StatsFuns


function update_parameters_dict(params_dict::OrderedDict; name::String, size::Int64, bij=Base.identity)

    if !("priors" in keys(params_dict))
        # create sub-dictionary if first call
        params_dict["priors"] = OrderedDict()  
    end

    if name in keys(params_dict["priors"])
        error("Prior name <$(name)> already used")
    end

    if !("tot_params" in keys(params_dict))
        params_dict["tot_params"] = size
        new_prior = OrderedDict(
            "size" => (size),
            "bij" => bij,
            "from" => 1,
            "to" => size
        )
    else
        params_dict["tot_params"] = params_dict["tot_params"] + size
        new_prior = OrderedDict(
            "size" => (size),
            "bij" => bij,
            "from" => params_dict["tot_params"] + 1,
            "to" => params_dict["tot_params"] + size
        )
    end

    params_dict["priors"][name] = new_prior

    return params_dict
end
