# posterior utilities
using OrderedCollections
using LinearAlgebra
using Distributions
using PDMats


function average_posterior(dict_posteriors::Dict, distribution_type)

    n_posteriors = length(dict_posteriors)

    params_dist = params(dict_posteriors[1])
    n_params_dist = length(params_dist)
    array_parameters = [Float32.(zeros(size(p))) for p in params(dict_posteriors[1])]

    for chain = 1:n_posteriors
        params_dist_chain = params(dict_posteriors[chain])
        for param = 1:n_params_dist
            array_parameters[param] .+= params_dist_chain[param]
        end
    end

    for param = 1:n_params_dist
        array_parameters[param] .= array_parameters[param] ./ n_posteriors
    end

    return distribution_type(array_parameters...)
end


function posterior_samples(; vi_posterior::Distributions.Distribution, params_dict::OrderedDict, n_samples::Int64=1)

    raw_sample = rand(vi_posterior, n_samples)
    t_sample = vcat(
        [params_dict["bijectors"][pp].(raw_sample[params_dict["ranges"][pp], :]) for pp in eachindex(params_dict["bijectors"])]...
    )

    return t_sample
end


function posterior_samples_from_dict(; vi_posterior::Dict, params_dict::OrderedDict, n_samples::Int64=1)

    n_chains = length(vi_posterior)
    raw_sample = Float32.(zeros(params_dict["tot_params"], n_samples, n_chains))

    for chain in eachindex(vi_posterior)
        raw_sample[:,:, chain] = posterior_samples(
            vi_posterior=vi_posterior[chain],
            params_dict=params_dict,
            n_samples=n_samples
        )
    end
    mean_sample = mean(raw_sample, dims=3)[:,:,1]

    return mean_sample
end


function extract_parameter(;prior::String, params_dict::OrderedDict, samples_posterior::AbstractArray)
    if size(samples_posterior, 1) != params_dict["tot_params"]
        @error "Dimension mismatch! n. rows samples_posterior must be the same as params_dict[tot_params]"
    end

    return samples_posterior[params_dict["priors"][prior]["range"], :]

end
