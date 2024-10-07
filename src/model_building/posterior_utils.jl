# posterior utilities
using OrderedCollections
using LinearAlgebra
using Distributions
using PDMats
using ComponentArrays, UnPack


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

    one_sample = vcat(
        [params_dict["bijectors"][pp](raw_sample[params_dict["ranges"][pp], 1]) for pp in eachindex(params_dict["bijectors"])]...
    )
    t_sample = zeros(size(one_sample)..., n_samples)

    for mc = 1:n_samples
        t_sample[:, mc] = vcat(
            [params_dict["bijectors"][pp](raw_sample[params_dict["ranges"][pp], mc]) for pp in eachindex(params_dict["bijectors"])]...
        )    
    end

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
    if size(samples_posterior, 1) != params_dict["tot_params_t"]
        @error "Dimension mismatch! n. rows samples_posterior must be the same as params_dict[tot_params_t]"
        return
    end

    return samples_posterior[params_dict["priors"][prior]["range_transformed"], :]

end


function posterior_cluster_prob(;y_i, x_i, samples_posterior, n_clusters, model, theta_axes)
    
    cluster_probs = zeros(n_clusters, size(samples_posterior)[2])

    for mc = 1:MC_SAMPLES
        model_output = model(
            ComponentArray(Float32.(samples_posterior[:, mc]), theta_axes);
            X=x_i
        )
        log_sum = DistributionsLogPdf.log_normal_mixture(y_i, model_output...)
        base_dist = Normal.(model_output[2], model_output[3])
    
        cluster_probs[:, mc] = exp.(logpdf.(base_dist, y_i) .+ log.(model_output[1]) .- log_sum)
    end

    return cluster_probs

end
