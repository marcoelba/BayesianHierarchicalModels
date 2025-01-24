# Model definition

using OrderedCollections
using ProgressMeter
using DiffResults
using StatsFuns
using ComponentArrays, UnPack
using ADTypes
using Zygote
using AdvancedVI


function update_parameters_dict(
    params_dict::OrderedDict;
    name::String,
    dim_theta::Tuple,
    logpdf_prior,
    dim_z::Int64,
    vi_family,
    init_z=randn(dim_z),
    dependency=[],
    random_variable::Bool=true,
    noisy_gradient::Int64=0
    )

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
        params_dict["ranges_theta"] = []

        params_dict["tot_vi_weights"] = 0
        params_dict["ranges_z"] = []
        params_dict["vi_family_array"] = []

        params_dict["keys_prior_position"] = OrderedDict()
        params_dict["random_weights"] = []
        params_dict["noisy_gradients"] = []
    end

    if !(parameter_already_included)
        
        # first time call #

        # theta
        range_theta = (params_dict["tot_params"] + 1):(params_dict["tot_params"] + prod(dim_theta))
        params_dict["tot_params"] = params_dict["tot_params"] + prod(dim_theta)
        # range VI weights (z)
        range_z = (params_dict["tot_vi_weights"] + 1):(params_dict["tot_vi_weights"] + dim_z)
        params_dict["tot_vi_weights"] = params_dict["tot_vi_weights"] + dim_z
    else
        range_theta = params_dict["priors"][name]["range_theta"]
        params_dict["tot_params"] = params_dict["tot_params"]
        # VI weights
        range_z = params_dict["priors"][name]["range_z"]
        params_dict["tot_vi_weights"] = params_dict["tot_vi_weights"]

    end

    new_prior = OrderedDict(
        "dim_theta" => dim_theta,
        "range_theta" => range_theta,
        "logpdf_prior" => logpdf_prior,
        "dim_z" => dim_z,
        "range_z" => range_z,
        "vi_family" => vi_family,
        "init_z" => init_z,
        "dependency" => dependency,
        "random_variable" => random_variable
    )

    params_dict["priors"][name] = new_prior

    # Create a tuple for the ranges and the transformations
    if !(parameter_already_included)
        push!(params_dict["ranges_theta"], params_dict["priors"][name]["range_theta"])
        push!(params_dict["ranges_z"], params_dict["priors"][name]["range_z"])
        push!(params_dict["vi_family_array"], params_dict["priors"][name]["vi_family"])
        push!(params_dict["random_weights"], params_dict["priors"][name]["random_variable"])
        append!(params_dict["noisy_gradients"], ones(dim_z) .* noisy_gradient)

        params_dict["keys_prior_position"][Symbol(name)] = length(params_dict["vi_family_array"])
        params_dict["tuple_prior_position"] = (; params_dict["keys_prior_position"]...)
    end
    
    return params_dict
end


function get_parameters_axes(params_dict)
    
        # Get params axes
        theta_components = tuple(Symbol.(params_dict["priors"].keys)...)

        vector_init = []
        theta = []
        for pp in params_dict["priors"].keys

            if prod(params_dict["priors"][pp]["size"]) > 1
                param_init = params_dict["priors"][pp]["bij"](ones(params_dict["priors"][pp]["size"]))
            else
                param_init = params_dict["priors"][pp]["bij"](ones(params_dict["priors"][pp]["size"])[1])
            end
            push!(vector_init, Symbol(pp) => param_init)
            push!(theta, param_init...)
        end

        proto_array = ComponentArray(; vector_init...)
        theta_axes = getaxes(proto_array)

        begin
            theta_components = ComponentArray(Float32.(theta), theta_axes)
        end

        return theta_axes, theta_components
end


function prediction_loglik(
    theta_components,
    model,
    log_likelihood,
    label,
    n_repeated_measures::Int64=1
    )

    loglik = 0f0
    if n_repeated_measures == 1
        predictions = model(
            theta_components
        )
        loglik += sum(log_likelihood(label, predictions...))
    else
        for rep = 1:n_repeated_measures
            predictions = model(
                theta_components,
                rep
            )
            loglik += sum(log_likelihood(label, predictions...))
        end
    end

    return loglik
end


function log_joint(theta; params_dict, theta_axes, model, log_likelihood, label, n_repeated_measures=1)

    priors = params_dict["priors"]
    bijectors = params_dict["bijectors"]
    ranges = params_dict["ranges"]

    theta_transformed = vcat(
        [bijectors[pp](theta[ranges[pp]]) for pp in eachindex(bijectors)]...
    )

    # parameters extraction
    theta_components = ComponentArray(theta_transformed, theta_axes)

    loglik = prediction_loglik(
        theta_components,
        model,
        log_likelihood,
        label,
        n_repeated_measures
    )
    
    # log prior
    log_prior = 0f0
    for prior in keys(priors)
        deps = priors[prior]["dependency"]
        log_prior += sum(priors[prior]["logpdf_prior"](
            theta_components[prior],
            [theta_components[dep] for dep in deps]...
        ))
    end

    loglik + log_prior

end


function polynomial_decay(t::Int64; a::Float32=1f0, b::Float32=0.01f0, gamma::Float32=0.75f0)
    a * (b + t)^(-gamma)
end

function cyclical_polynomial_decay(n_iter::Int64, n_cycles::Int64=2)
    steps_per_cycle = Int(n_iter / n_cycles)
    lr_schedule = []
    for cycle = 1:n_cycles
        push!(lr_schedule, polynomial_decay.(range(1, steps_per_cycle))...)
    end
    return lr_schedule
end


function training_loop(;
    log_joint,
    vi_dist,
    optimizer,
    z_dim::Int64,
    n_iter::Int64,
    n_chains::Int64=1,
    samples_per_step::Int64=4,
    sd_init::Float32=0.5f0,
    use_noisy_grads::Bool=false,
    n_cycles::Int64=2
    )

    steps_per_cycle = Int(n_iter / n_cycles)
    n_iter_tot = n_iter + steps_per_cycle

    elbo_trace = zeros(n_iter_tot, n_chains)
    best_iter = zeros(n_chains)

    posteriors = Dict()

    for chain in range(1, n_chains)

        println("Chain number: ", chain)

        # Define objective
        variational_objective = AdvancedVI.ELBO()
        # VI algorithm
        alg = AdvancedVI.ADVI(samples_per_step, n_iter_tot, adtype=ADTypes.AutoZygote())

        # --- Train loop ---
        converged = false
        step = 1

        prog = ProgressMeter.Progress(n_iter_tot, 1)
        # Init
        z = Float32.(randn(z_dim)) * sd_init

        best_z = copy(z)
        best_elbo = AdvancedVI.elbo(
            alg, vi_dist(z), log_joint, samples_per_step
        )

        diff_results = DiffResults.GradientResult(z)

        lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)

        while (step ≤ n_iter_tot) && !converged
            
            AdvancedVI.grad!(
                variational_objective,
                alg,
                vi_dist,
                log_joint,
                z,
                diff_results,
                samples_per_step
            )

            # # 2. Extract gradient from `diff_result`
            gradient_step = DiffResults.gradient(diff_results)

            # # 3. Apply optimizer, e.g. multiplying by step-size
            diff_grad = apply!(optimizer, z, gradient_step)

            # 4. Update parameters
            if step <= n_iter
                if use_noisy_grads
                    grad_noise = Float32.(randn(z_dim)) .* lr_schedule[step]
                else
                    grad_noise = 0f0
                end
            else
                grad_noise = 0f0
            end
            @. z = z - diff_grad + grad_noise

            # 5. Do whatever analysis you want - Store ELBO value
            current_elbo = AdvancedVI.elbo(
                alg, vi_dist(z), log_joint, samples_per_step
            )
            elbo_trace[step, chain] = current_elbo

            # elbo check
            if current_elbo > best_elbo
                best_elbo = copy(current_elbo)
                best_z = copy(z)
                best_iter[chain] = copy(step)
            end

            step += 1
            ProgressMeter.next!(prog)
        end

        q = vi_dist(best_z)
        posteriors[chain] = q
    end
    
    return Dict(
        "posteriors" => posteriors,
        "elbo_trace" => elbo_trace,
        "best_iter" => best_iter
    )

end
