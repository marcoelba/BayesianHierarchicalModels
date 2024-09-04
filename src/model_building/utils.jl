# Model definition

using OrderedCollections
using ProgressMeter
using DiffResults
using StatsFuns
using ComponentArrays, UnPack
using ADTypes
using Zygote
using AdvancedVI

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "utils", "decayed_ada_grad.jl"))


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


function log_joint(theta; priors_dict, theta_axes, model, log_likelihood, label)
    
    # parameters extraction
    begin
        theta_components = ComponentArray(theta, theta_axes)
    end

    predictions = model(
        theta_components,
        priors_dict
    )

    loglik = sum(log_likelihood(label, predictions...))

    # log prior
    log_prior = 0f0
    for prior in keys(priors_dict)
        println(prior)
        log_prior += priors_dict[prior]["log_prob"](
            priors_dict[prior]["bij"].(theta_components[prior])
        )
    end

    loglik + log_prior

end


function training_loop(;
    log_joint,
    vi_dist,
    z,
    n_iter::Int64,
    n_chains::Int64=1,
    samples_per_step::Int64=4
    )

    elbo_trace = zeros(n_iter, n_chains)

    posteriors = Dict()

    for chain in range(1, n_chains)

        println("Chain number: ", chain)

        # Define objective
        variational_objective = AdvancedVI.ELBO()
        # Optimizer
        optimizer = MyDecayedADAGrad()
        # VI algorithm
        alg = AdvancedVI.ADVI(samples_per_step, n_iter, adtype=ADTypes.AutoZygote())

        # --- Train loop ---
        converged = false
        step = 1

        prog = ProgressMeter.Progress(n_iter, 1)
        diff_results = DiffResults.GradientResult(z)

        while (step ≤ n_iter) && !converged

            # 1. Compute gradient and objective value; results are stored in `diff_results`
            # partial_log_joint(theta::AbstractArray) = log_joint(
            #     theta;
            #     priors_dict=priors_dict,
            #     theta_axes=theta_axes,
            #     predictor=Predictors.linear_model,
            #     log_likelihood=DistributionsLogPdf.log_normal,
            #     input=X,
            #     label=y
            # )
            
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
            @. z = z - diff_grad

            # 5. Do whatever analysis you want - Store ELBO value
            elbo_trace[step, chain] = AdvancedVI.elbo(
                alg, vi_dist(z), log_joint, samples_per_step
            )

            step += 1
            ProgressMeter.next!(prog)
        end

        q = vi_dist(z)
        posteriors[chain] = q
    end
    
    return Dict("posteriors" => posteriors, "elbo_trace" => elbo_trace)

end


function joint_log_prior(;priors_dict, theta_components)
    for prior in keys(priors_dict)
        log_prior += priors_dict[prior]["log_prob"](
            priors_dict[prior]["bij"].(theta_components[prior])
        )

    end
end
