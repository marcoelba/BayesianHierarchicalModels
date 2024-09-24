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


function update_parameters_dict(
    params_dict::OrderedDict;
    name::String,
    size::Int64,
    log_prob_fun,
    bij=Base.identity,
    dependency=[]
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
        params_dict["ranges"] = []
        params_dict["bijectors"] = []    
    end

    if !(parameter_already_included)
        range = (params_dict["tot_params"] + 1):(params_dict["tot_params"] + size)
        params_dict["tot_params"] = params_dict["tot_params"] + size
    else
        range = params_dict["priors"][name]["range"]
        params_dict["tot_params"] = params_dict["tot_params"]
    end

    new_prior = OrderedDict(
        "size" => (size),
        "bij" => bij,
        "range" => range,
        "log_prob" => log_prob_fun,
        "dependency" => dependency
    )

    params_dict["priors"][name] = new_prior

    # Create a tuple for the ranges and the transformations
    if !(parameter_already_included)
        push!(params_dict["ranges"], params_dict["priors"][name]["range"])
        push!(params_dict["bijectors"], params_dict["priors"][name]["bij"])
    end
    
    return params_dict
end

function get_parameters_axes(params_dict)
    
        # Get params axes
        theta_components = tuple(Symbol.(params_dict["priors"].keys)...)
        proto_array = ComponentArray(;
            [Symbol(pp) => ifelse(params_dict["priors"][pp]["size"] > 1, ones(params_dict["priors"][pp]["size"]), ones(params_dict["priors"][pp]["size"])[1])  for pp in params_dict["priors"].keys]...
        )
        theta_axes = getaxes(proto_array)
        theta = Float32.(randn(params_dict["tot_params"]))
        begin
            theta_components = ComponentArray(theta, theta_axes)
        end

        return theta_axes, theta_components
end


function log_joint(theta; params_dict, theta_axes, model, log_likelihood, label)

    priors = params_dict["priors"]
    bijectors = params_dict["bijectors"]
    ranges = params_dict["ranges"]

    theta_transformed = vcat([bijectors[pp].(theta[ranges[pp]]) for pp in eachindex(bijectors)]...)

    # parameters extraction
    theta_components = ComponentArray(theta_transformed, theta_axes)

    predictions = model(
        theta_components
    )

    loglik = sum(log_likelihood(label, predictions...))

    # log prior
    log_prior = 0f0
    for prior in keys(priors)
        deps = priors[prior]["dependency"]
        log_prior += sum(priors[prior]["log_prob"](
            theta_components[prior],
            [theta_components[dep] for dep in deps]...
        ))
    end

    loglik + log_prior

end


function polynomial_decay(t::Int64; a::Float32=0.5f0, b::Float32=0.01f0, gamma::Float32=0.75f0)
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
    z_dim::Int64,
    n_iter::Int64,
    n_chains::Int64=1,
    samples_per_step::Int64=4,
    sd_init::Float32=0.5f0,
    use_noisy_grads::Bool=false,
    n_cycles::Int64=2
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
        z = Float32.(randn(z_dim)) * sd_init
        diff_results = DiffResults.GradientResult(z)

        lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)

        while (step ≤ n_iter) && !converged
            
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
            if use_noisy_grads
                grad_noise = Float32.(randn(z_dim)) .* lr_schedule[step]
            else
                grad_noise = 0f0
            end
            @. z = z - diff_grad + grad_noise

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
