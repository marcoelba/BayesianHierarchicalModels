# training utils
using ProgressMeter
using Zygote
using Optimisers
using ComponentArrays, UnPack


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


function compute_logpdf_prior(theta::ComponentArray; params_dict::OrderedDict)
    log_prior = 0f0
    priors = params_dict["priors"]

    for (ii, prior) in enumerate(keys(priors))
        deps = priors[prior]["dependency"]
        log_prior += sum(priors[prior]["logpdf_prior"](
            theta[Symbol(prior)],
            [theta[Symbol(dep)] for dep in deps]...
        ))
    end

    return log_prior
end


function get_parameters_axes(params_dict)
    
    # Get params axes
    theta_components = tuple(Symbol.(params_dict["priors"].keys)...)

    vector_init = []
    for pp in params_dict["priors"].keys

        if prod(params_dict["priors"][pp]["dim_theta"]) > 1
            param_init = ones(params_dict["priors"][pp]["dim_theta"])
        else
            param_init = ones(params_dict["priors"][pp]["dim_theta"])[1]
        end
        push!(vector_init, Symbol(pp) => param_init)
    end

    proto_array = ComponentArray(; vector_init...)
    theta_axes = getaxes(proto_array)

    return theta_axes
end


function elbo(
    z::AbstractArray;
    y::AbstractArray,
    X::AbstractArray,
    ranges_z::AbstractArray,
    vi_family_array::AbstractArray,
    random_weights::AbstractArray,
    model,
    theta_axes,
    log_likelihood,
    log_prior=zero,
    n_samples::Int64=1,
    n_repeated_measures::Int64=1
    )

    # get a specific distribution using the weights z, from the variational family
    q_dist_array = VariationalDistributions.get_variational_dist(z, vi_family_array, ranges_z)

    # evaluate the log-joint
    res = zero(eltype(z))

    for mc = 1:n_samples
        
        # random sample from VI distribution
        theta, abs_jacobian = VariationalDistributions.rand_with_logjacobian(q_dist_array, random_weights=random_weights)
        theta_components = ComponentArray(theta, theta_axes)

        # evaluate the log-joint
        loglik = 0.
        if n_repeated_measures == 1
            pred = model(theta_components; X=X)
            loglik += sum(log_likelihood(y, pred...))
        else
            for measurement = 1:n_repeated_measures
                pred = model(theta_components, measurement; X=X)
                loglik += sum(log_likelihood(y[:, :, measurement], pred...))
            end
        end
        logprior = sum(log_prior(theta_components))

        res += (loglik + logprior + sum(abs_jacobian)) / n_samples
    end

    # add entropy
    for d in q_dist_array
        res += VariationalDistributions.entropy(d)
    end

    return -res
end


function hybrid_training_loop(;
    z::AbstractArray,
    y::AbstractArray,
    X::AbstractArray,
    params_dict::OrderedDict,
    model,
    log_likelihood,
    log_prior=zero,
    n_iter::Int64,
    optimiser::Optimisers.AbstractRule,
    save_all::Bool=false,
    use_noisy_grads::Bool=false,
    elbo_samples::Int64=1,
    lr_schedule=nothing,
    n_repeated_measures::Int64=1
    )

    vi_family_array = params_dict["vi_family_array"]
    ranges_z = params_dict["ranges_z"]
    random_weights = params_dict["random_weights"]
    noisy_gradients = params_dict["noisy_gradients"]
    theta_axes = get_parameters_axes(params_dict)

    # store best setting
    best_z = copy(z)
    best_loss = eltype(z)(Inf)
    best_iter = 1

    # lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)
    state = Optimisers.setup(optimiser, z)

    prog = ProgressMeter.Progress(n_iter, 1)
    iter = 1

    # store results
    if save_all
        z_trace = zeros(eltype(z), n_iter, length(z))
    else
        z_trace = nothing
    end
    loss = []
    
    while iter ≤ n_iter
        
        train_loss, grads = Zygote.withgradient(z) do zp
            elbo(zp;
                y=y,
                X=X,
                ranges_z=ranges_z,
                vi_family_array=vi_family_array,
                random_weights=random_weights,
                model=model,
                theta_axes=theta_axes,
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                n_samples=elbo_samples,
                n_repeated_measures=n_repeated_measures
            )
        end
    
        # z update
        Optimisers.update!(state, z, grads[1])
        
        # Using noisy gradients?
        if use_noisy_grads
            rand_z = randn(eltype(z), length(z)) .* noisy_gradients .* lr_schedule[iter]
            z .+= rand_z
        end

        # store
        push!(loss, train_loss)
        if save_all
            z_trace[iter, :] = z
        end
        
        # elbo check
        if train_loss < best_loss
            best_loss = copy(train_loss)
            best_z = copy(z)
            best_iter = copy(iter)
        end

        iter += 1
        ProgressMeter.next!(prog)
    end

    loss_dict = Dict("loss" => loss, "z_trace" => z_trace)

    best_iter_dict = Dict(
        "best_loss" => best_loss,
        "best_z" => best_z,
        "final_z" => z,
        "best_iter" => best_iter
    )

    return Dict("loss_dict" => loss_dict, "best_iter_dict" => best_iter_dict)
end
