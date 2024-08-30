# Logistic Model
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using OrderedCollections
using ProgressMeter

using StatsFuns
using Bijectors
using DiffResults
# bijector transfrom FROM the latent space TO the REAL line
using ComponentArrays, UnPack
using ADTypes
using Flux
using Zygote
using Turing
using AdvancedVI

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "utils", "decayed_ada_grad.jl"))
include(joinpath(abs_project_path, "turing_experiments", "gaussian_spike_slab.jl"))
include(joinpath(abs_project_path, "mixed_models", "relaxed_bernoulli.jl"))


function logistic_model(;
    data_dict,
    num_steps=2000,
    n_runs=1,
    n_batches=1,
    samples_per_step=4,
    theta_start=false,
    theta_init_noise=0.1f0,
    sigma_cauchy=1f0
    )
    # Prior distributions
    params_dict = OrderedDict()
    num_params = 0

    # Fixed effects

    # ---------- Horseshoe distribution ----------
    params_dict["sigma_beta"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => StatsFuns.softplus)
    num_params += p
    log_prior_sigma_beta(sigma_beta::AbstractArray{Float32}) = sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, sigma_cauchy), 0f0, Inf32),
        sigma_beta
    ))
    
    params_dict["beta_fixed"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => identity)
    num_params += p
    function log_prior_beta_fixed(sigma_beta::AbstractArray{Float32}, beta::AbstractArray{Float32})
        sum(-0.5f0 * log.(2*Float32(pi)) .- log.(sigma_beta) .- 0.5f0 * (beta ./ sigma_beta).^2f0)
    end
    

    # Intercept
    params_dict["beta0_fixed"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => identity)
    num_params += 1
    prior_beta0_fixed = Distributions.Normal(0f0, 5f0)
    log_prior_beta0_fixed(beta0_fixed::Float32) = Distributions.logpdf(prior_beta0_fixed, beta0_fixed)


    # Likelihood
    function likelihood(;
        beta0_fixed::Float32,
        beta_fixed::AbstractArray{Float32},
        Xfix::AbstractArray{Float32}
        )
        lin_pred = beta0_fixed .+ Xfix * beta_fixed
        arraydist([Distributions.BernoulliLogit(logitp) for logitp in lin_pred])
    end

    function log_likelihood(;
        y::AbstractArray,
        beta0_fixed::Float32,
        beta_fixed::AbstractArray{Float32},
        Xfix::AbstractArray{Float32}
        )
        Distributions.logpdf(likelihood(
            beta0_fixed=beta0_fixed,
            beta_fixed=beta_fixed,
            Xfix=Xfix
        ), y)        
    end

    # Joint
    params_names = tuple(Symbol.(params_dict.keys)...)
    proto_array = ComponentArray(;
        [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn32(params_dict[pp]["size"]), randn32(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
    )
    proto_axes = getaxes(proto_array)
    num_params = length(proto_array)


    function log_joint(theta_hat::AbstractArray{Float32}; X_batch::AbstractArray{Float32}, y_batch::AbstractArray, n_batches::Int64)
        begin
            params_names = ComponentArray(theta_hat, proto_axes)
        end

        sigma_beta = params_dict["sigma_beta"]["bij"].(params_names.sigma_beta)
        beta_fixed = params_names.beta_fixed

        beta0_fixed = params_names.beta0_fixed

        loglik = log_likelihood(
            y=y_batch,
            beta0_fixed=beta0_fixed,
            beta_fixed=beta_fixed,
            Xfix=X_batch
        )

        log_prior = log_prior_sigma_beta(sigma_beta) +
            log_prior_beta_fixed(sigma_beta, beta_fixed) +
            log_prior_beta0_fixed(beta0_fixed)
        
        loglik * n_batches + log_prior 
    end

    # Variational Distribution
    # Provide a mapping from distribution parameters to the distribution θ ↦ q(⋅∣θ):

    # Here MeanField approximation
    # theta is the parameter vector in the unconstrained space
    dim_q = num_params * 2
    half_dim_q = num_params

    function getq(theta::AbstractArray{Float32})
        Distributions.MultivariateNormal(
            theta[1:half_dim_q],
            StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
        )
    end

    # >>>>>>>>>>>>>>>> Manual training loop <<<<<<<<<<<<<<<<<
    num_steps = num_steps
    samples_per_step = samples_per_step

    n_runs = n_runs
    elbo_trace = zeros32(num_steps, n_runs)

    posteriors = Dict()

    batch_size = Int(n_individuals / n_batches)
    elbo_trace_batch = zeros32(num_steps * n_batches, n_runs)
    theta = randn32(dim_q)

    for chain in range(1, n_runs)

        println("Chain number: ", chain)

        # Define objective
        variational_objective = Turing.Variational.ELBO()

        # Optimizer
        optimizer = DecayedADAGrad()

        # VI algorithm
        alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

        # --- Train loop ---
        converged = false
        step = 1
        if length(theta_start) > 1
            theta = theta_start
        else
            theta = randn32(dim_q) * theta_init_noise
        end

        prog = ProgressMeter.Progress(num_steps, 1)
        diff_results = DiffResults.GradientResult(theta)

        counter = 1

        while (step ≤ num_steps) && !converged
            batch_indexes = collect(Base.Iterators.partition(Random.shuffle(1:n_individuals), batch_size))

            for batch = 1:n_batches
                X_batch = data_dict["X"][batch_indexes[batch], :]
                y_batch = data_dict["y"][batch_indexes[batch]]

                # 1. Compute gradient and objective value; results are stored in `diff_results`
                batch_log_joint(theta_hat::AbstractArray{Float32}) = log_joint(
                    theta_hat,
                    X_batch=X_batch,
                    y_batch=y_batch,
                    n_batches=n_batches
                )

                AdvancedVI.grad!(
                    variational_objective,
                    alg,
                    getq,
                    batch_log_joint,
                    theta,
                    diff_results,
                    samples_per_step
                )

                # # 2. Extract gradient from `diff_result`
                gradient_step = DiffResults.gradient(diff_results)

                # # 3. Apply optimizer, e.g. multiplying by step-size
                diff_grad = apply!(optimizer, theta, gradient_step)

                # 4. Update parameters
                @. theta = theta - diff_grad

                elbo_trace_batch[counter, chain] = AdvancedVI.elbo(
                    alg,
                    getq(theta),
                    batch_log_joint,
                    samples_per_step
                )

                counter += 1
            end # end bacth loop

            # 5. Do whatever analysis you want - Store ELBO value
            full_log_joint(theta_hat::AbstractArray{Float32}) = log_joint(
                theta_hat,
                X_batch=data_dict["X"],
                y_batch=data_dict["y"],
                n_batches=1
            )

            elbo_trace[step, chain] = AdvancedVI.elbo(alg, getq(theta), full_log_joint, samples_per_step)

            step += 1

            ProgressMeter.next!(prog)
        end

        q = getq(theta)
        posteriors["$(chain)"] = q

    end

    return posteriors, elbo_trace, elbo_trace_batch, params_dict, theta

end
