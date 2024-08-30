# Random Int Model
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


function random_intercept_model(;
    data_dict,
    num_steps=2000,
    n_runs=1,
    samples_per_step=4,
    cauchy_scale=1f0
    )
    # Prior distributions
    params_dict = OrderedDict()
    num_params = 0

    # Variance
    params_dict["sigma_y"] = OrderedDict("size" => (1), "from" => 1, "to" => 1, "bij" => StatsFuns.softplus)
    num_params += 1
    prior_sigma_y = truncated(Normal(0f0, 1f0), 0f0, Inf32)
    log_prior_sigma_y(sigma_y::Float32) = Distributions.logpdf(prior_sigma_y, sigma_y)

    # Fixed effects
    # ---------- Horseshoe distribution ----------
    params_dict["sigma_beta"] = OrderedDict("size" => (p), "from" => num_params+1, "to" => num_params + p, "bij" => StatsFuns.softplus)
    num_params += p
    log_prior_sigma_beta(sigma_beta::AbstractArray{Float32}) = sum(Distributions.logpdf.(
        truncated(Cauchy(0f0, cauchy_scale), 0f0, Inf32),
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

    params_dict["sigma_beta0"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
    num_params += 1
    prior_sigma_beta0 = truncated(Normal(0f0, 1f0), 0f0, Inf)
    log_prior_sigma_beta0(sigma_beta0::Float32) = Distributions.logpdf(prior_sigma_beta0, sigma_beta0)

    # Random Intercept
    params_dict["beta0_random"] = OrderedDict("size" => (n_individuals), "from" => num_params+1, "to" => num_params + n_individuals, "bij" => identity)
    num_params += n_individuals
    function log_prior_beta0_random(sigma_beta0::Float32, beta0_random::AbstractArray{Float32})
        Distributions.logpdf(
            Turing.filldist(Distributions.Normal(0f0, sigma_beta0), n_individuals),
            beta0_random
        )
    end

    # Random time component
    params_dict["sigma_beta_time"] = OrderedDict("size" => (1), "from" => num_params+1, "to" => num_params + 1, "bij" => StatsFuns.softplus)
    num_params += 1
    prior_sigma_beta_time = truncated(Normal(0f0, 5f0), 0f0, Inf32)
    log_prior_sigma_beta_time(sigma_beta_time::Float32) = Distributions.logpdf(prior_sigma_beta_time, sigma_beta_time)

    params_dict["beta_time"] = OrderedDict("size" => (n_per_ind), "from" => num_params+1, "to" => num_params + n_per_ind, "bij" => identity)
    num_params += n_per_ind
    function log_prior_beta_time(sigma_beta_time::Float32, beta_time::AbstractArray{Float32})
        Distributions.logpdf(
            Turing.filldist(Distributions.Normal(0f0, sigma_beta_time), n_per_ind),
            beta_time
        )
    end


    # Likelihood
    function likelihood(;
        beta0_fixed::Float32,
        beta0_random::AbstractArray{Float32},
        beta_fixed::AbstractArray{Float32},
        beta_time::AbstractArray{Float32},
        sigma_y::Float32,
        Xfix::AbstractArray{Float32}
        )
        Turing.arraydist([
            Distributions.MultivariateNormal(
                beta0_fixed .+ beta0_random .+ Xfix * beta_fixed .+ beta_time[tt],
                ones32(n_individuals) .* sigma_y
            ) for tt in range(1, n_per_ind)
        ])
    end
    
    function log_likelihood(;
        y::AbstractArray{Float32},
        beta0_fixed::Float32,
        beta0_random::AbstractArray{Float32},
        beta_fixed::AbstractArray{Float32},
        beta_time::AbstractArray{Float32},
        sigma_y::Float32,
        Xfix::AbstractArray{Float32}
        )
        sum(
            Distributions.logpdf(likelihood(
                beta0_fixed=beta0_fixed,
                beta0_random=beta0_random,
                beta_fixed=beta_fixed,
                beta_time=beta_time,
                sigma_y=sigma_y,
                Xfix=Xfix
            ), y)
        )
    end
    
    # Joint
    params_names = tuple(Symbol.(params_dict.keys)...)
    proto_array = ComponentArray(;
        [Symbol(pp) => ifelse(params_dict[pp]["size"] > 1, randn32(params_dict[pp]["size"]), randn32(params_dict[pp]["size"])[1])  for pp in params_dict.keys]...
    )
    proto_axes = getaxes(proto_array)
    num_params = length(proto_array)


    function log_joint(theta_hat::AbstractArray{Float32}; X_batch::AbstractArray{Float32}, y_batch::AbstractArray{Float32})
        begin
            params_names = ComponentArray(theta_hat, proto_axes)
        end

        sigma_y = params_dict["sigma_y"]["bij"].(params_names.sigma_y)

        sigma_beta = params_dict["sigma_beta"]["bij"].(params_names.sigma_beta)
        beta_fixed = params_names.beta_fixed

        beta0_fixed = params_names.beta0_fixed
        sigma_beta0 = params_dict["sigma_beta0"]["bij"].(params_names.sigma_beta0)
        beta0_random = params_names.beta0_random
    
        sigma_beta_time = params_dict["sigma_beta_time"]["bij"].(params_names.sigma_beta_time)
        beta_time = params_names.beta_time
    
        loglik = log_likelihood(
            y=y_batch,
            beta0_fixed=beta0_fixed,
            beta0_random=beta0_random,
            beta_fixed=beta_fixed,
            beta_time=beta_time,
            sigma_y=sigma_y,
            Xfix=X_batch
        )
    
        log_prior = log_prior_sigma_y(sigma_y) +
            log_prior_sigma_beta(sigma_beta) +
            log_prior_beta_fixed(sigma_beta, beta_fixed) +
            log_prior_beta0_fixed(beta0_fixed) +
            log_prior_sigma_beta0(sigma_beta0) +
            log_prior_beta0_random(sigma_beta0, beta0_random) +
            log_prior_sigma_beta_time(sigma_beta_time) +
            log_prior_beta_time(sigma_beta_time, beta_time)
        
        loglik + log_prior

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

    n_runs = n_runs
    elbo_trace = zeros32(num_steps, n_runs)

    posteriors = Dict()

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
        theta = randn32(dim_q) * 0.2f0

        prog = ProgressMeter.Progress(num_steps, 1)
        diff_results = DiffResults.GradientResult(theta)

        while (step ≤ num_steps) && !converged


            # 1. Compute gradient and objective value; results are stored in `diff_results`
            batch_log_joint(theta_hat::AbstractArray{Float32}) = log_joint(
                theta_hat,
                X_batch=data_dict["Xfix"],
                y_batch=data_dict["y"]
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

            # 5. Do whatever analysis you want - Store ELBO value
            elbo_trace[step, chain] = AdvancedVI.elbo(
                alg, getq(theta), batch_log_joint, samples_per_step
            )

            step += 1

            ProgressMeter.next!(prog)
        end

        q = getq(theta)
        posteriors["$(chain)"] = q

    end

    return posteriors, elbo_trace, params_dict

end
