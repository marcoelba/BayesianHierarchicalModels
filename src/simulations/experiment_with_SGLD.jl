# Experiment with SGLD
using CSV
using DataFrames

using OrderedCollections
using Distributions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))

# parameters extraction
function get_transformed_parameter(z_trace, priors; variable)
    prior_keys = keys(priors)
    bij = priors[variable]["bij"]
    range_var = priors[variable]["range"]

    return bij(z_trace[range_var, :])
end


n_individuals = 200

p = 1000
prop_non_zero = 0.02
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

fdr_target = 0.1
n_simulations = 10
random_seed = 1234

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dimension=(1, ),
    log_prob_fun=x::Float32 -> DistributionsLogPdf.log_normal(x)
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dimension=(p, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(
        x, sigma=Float32.(ones(p)*1)
    )
)
update_parameters_dict(
    params_dict;
    name="beta",
    dimension=(p, ),
    log_prob_fun=(x::AbstractArray{Float32}, sigma::AbstractArray{Float32}) -> DistributionsLogPdf.log_normal(x, sigma=sigma),
    dependency=["sigma_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dimension=(1, ),
    bij=VectorizedBijectors.softplus,
    log_prob_fun=x::Float32 -> Distributions.logpdf(
        truncated(Normal(0f0, 1f0), 0f0, Inf32),
        x
    )
)

theta_axes, _ = get_parameters_axes(params_dict)

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)

# model predictions
model(theta_components) = Predictors.linear_model(
    theta_components;
    X=data_dict["X"]
)

# model log joint
partial_log_joint(theta) = - log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)

# Training
n_iter = 6000
n_chains = 4
n_cycles = 2
steps_per_cycle = Int(n_iter / n_cycles)
n_iter_tot = n_iter
z_dim = params_dict["tot_params"]
use_noisy_grads = true

lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)

freq_store = 5
z_store_cycle_schedule = Int.(range(steps_per_cycle / 2, steps_per_cycle, step=freq_store))
z_store_schedule = vcat([z_store_cycle_schedule .+ (steps_per_cycle * cycle) for cycle = 1:(n_cycles - 1)]...)
n_mcmc_samples = length(z_store_schedule)

all_iterations = Int.(zeros(n_iter_tot))
all_iterations[z_store_schedule] .= 1

sd_init = Float32.(0.5)

posteriors = Dict()
loss_trace = zeros(n_iter_tot, n_chains)

for chain in range(1, n_chains)

    println("Chain number: ", chain)

    # Optimizer
    optimizer = MyDecayedADAGrad()

    # --- Train loop ---
    converged = false
    step = 1
    store_step = 1

    prog = ProgressMeter.Progress(n_iter_tot, 1)
    z = Float32.(randn(z_dim)) * sd_init
    z_trace = Float32.(zeros(z_dim, n_mcmc_samples))

    while (step ≤ n_iter_tot) && !converged

        loss, grads = Zygote.withgradient(partial_log_joint, z)

        # 3. Apply optimizer, e.g. multiplying by step-size
        diff_grad = apply!(optimizer, z, grads[1])

        # 4. Update parameters
        if use_noisy_grads
            grad_noise = Float32.(randn(z_dim)) .* lr_schedule[step]
        else
            grad_noise = 0.
        end

        @. z = z - diff_grad + grad_noise
        
        # 5. Do whatever analysis you want - Store ELBO value
        loss_trace[step, chain] = loss

        if all_iterations[step] == 1
            z_trace[:, store_step] = z
            store_step += 1
        end

        step += 1
        ProgressMeter.next!(prog)
    end

    posteriors[chain] = z_trace

end

plot(loss_trace[:, 1])
plot!(loss_trace[:, 2])
plot!(loss_trace[:, 3])
plot!(loss_trace[:, 4])

plot(loss_trace[(n_iter - 1000):end, 1])
plot!(loss_trace[(n_iter - 1000):end, 2])


priors = params_dict["priors"]

z_trace = hcat([posteriors[cc] for cc = 1:n_chains]...)

beta0 = get_transformed_parameter(z_trace, priors, variable="beta0")[1,:]
beta = get_transformed_parameter(z_trace, priors, variable="beta")
sigma_beta = get_transformed_parameter(z_trace, priors, variable="sigma_beta")

histogram(beta0)
histogram(beta[1, :])
scatter(beta[p, :])
scatter(beta[p-1, :])


# Using the FDR criterion from MS
fdr_target = 0.1
mc_samples = Int(n_mcmc_samples / 2)

s1 = sample(1:n_mcmc_samples, mc_samples, replace=false)
s2 = setdiff(1:n_mcmc_samples, s1)

beta_1 = beta[:, s1]
beta_2 = beta[:, s2]
mirror_coefficients = MirrorStatistic.mirror_statistic(beta_1, beta_2)
plt_ms = scatter(mirror_coefficients[:, 1], label=false, markersize=3)

opt_t = 0
t = 0
for t in range(0., maximum(mirror_coefficients), length=2000)
    n_left_tail = sum(mirror_coefficients .< -t)
    n_right_tail = sum(mirror_coefficients .> t)
    n_right_tail = ifelse(n_right_tail .> 0, n_right_tail, 1)

    fdp = n_left_tail ./ n_right_tail

    if fdp .<= fdr_target
        opt_t = t
        break
    end
end

inclusion_matrix = mirror_coefficients .> opt_t
n_inclusion_per_mc = sum(inclusion_matrix, dims=1)[1,:]
average_inclusion_number = Int(round(mean(n_inclusion_per_mc)))

plt_n = histogram(n_inclusion_per_mc, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([average_inclusion_number], color = :red, linewidth=5, label="average")
display(plt_n)

n_inclusion_per_coef = sum(inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)
