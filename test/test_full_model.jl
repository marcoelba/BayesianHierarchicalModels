# Test with noise injected to covariates
using CSV
using DataFrames

using OrderedCollections
using Distributions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "posterior_utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "model_building", "vectorised_bijectors.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))


n_individuals = 200

p = 200
prop_non_zero = 0.1
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

n_chains = 1
num_iter = 2000
MC_SAMPLES=2000
fdr_target = 0.1
n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

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
    log_prob_fun=x::AbstractArray{Float32} -> DistributionsLogPdf.log_half_cauchy(x, sigma=Float32.(ones(p)*1))
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
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    )
)

theta_axes, theta_components = get_parameters_axes(params_dict)

# --------------------------------------------------------------------
# Single Run of Bayesian Model

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed+1
)

# model predictions
model(theta_components) = Predictors.linear_model(
    theta_components;
    X=data_dict["X"] .+ Float32.(randn(size(data_dict["X"]))*0.5)
)

# model log joint
partial_log_joint(theta) = log_joint(
    theta;
    params_dict=params_dict,
    theta_axes=theta_axes,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    label=data_dict["y"]
)
partial_log_joint(theta_components)

# VI distribution
vi_dist(z::AbstractArray) = VariationalDistributions.meanfield(z, tot_params=params_dict["tot_params"])

# Training
res = training_loop(;
    log_joint=partial_log_joint,
    vi_dist=vi_dist,
    z_dim=params_dict["tot_params"]*2,
    n_iter=2000,
    n_chains=n_chains,
    samples_per_step=2,
    use_noisy_grads=false,
    n_cycles=1
)

plot(res["elbo_trace"][num_iter:end, 1])

vi_posterior = average_posterior(
    res["posteriors"],
    Distributions.MultivariateNormal
)
samples_posterior = posterior_samples(
    vi_posterior=vi_posterior,
    params_dict=params_dict,
    n_samples=MC_SAMPLES
)
beta_samples = extract_parameter(
    prior="beta",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(beta_samples', label=false)

# ------ Mirror Statistic ------

ms_dist = MirrorStatistic.posterior_ms_coefficients(
    vi_posterior=vi_posterior,
    prior="beta",
    params_dict=params_dict
)

metrics = MirrorStatistic.optimal_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=fdr_target
)

n_inclusion_per_coef = sum(metrics.inclusion_matrix, dims=2)[:,1]
mean_inclusion_per_coef = mean(metrics.inclusion_matrix, dims=2)[:,1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)
sum((1 .- mean_inclusion_per_coef) .<= c_opt)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)

plt_probs = scatter(
    findall((1 .- mean_inclusion_per_coef) .> c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .> c_opt)],
    label=false, markersize=3,
)
scatter!(
    findall((1 .- mean_inclusion_per_coef) .<= c_opt),
    mean_inclusion_per_coef[findall((1 .- mean_inclusion_per_coef) .<= c_opt)],
    label="Selected", markersize=5,
)
xlabel!("Coefficients", labelfontsize=15)
ylabel!("Inclusion Probability", labelfontsize=15)
vspan!(plt_probs, [p0+1, p], color = :green, alpha = 0.2, labels = "true active coefficients")
display(plt_probs)

plt = plot(plt_n, plt_probs)


# range
boxplot(metrics.fdr_range, label="FDR")
boxplot!(metrics.tpr_range, label="TPR")

metrics.metrics_mean
metrics.metrics_median
plt = scatter_sel_matrix(metrics.inclusion_matrix, p0=p0)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_inclusion_probs.pdf"))

metrics.metrics_relative

range_p = collect(range(1, p))
scatter(metrics.relative_inclusion_freq, markersize=3, label="Sorted relative inclusion freq")
hline!([metrics.min_inclusion_freq], label="Cutoff inclusion", linewidth=2)
sum(metrics.relative_inclusion_freq .> metrics.min_inclusion_freq)

excluded = metrics.relative_inclusion_freq .<= metrics.min_inclusion_freq
included = metrics.relative_inclusion_freq .> metrics.min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], metrics.relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], metrics.relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

# Posterior
inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)
sum(selection)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)

MirrorStatistic.posterior_ms_inclusion(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=0.1
)


metrics = MirrorStatistic.fdr_distribution(
    ms_dist_vec=ms_dist,
    mc_samples=MC_SAMPLES,
    beta_true=data_dict["beta"],
    fdr_target=fdr_target
)

metrics.metrics_relative

excluded = metrics.relative_inclusion_freq .<= metrics.min_inclusion_freq
included = metrics.relative_inclusion_freq .> metrics.min_inclusion_freq
range_p = collect(range(1, p))

plt = scatter(range_p[excluded], metrics.relative_inclusion_freq[excluded], label="Out")
scatter!(range_p[included], metrics.relative_inclusion_freq[included], label="In")
vspan!(plt, [p0 + 1, p], color = :green, alpha = 0.2, labels = "true active coefficients")

# Using lambda as inclusion probabilities
sigma_samples = extract_parameter(
    prior="sigma_y",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
mean(sigma_samples)

lambda_samples = extract_parameter(
    prior="sigma_beta",
    params_dict=params_dict,
    samples_posterior=samples_posterior
)
plt = density(lambda_samples', label=false)
mean_lambda = mean(lambda_samples, dims=2)[:, 1]
scatter(mean_lambda)
scatter(1 .- 1 ./ (1 .+ mean_lambda.^2 .* mean(sigma_samples)) )
probs_inc = 1 .- 1 ./ (1 .+ mean_lambda.^2)

function fdp_estimate(ms_samples, probs, t)
    
    Dp = ms_samples .> t
    
    # d(t)(w > t) * (1 - r)
    fp_est = sum(Dp .* (1 .- probs))
    tot_d = sum(Dp)
    fdp = fp_est / tot_d
    
    return fdp
end

function mc_fdp_estimate(ms_samples, probs, t)
    fdr_MC_t = []
    for m in eachcol(ms_samples)
        push!(fdr_MC_t, fdp_estimate(m, probs, t))
    end
    return fdr_MC_t
end

ms_samples = rand(ms_dist, MC_SAMPLES)

sum(ms_samples .> 0.5, dims=2) / MC_SAMPLES

fdp_estimate(ms_samples[:, 1], probs_inc, 0.8)
mc_fdp_estimate(ms_samples, probs_inc, 0.3)

ms_samples = rand(ms_dist, 1)
scatter(ms_samples)
fdr = []
t_range = range(minimum(abs.(ms_samples)), maximum(abs.(ms_samples)), length=100)
t_range = sort(abs.(ms_samples[:, 1]))
for t in t_range
    push!(fdr, mean(mc_fdp_estimate(ms_samples, probs_inc, t)))
end
scatter(t_range, fdr)

opt_t = 0
for t in t_range
    if mean(mc_fdp_estimate(ms_samples, probs_inc, t)) < 0.1
        opt_t = t
        break
    end
end
mean(mc_fdp_estimate(ms_samples, probs_inc, opt_t))


# -------------------------------------------------------
# Knockoffs 
using GLMNet

fdr = []
tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    Xk_0 = rand(MultivariateNormal(data_dict["cov_matrix_0"]), n_individuals)
    Xk_1 = rand(MultivariateNormal(data_dict["cov_matrix_1"]), n_individuals)
    Xk = transpose(vcat(Xk_0, Xk_1))

    X_aug = hcat(data_dict["X"], Xk)

    glm_k = GLMNet.glmnetcv(X_aug, Float64.(data_dict["y"]))
    coefs = GLMNet.coef(glm_k)

    beta_1 = coefs[1:p]
    beta_2 = coefs[p+1:2*p]

    mirror_coeffs = abs.(beta_1) .- abs.(beta_2)
    
    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(fdr, metrics.fdr)
    push!(tpr, metrics.tpr)

end

plt = violin([1], fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_Knock.pdf"))
