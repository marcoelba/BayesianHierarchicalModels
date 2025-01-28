# Simulations linear model
using CSV
using DataFrames

using Optimisers
using Distributions
using DistributionsAD
using LogExpFunctions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "my_optimisers.jl"))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "training_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "plot_utils.jl"))

include(joinpath(abs_project_path, "src", "model_building", "bijectors_extension.jl"))
include(joinpath(abs_project_path, "src", "model_building", "variational_distributions.jl"))

include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))
include(joinpath(abs_project_path, "src", "utils", "mixed_models_data_generation.jl"))
include(joinpath(abs_project_path, "src", "model_building", "mirror_statistic.jl"))



n_individuals = 300

p = 1000
prop_non_zero = 0.05
p1 = Int(p * prop_non_zero)
p0 = p - p1
corr_factor = 0.5

num_iter = 1000
MC_SAMPLES = 1000
fdr_target = 0.1
n_simulations = 10
random_seed = 1234
simulations_models = Dict()
simulations_metrics = Dict()

# label_files = "algo_HS_linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"
label_files = "algo_product_linear_n$(n_individuals)_p$(p)_active$(p1)_r$(Int(corr_factor*100))"

params_dict = OrderedDict()

# beta 0
update_parameters_dict(
    params_dict;
    name="beta0",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_normal(x),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=identity),
    init_z=[0., 0.1]
)

# beta fixed
update_parameters_dict(
    params_dict;
    name="tau_beta",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> DistributionsLogPdf.log_half_cauchy(x, sigma=1.),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(0.1, 0.1)
)

update_parameters_dict(
    params_dict;
    name="sigma_beta",
    dim_theta=(p, ),
    logpdf_prior=x::AbstractArray -> DistributionsLogPdf.log_beta(x, 0.5, 0.5),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=LogExpFunctions.logistic),
    init_z=vcat(randn(p)*0.1, randn(p)*0.1),
    dependency=[]
)
update_parameters_dict(
    params_dict;
    name="beta",
    dim_theta=(p, ),
    logpdf_prior=(x::AbstractArray, sigma::AbstractArray, tau::Real) -> DistributionsLogPdf.log_normal(x, sigma=sigma .* tau),
    dim_z=p*2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_mv_normal(z; bij=identity),
    init_z=vcat(randn(p)*0.01, randn(p)*0.01),
    dependency=["sigma_beta", "tau_beta"]
)

# sigma y
update_parameters_dict(
    params_dict;
    name="sigma_y",
    dim_theta=(1, ),
    logpdf_prior=x::Real -> Distributions.logpdf(
        truncated(Normal(0f0, 0.5f0), 0f0, Inf32),
        x
    ),
    dim_z=2,
    vi_family=z::AbstractArray -> VariationalDistributions.vi_normal(z; bij=LogExpFunctions.log1pexp),
    init_z=vcat(1., 0.5)
)

prior_position = params_dict["tuple_prior_position"]


for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )
    
    function model(theta::AbstractArray; X::AbstractArray, prior_position=prior_position)
        beta_reg = theta[prior_position[:beta]] .* theta[prior_position[:sigma_beta]]
        mu = X * beta_reg .+ theta[prior_position[:beta0]]
        return (mu, ones(eltype(mu), size(mu)) .* theta[prior_position[:sigma_y]])
    end
    
    # Training
    z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
    optimiser = MyOptimisers.DecayedADAGrad()
    # optimiser = Optimisers.RMSProp(0.01)
    
    res = hybrid_training_loop(
        z=z,
        y=data_dict["y"],
        X=data_dict["X"],
        params_dict=params_dict,
        model=model,
        log_likelihood=DistributionsLogPdf.log_normal,
        log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
        n_iter=num_iter,
        optimiser=optimiser,
        save_all=false,
        use_noisy_grads=false,
        elbo_samples=3
    )
    
    z = res["best_iter_dict"]["best_z"]

    q = VariationalDistributions.get_variational_dist(
        z,
        params_dict["vi_family_array"],
        params_dict["ranges_z"]
    )
    
    plot(res["loss_dict"]["loss"])
    beta = rand(q[prior_position[:beta]], 500)
    density(beta', label=false)

    simulations_models[simu] = (q, res["loss_dict"]["loss"], params_dict)

    # ------ Mirror Statistic ------

    ms_dist = MirrorStatistic.posterior_ms_coefficients(q[prior_position[:beta]].dist)

    metrics = MirrorStatistic.optimal_inclusion(
        ms_dist_vec=ms_dist,
        mc_samples=MC_SAMPLES,
        beta_true=data_dict["beta"],
        fdr_target=0.1
    )
    metrics_dict = Dict()

    # Posterior
    inclusion_probs = mean(metrics.inclusion_matrix, dims=2)[:, 1]
    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(inclusion_probs, fdr_target)

    metrics_posterior = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        selection
    )
    
    metrics_dict["metrics_posterior"] = metrics_posterior

    # Monte Carlo loop
    mc_samples = 2000
    ms_samples = Int(mc_samples / 2)
    mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
    beta = rand(q[prior_position[:beta]], mc_samples) .* mean_sigma

    ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
    opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
    inclusion_matrix = ms_coeffs .> opt_t
    mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

    c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
        mean_inclusion_per_coef,
        fdr_target
    )

    metrics_mc = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        selection
    )
    metrics_dict["metrics_mc"] = metrics_mc


    simulations_metrics[simu] = metrics_dict

end


mc_fdr = []
mc_tpr = []
posterior_fdr = []
posterior_tpr = []

for simu = 1:n_simulations
    push!(mc_fdr, simulations_metrics[simu]["metrics_mc"].fdr)
    push!(mc_tpr, simulations_metrics[simu]["metrics_mc"].tpr)

    push!(posterior_fdr, simulations_metrics[simu]["metrics_posterior"].fdr)
    push!(posterior_tpr, simulations_metrics[simu]["metrics_posterior"].tpr)
end

all_metrics = hcat(mc_fdr, posterior_fdr, mc_tpr, posterior_tpr)
df = DataFrame(all_metrics, ["mc_fdr", "posterior_fdr", "mc_tpr", "posterior_tpr"])

CSV.write(
    joinpath(abs_project_path, "results", "simulations", "$(label_files).csv"),
    df
)


plt = violin([1], posterior_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], posterior_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], posterior_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], posterior_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))

# MC
plt = violin([1], mc_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], mc_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], mc_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], mc_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot.pdf"))

# ----------- Data Splitting -----------
include(joinpath(abs_project_path, "src", "utils", "variable_selection_plus_inference.jl"))
ds_fdr = []
ds_tpr = []

for simu = 1:n_simulations

    println("Simulation: $(simu)")

    data_dict = generate_linear_model_data(
        n_individuals=n_individuals,
        p=p, p1=p1, p0=p0, corr_factor=corr_factor,
        random_seed=random_seed + simu
    )

    res = variable_selection_plus_inference.lasso_plus_ols(;
        X1=Float64.(data_dict["X"][1:Int(n_individuals/2), :]),
        X2=Float64.(data_dict["X"][Int(n_individuals/2)+1:end, :]),
        y1=Float64.(data_dict["y"][1:Int(n_individuals/2)]),
        y2=Float64.(data_dict["y"][Int(n_individuals/2)+1:end]),
        add_intercept=true,
        alpha_lasso=1.
    )

    beta_1 = res[1]
    beta_2 = res[2]

    mirror_coeffs = MirrorStatistic.mirror_statistic(beta_1, beta_2)

    opt_t = MirrorStatistic.get_t(mirror_coeffs; fdr_target=fdr_target)
    n_selected = sum(mirror_coeffs .> opt_t)

    metrics = MirrorStatistic.wrapper_metrics(
        data_dict["beta"] .!= 0.,
        mirror_coeffs .> opt_t
    )

    push!(ds_fdr, metrics.fdr)
    push!(ds_tpr, metrics.tpr)

end

plt = violin([1], ds_fdr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([1], ds_fdr, label=false, color="blue", fillalpha=0.1, linewidth=2)

violin!([2], ds_tpr, color="lightblue", label=false, alpha=1, linewidth=0)
boxplot!([2], ds_tpr, label=false, color="blue", fillalpha=0.1, linewidth=2)

xticks!([1, 2], ["FDR", "TPR"], tickfontsize=15)
yticks!(range(0, 1, step=0.1), tickfontsize=15)

savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_fdrtpr_boxplot_DS.pdf"))


# --------------------------------------------------------------------
# Single Run of Bayesian Model

data_dict = generate_linear_model_data(
    n_individuals=n_individuals,
    p=p, p1=p1, p0=p0, corr_factor=corr_factor,
    random_seed=random_seed
)

function model(theta::AbstractArray; X::AbstractArray, prior_position=prior_position)
    beta_reg = theta[prior_position[:beta]] .* theta[prior_position[:sigma_beta]]
    mu = X * beta_reg .+ theta[prior_position[:beta0]]
    return (mu, ones(eltype(mu), size(mu)) .* theta[prior_position[:sigma_y]])
end

# Training
z = VariationalDistributions.get_init_z(params_dict, dtype=Float64)
optimiser = MyOptimisers.DecayedADAGrad()
# optimiser = Optimisers.RMSProp(0.01)

res = hybrid_training_loop(
    z=z,
    y=data_dict["y"],
    X=data_dict["X"],
    params_dict=params_dict,
    model=model,
    log_likelihood=DistributionsLogPdf.log_normal,
    log_prior=x::AbstractArray -> compute_logpdf_prior(x; params_dict=params_dict),
    n_iter=num_iter,
    optimiser=optimiser,
    save_all=false,
    use_noisy_grads=false,
    elbo_samples=3
)

z = res["best_iter_dict"]["best_z"]

q = VariationalDistributions.get_variational_dist(
    z,
    params_dict["vi_family_array"],
    params_dict["ranges_z"]
)

plot(res["loss_dict"]["loss"])
beta_samples = rand(q[prior_position[:beta]], 500)
sigma_beta_samples = rand(q[prior_position[:sigma_beta]], 500)

density(beta_samples', label=false)
density(beta_samples' .* sigma_beta_samples', label=false)
density(beta_samples' .* mean(sigma_beta_samples, dims=2)[:, 1]', label=false)


plt = density(beta_samples' .* sigma_beta_samples', label=false)
ylabel!("Density")
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_posterior_beta.pdf"))

# ------ Mirror Statistic ------

# Monte Carlo loop
mc_samples = 2000
ms_samples = Int(mc_samples / 2)

mean_sigma = mean(rand(q[prior_position[:sigma_beta]], mc_samples), dims=2)[:, 1]
beta = rand(q[prior_position[:beta]], mc_samples) .* mean_sigma

ms_coeffs = MirrorStatistic.mirror_statistic(beta[:, 1:ms_samples], beta[:, ms_samples+1:mc_samples])
opt_t = MirrorStatistic.get_t(ms_coeffs; fdr_target=0.1)
inclusion_matrix = ms_coeffs .> opt_t
n_inclusion_per_mc = sum(inclusion_matrix, dims=1)[1, :]
mean_inclusion_per_coef = mean(inclusion_matrix, dims=2)[:, 1]

c_opt, selection = MirrorStatistic.posterior_fdr_threshold(
    mean_inclusion_per_coef,
    fdr_target
)

metrics = MirrorStatistic.wrapper_metrics(
    data_dict["beta"] .!= 0.,
    selection
)


plt_n = histogram(n_inclusion_per_mc, bins=10, label=false, normalize=true)
xlabel!("# variables included", labelfontsize=15)
vline!([mean(n_inclusion_per_mc)], color = :red, linewidth=5, label="average")
display(plt_n)
savefig(plt_n, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_n_vars_included.pdf"))

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
savefig(plt_probs, joinpath(abs_project_path, "results", "ms_analysis", "$(label_files)_mean_selection_matrix.pdf"))

plt = plot(plt_n, plt_probs)
savefig(plt, joinpath(abs_project_path, "results", "simulations", "$(label_files)_n_and_probs.pdf"))


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
