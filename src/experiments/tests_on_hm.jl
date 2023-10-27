# Experiments on Turing models
using Pkg
Pkg.status()

using Turing
using Distributions
using StatsPlots
using Random
using DataFrames

abs_project_path = normpath(joinpath(@__FILE__,"..", ".."))
include(joinpath(abs_project_path, "utils", "posterior_inference.jl"))
include(joinpath(abs_project_path, "models", "hierarchical_models.jl"))


# Generate data from linear mixed model
n_groups = 3
p_groups = 2
p_global = 3
n = fill(100, n_groups)
# sigma_eta = 0.5
sigma_y = 1.
# beta_x is (p_x) - no intercept here
beta_global = [1., -1., 1.]
# beta groups (random effects)
beta_groups = [[0., 1.], [1., 1.5], [0.5, 1.]]

x_distr = Distributions.Normal()
X = zeros(n_groups, maximum(n), p_global)
for l in range(1, n_groups)
    X[l, :, :] = Random.rand(x_distr, n[l], p_global)
end

# group specific covs
q_distr = Distributions.Normal()
Q = ones(n_groups, maximum(n), p_groups)
for l in range(1, n_groups)
    Q[l, :, 2:p_groups] = Random.rand(q_distr, n[l], p_groups - 1)
end

y = zeros(maximum(n), n_groups)
for l in range(1, n_groups)
    lin_pred_l = X[l, :, :] * beta_global + Q[l, :, :] * beta_groups[l]
    y[:, l] = Random.rand(
        Distributions.MultivariateNormal(lin_pred_l, sigma_y),
    )
end


# HMC with normal prior model
nuts_samples = sample(
    hierarchical_models.hierarchical_lm_normal_prior(y, X, Q, n_groups),
    NUTS(0.65),
    1000
)
Turing.summarystats(nuts_samples)
df_nuts_chains = DataFrames.DataFrame(nuts_samples)

plot(df_nuts_chains[!, "beta_global[1]"])
plot!(df_nuts_chains[!, "beta_global[2]"])
plot!(df_nuts_chains[!, "beta_global[3]"])

plot(df_nuts_chains[!, "beta_groups[1,:][1]"])
plot!(df_nuts_chains[!, "beta_groups[1,:][2]"])

df_hpd = posterior_inference.get_parameters_interval(
    nuts_samples;
    interval_type="HPD",
    alpha_prob=0.05
)


" HMC with horseshoe prior model "

nuts_samples = sample(
    hierarchical_models.hierarchical_lm_hs_prior(y, X, Q, n_groups),
    NUTS(0.65),
    1000
)
Turing.summarystats(nuts_samples)
df_nuts_chains = DataFrames.DataFrame(nuts_samples)

plot(df_nuts_chains[!, "beta_global[1]"])
plot!(df_nuts_chains[!, "beta_global[2]"])
plot!(df_nuts_chains[!, "beta_global[3]"])
plot!(df_nuts_chains[!, "beta_global_int"])

plot(df_nuts_chains[!, "beta_groups[1,:][1]"])
plot!(df_nuts_chains[!, "beta_groups[1,:][2]"])

df_hpd = posterior_inference.get_parameters_interval(
    nuts_samples;
    interval_type="HPD",
    alpha_prob=0.05
)
