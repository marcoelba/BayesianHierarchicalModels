# Mixed model inference via NUTS (HMC)
using Pkg
Pkg.status()

using Turing
using Distributions
using StatsPlots
using Random
using DataFrames

include("../utils/posterior_inference.jl")

# define mixed model using Turing syntax
# Simple Random intercept only: y_l = mu + 1*eta_l + eps_l
# l = 1,...,L groups
# n_l observtions for each group: i = 1,...,n_l
# eta_l ~ Normal(mu, sigma_eta)
# mu ~ Normal(0, 5)
# sigma_eta ~ Half-Normal(1)

@model function mixed_model(y, L, n_groups)
    # Variances
    sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    sigma_eta ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    
    # coefficients:
    mu ~ Turing.Normal(0., 5.)
    eta_vec ~ Turing.MultivariateNormal(fill(mu, L), sigma_eta)
    
    for l in range(1, L)
        y[:, l] ~ Turing.MultivariateNormal(fill(eta_vec[l], n_groups[l]), sigma2_y)
    end

end


" Generate toy dataset for L=3 groups "
L = 3
n_groups = [50, 50, 50]
mu_groups = [-2, 0, 2]
sigma_y = 1.

y = zeros(maximum(n_groups), L)
for l in range(1, L)
    y[:, l] = Random.rand(Distributions.Normal(mu_groups[l], sigma_y), n_groups[l])
end

nuts_mm = sample(mixed_model(y, L, n_groups), NUTS(0.65), 1000)

df_nuts_chains = DataFrames.DataFrame(nuts_mm)

plot(df_nuts_chains[!, "mu"])
plot!(df_nuts_chains[!, "eta_vec[1]"])
plot!(df_nuts_chains[!, "eta_vec[2]"])
plot!(df_nuts_chains[!, "eta_vec[3]"])

# Sigma eta
plot(df_nuts_chains[!, "sigma_eta"])


" Generate dataset for L=10 groups "
L = 10
n_groups = fill(50, L)
sigma_eta = 1.
eta_rv = Distributions.Normal(0., sigma_eta)
mu_groups = Random.rand(eta_rv, L)
sigma_y = 1.

y = zeros(maximum(n_groups), L)
for l in range(1, L)
    y[:, l] = Random.rand(Distributions.Normal(mu_groups[l], sigma_y), n_groups[l])
end

nuts_mm = sample(mixed_model(y, L, n_groups), NUTS(0.65), 1000)
Turing.summarystats(nuts_mm)

df_nuts_chains = DataFrames.DataFrame(nuts_mm)

plot(df_nuts_chains[!, "mu"])
plot!(df_nuts_chains[!, "eta_vec[1]"])
plot!(df_nuts_chains[!, "eta_vec[2]"])


df_hpd = get_parameters_interval(
    nuts_mm;
    interval_type="HPD",
    alpha_prob=0.05
)
df_ci = get_parameters_interval(
    nuts_mm;
    interval_type="CI",
    alpha_prob=0.05
)

density(df_nuts_chains[!, "mu"])
vline!(
    Matrix(df_hpd[df_hpd.parameters .== :mu, ["lower", "upper"]])',
    label="HPD"
)
vline!(
    Matrix(df_ci[df_ci.parameters .== :mu, ["lower", "upper"]])',
    label="CI"
)


" Non-Central parametrisation "

@model function nc_mixed_model(y, L, n_groups)
    # Variances
    sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    sigma_eta ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    
    # coefficients:
    mu ~ Turing.Normal(0., 5.)
    eta_z ~ Turing.MultivariateNormal(zeros(L), 1.)
    eta_vec = fill(mu, L) + eta_z .* sigma_eta
    
    for l in range(1, L)
        y[:, l] ~ Turing.MultivariateNormal(fill(eta_vec[l], n_groups[l]), sigma2_y)
    end

end


nuts_mm = sample(nc_mixed_model(y, L, n_groups), NUTS(0.65), 1000)
Turing.summarystats(nuts_mm)
# Worse mixing
df_nuts_chains = DataFrames.DataFrame(nuts_mm)

plot(df_nuts_chains[!, "mu"])


df_hpd = get_parameters_interval(
    nuts_mm;
    interval_type="HPD",
    alpha_prob=0.05
)
df_ci = get_parameters_interval(
    nuts_mm;
    interval_type="CI",
    alpha_prob=0.05
)


" Mixed Model (Hierarchical) with Covariates "
# l = 1,...,L groups
# n_l observtions for each group: i = 1,...,n_l
# Simple Random intercept only: y_l = mu + eta_l + eps_l
# mu_i = Xi * beta_x
# eta_l = Z_l * beta_zl
# beta_x ~ Normal(0, 5)
# beta_zl ~ Normal(mu, sigma_eta)
# sigma_eta ~ Half-Normal(1)

@model function hierarchical_model(y, X, Q, L, n_groups)
    p_fixed = size(X)[3]
    p_random = size(Q)[3]
    # container for beta Random
    beta_q = zeros(L, p_random)

    # Variances
    sigma2_y ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    sigma_eta ~ Turing.TruncatedNormal(1., 1., 0., Inf)
    
    # Coefficients:
    beta_x ~ Turing.MultivariateNormal(zeros(p_fixed), sqrt(10.))
    # beta_q ~ Turing.MultivariateNormal(zeros(L), sigma_eta)
    
    # Likelihood
    for l in range(1, L)
        beta_q[l, :] ~ Turing.MultivariateNormal(zeros(p_random), sigma_eta)
        mu = Q[l, :, :] * beta_q[l, :] + X[l, :, :] * beta_x
        y[:, l] ~ Turing.MultivariateNormal(mu, sigma2_y)
    end

end


# Generate dataset for L=3 groups and p_x=3 covs and Z=1 (only intercept)
L = 3
p_f = 3
p_r = 2
n_groups = fill(100, L)
# sigma_eta = 0.5
sigma_y = 1.
# beta_x is (p_x)
beta_x = [1., -1., 1.]
# beta_z is (p_x, L)
beta_z = [[0., 1.], [1., 1.5], [0.5, 1.]]

x_distr = Distributions.Normal()
X = ones(L, maximum(n_groups), p_f)
for l in range(1, L)
    X[l, :, 2:p_f] = Random.rand(x_distr, n_groups[l], p_f - 1)
end

# Only intercept here
q_distr = Distributions.Normal()
Q = ones(L, maximum(n_groups), p_r)
for l in range(1, L)
    Q[l, :, 2:p_r] = Random.rand(q_distr, n_groups[l], p_r - 1)
end

y = zeros(maximum(n_groups), L)
for l in range(1, L)
    lin_pred_l = X[l, :, :] * beta_x + Q[l, :, :] * beta_z[l]
    y[:, l] = Random.rand(
        Distributions.MultivariateNormal(lin_pred_l, sigma_y),
    )
end

# MCMC
nuts_samples = sample(
    hierarchical_model(y, X, Q, L, n_groups),
    NUTS(0.65),
    1000
)
Turing.summarystats(nuts_samples)
df_nuts_chains = DataFrames.DataFrame(nuts_samples)

plot(df_nuts_chains[!, "beta_x[1]"])
plot!(df_nuts_chains[!, "beta_x[2]"])
plot!(df_nuts_chains[!, "beta_x[3]"])


df_hpd = get_parameters_interval(
    nuts_samples;
    interval_type="HPD",
    alpha_prob=0.05
)

df_ci = get_parameters_interval(
    nuts_samples;
    interval_type="CI",
    alpha_prob=0.05
)
