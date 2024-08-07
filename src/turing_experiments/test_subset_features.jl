# Test subset features
using Distributions
using Random
using LinearAlgebra
using ToeplitzMatrices
using Turing
using StatsPlots
using StatsFuns
using ProgressMeter
using DiffResults
using ADTypes
using Zygote
using AdvancedVI


include(joinpath("decayed_ada_grad.jl"))


n = 100
p = 2
beta_true = rand([-1., 1.], p)


# Correlation
corr_factor = 0.99
cor_coefs = vcat([1.], [corr_factor * (p - ll) / (p - 1) for ll in range(1, p-1)])
cov_matrix = Array(Toeplitz(cor_coefs, cor_coefs))

X = rand(MultivariateNormal(cov_matrix), n)'
y = X * beta_true .+ randn(n) * 0.5


@model function lin_model(y, X)
    p = size(X, 2)

    # Variance
    sigma_y ~ truncated(Normal(0., 1.), 0., Inf)

    # Covariates
    beta ~ Turing.MultivariateNormal(zeros(p), 5.)

    # Intercept
    beta0 ~ Turing.Normal(0., 5.)

    mu = beta0 .+ X * beta
    y ~ Turing.MultivariateNormal(mu, sigma_y)
end

model = lin_model(y, X[:, 1:1])

nuts_lm = sample(model, NUTS(0.65), 1000)
plot(nuts_lm)


model = lin_model(X[:, 2:2], X[:, 1:1])

nuts_lm = sample(model, NUTS(0.65), 1000)
plot(nuts_lm)

inv(X * X')


# Variational Inference
function vi_lin_model(y, X)
    _p = size(X, 2)

    # Variational Inference
    function log_likelihood(; y, X, beta, beta0, sd_y)
        logpdf(
            MultivariateNormal(X*beta .+ beta0, sd_y), y
        )
    end

    # Joint
    function log_joint(theta_hat)

        beta = theta_hat[1:_p]
        beta0 = theta_hat[_p + 1]
        sd_y = StatsFuns.softplus(theta_hat[_p + 2])

        loglik = log_likelihood(
            y=y, X=X, beta=beta, beta0=beta0, sd_y=sd_y
        )

        log_prior = logpdf(MvNormal(zeros(_p), I*5.), beta) +
            logpdf(Normal(0., 5.), beta0) +
            logpdf(truncated(Normal(0., 0.5), 0., Inf), sd_y)
            
        loglik + log_prior
    end

    # Variational Distribution
    num_params = _p + 1 + 1

    dim_q = num_params * 2
    half_dim_q = num_params

    function getq(theta)
        Distributions.MultivariateNormal(
            theta[1:half_dim_q],
            StatsFuns.softplus.(theta[(half_dim_q+1):dim_q])
        )
    end

    num_steps = 1000
    samples_per_step = 5

    elbo_trace = zeros(num_steps)
    theta_trace = zeros(num_steps, num_params)

    # Define objective
    variational_objective = Turing.Variational.ELBO()

    # Optimizer
    optimizer = DecayedADAGrad()

    # VI algorithm
    alg = AdvancedVI.ADVI(samples_per_step, num_steps, adtype=ADTypes.AutoZygote())

    # --- Train loop ---
    converged = false
    step = 1
    theta = randn(dim_q) * 0.2f0

    prog = ProgressMeter.Progress(num_steps, 1)
    diff_results = DiffResults.GradientResult(theta)

    while (step ≤ num_steps) && !converged
        # 1. Compute gradient and objective value; results are stored in `diff_results`
        AdvancedVI.grad!(variational_objective, alg, getq, log_joint, theta, diff_results, samples_per_step)

        # # 2. Extract gradient from `diff_result`
        gradient_step = DiffResults.gradient(diff_results)

        # # 3. Apply optimizer, e.g. multiplying by step-size
        diff_grad = apply!(optimizer, theta, gradient_step)

        # 4. Update parameters
        @. theta = theta - diff_grad

        # 5. Do whatever analysis you want - Store ELBO value
        q_temp = getq(theta)

        theta_trace[step, 1:_p] = q_temp.μ[1:_p]
        theta_trace[step, _p+1] = q_temp.μ[_p+1]
        theta_trace[step, _p+2] = StatsFuns.softplus(q_temp.μ[_p+2])

        elbo_trace[step] = AdvancedVI.elbo(alg, q_temp, log_joint, samples_per_step)

        step += 1

        ProgressMeter.next!(prog)
    end

    q = getq(theta)

    return q, elbo_trace, theta_trace

end

p_star = 15
q, elbo_trace, theta_trace = vi_lin_model(y, X[:, 1:p_star])

plot(elbo_trace)
plot(theta_trace)

samples = rand(q, 2000)

plt = density(samples[1, :], label=false)
for j = 2:p_star
    density!(samples[j, :], label=false)
end
display(plt)
