# Test Zygote AD on product distribution
using Distributions
using Random
using StatsFuns
using ADTypes
using Zygote
using Turing
using AdvancedVI



p = 5
n = 100

Random.seed!(123)
beta = [-1., 1., -1., 1., -1.]
X = randn(n, p)
y = X * beta + randn(n)


function log_joint(theta::AbstractArray{<:Real})
    loglik = logpdf(MultivariateNormal(X * theta, ones(n)), y)
    log_prior = logpdf(Turing.filldist(Normal(0., 5.), p), theta)
    loglik + log_prior
end


half_n_params = p
n_params = p * 2

Random.seed!(1243)
theta = randn(n_params)

function get_q_multivariate(theta::AbstractArray{<:Real})
    Distributions.MultivariateNormal(
        theta[1:half_n_params],
        StatsFuns.softplus.(theta[(half_n_params+1):n_params])
    )
end


function get_q_product(theta::AbstractArray{<:Real})
    mu_vec = theta[1:half_n_params]
    sigma_vec = StatsFuns.softplus.(theta[(half_n_params+1):n_params])

    Turing.arraydist([
        Normal(mu_vec[w], sigma_vec[w]) for w in range(1, half_n_params)
    ])
end


# ----------- Defining q via Bijectors -----------
dists = []
for jj = 1:p
    push!(dists, Normal())
end

ranges = []
idx = 1

for i = 1:length(dists)
    d = dists[i]
    push!(ranges, idx:idx + length(d) - 1)
    # global idx
    idx += length(d)
end

# Base distribution; mean-field normal
num_params = ranges[end][end]
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(num_params), ones(num_params))

# Construct the constrained transform
to_constrained_bij = Bijectors.bijector.(dists)
# stacked bijectors
stack_con_bij = Bijectors.Stacked(to_constrained_bij, ranges)

# Construct the inverse
to_unconstrained_bij = Bijectors.inverse.(to_constrained_bij)
# stacked bijectors
stack_ucon_bij = Bijectors.Stacked(to_unconstrained_bij, ranges)

# transformed bsae dist
t_base_dist = Bijectors.transformed(base_dist, sb)
rand(t_base_dist)


function get_q_bijector(theta::AbstractArray{<:Real})
    mu_vec = theta[1:half_n_params]
    sigma_vec = StatsFuns.softplus.(theta[(half_n_params+1):n_params])

    # Define full transformation to constrained space
    full_inv_bj = stack_con_bij ∘ Bijectors.Shift(mu_vec) ∘ Bijectors.Scale(sigma_vec)

    return Bijectors.transformed(base_dist, full_inv_bj)
end

q_multivariate = get_q_multivariate(theta)
q_product = get_q_product(theta)
q_bijector = get_q_bijector(theta)

entropy(q_multivariate)
# 10.743385604419704
entropy(q_product)
# 10.743385604419705
entropy(q_bijector.dist) + Bijectors.with_logabsdet_jacobian(q_bijector.transform, rand(q_bijector))[2]

# Define objective
variational_objective = Turing.Variational.ELBO()

# VI algorithm
MC_samples = 1000
num_steps = 100
alg = AdvancedVI.ADVI(MC_samples, num_steps, adtype=ADTypes.AutoZygote())

# reproduce the steps in Zygote ADVI function from the AdvancedVI library
Random.seed!(355)
test_theta = randn(n_params)

# with MultivariateNormal
f_multivariate(t) = -variational_objective(alg, get_q_multivariate(t), log_joint, MC_samples)

f_multivariate(vcat(zeros(p), ones(p)))
f_multivariate(vcat(ones(p), ones(p)))

f_m, back = Zygote.pullback(f_multivariate, test_theta)
print(f_m)
dy_m = first(back(1.0))
print(dy_m)

function elbo_m(t)
    q = get_q_multivariate(t)
    z = rand(q)
    entropy(q) + log_joint(z)
end

f_m, back = Zygote.pullback(elbo_m, theta)
dy_m = first(back(1.0))
print(dy_m)

elbo_m(ones(p*2))
elbo_m(ones(p*2) * 0.5)


# with Bijector dist
f_bijector(t) = -variational_objective(alg, get_q_bijector(t), log_joint, MC_samples)

f_bijector(vcat(zeros(p), ones(p)))
f_bijector(vcat(ones(p), ones(p)))
f_bijector(test_theta)

f_b, back = Zygote.pullback(f_bijector, test_theta)
print(f_b)
dy_b = first(back(1.0))stack_con_bij
print(dy_b)


# With arraydist
f_product(t) = -variational_objective(alg, get_q_product(t), log_joint, MC_samples)

f_p, back = Zygote.pullback(f_product, test_theta)
dy_p = first(back(1.0))
print(dy_p)

function elbo_p(t)
    q = get_q_product(t)
    z = rand(q)
    # entropy(q) + log_joint(z)
    logpdf(q, z) + log_joint(z)
end

f_p, back = Zygote.pullback(elbo_p, theta)
dy_p = first(back(1.0))
print(dy_p)

elbo_p(ones(p*2))
elbo_p(ones(p*2) * 0.5)

###
Random.seed!(355)
theta = randn(10)

function qm(theta::AbstractArray{<:Real})
    p = size(theta)[1]
    p_half = Int(p / 2)
    dist = Distributions.MultivariateNormal(
        theta[1:p_half],
        StatsFuns.softplus.(theta[(p_half+1):p])
    )

    entropy(dist)
end

function qp(theta::AbstractArray{<:Real})
    p = size(theta)[1]
    p_half = Int(p / 2)

    mu_vec = theta[1:p_half]
    sigma_vec = StatsFuns.softplus.(theta[(p_half+1):p])

    dist = Turing.arraydist([
        Normal(mu_vec[w], sigma_vec[w]) for w in range(1, p_half)
    ])

    # logpdf(dist, ones(p_half) * 0.5)
    entropy(dist)
end


# with MultivariateNormal
f_m, back = Zygote.pullback(qm, theta)
println(f_m)
dy_m = first(back(1.0))


p = size(theta)[1]
p_half = Int(p / 2)
dist = Distributions.MultivariateNormal(
    theta[1:p_half],
    StatsFuns.softplus.(theta[(p_half+1):p])
)
entropy(dist)

f_m, back = Zygote.pullback(entropy, dist)


# With arraydist
f_p, back = Zygote.pullback(qp, theta)
dy_p = first(back(1.0))
