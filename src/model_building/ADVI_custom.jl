# ADVI with custom variational distributions

module ADVI

using Zygote
using Bijectors
using Distributions
using DistributionsAD
using LogExpFunctions
using StatsPlots

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "model_building", "my_optimisers.jl"))


function LogExpFunctions.log1pexp(x::AbstractArray)
    LogExpFunctions.log1pexp.(x)
end


function log_joint(theta; y, X)
    pred = X * theta[1]

    loglik = sum(Distributions.logpdf(
        DistributionsAD.MvNormal(pred, theta[2]),
        y
    ))

    logprior = sum(DistributionsAD.logpdf.(DistributionsAD.Normal(0., 1.), theta[1]))
        sum(DistributionsAD.logpdf(truncated(DistributionsAD.Normal(0., 1.), 0., Inf64), theta[2]))

    return loglik + logprior
end

X = randn(100, 3)
y = X * [1, 2, 1] .+ randn(100)
p = 3


# Variational Distributions
function vi_theta_beta(z::AbstractArray)
    zdim = length(z)
    pdim = Int(zdim / 2)

    Bijectors.transformed(
        DistributionsAD.MvNormal(
            z[1:pdim],
            LogExpFunctions.log1pexp.(z[pdim+1:zdim])
        ),
        identity
    )
end

dd = vi_theta_beta(randn(p*2))
z_1_pos = 1:p*2
rand(dd)
rand(dd.dist)
dd.transform


function vi_theta_sigma(z::AbstractArray)
    Bijectors.transformed(
        DistributionsAD.Normal(z[1], LogExpFunctions.log1pexp.(z[2])),
        log1pexp
    )
end

dd2 = vi_theta_sigma(randn(2))
z_2_pos = range(1, 2) .+ p*2

rand(dd2)
rand(dd2.dist)
dd2.transform

# 
range_z = [z_1_pos, z_2_pos]
q_family_array = [vi_theta_beta, vi_theta_sigma]
dim_z = 8
z = randn(dim_z)


function get_variational_dist(z::AbstractArray, q_family_array::AbstractArray, range_z::AbstractVector)
    q_vi = [q_family_array[cc](z[range_z[cc]]) for cc in eachindex(q_family_array)]
    return q_vi
end
q_dist_array = get_variational_dist(z, q_family_array, range_z)


function rand_array(q_dist_array::AbstractArray; from_base_dist::Bool=false, reduce_to_vec::Bool=false)
    if from_base_dist
        v_sample = [rand(dist.dist) for dist in q_dist_array]
    else
        v_sample = [rand(dist) for dist in q_dist_array]
    end

    if reduce_to_vec
        v_sample = reduce(vcat, v_sample)
    end

    return v_sample
end
rand_array(q_dist_array, reduce_to_vec=true)
rand_array(q_dist_array, reduce_to_vec=false)
rand_array(q_dist_array, from_base_dist=true)

# Bijectors - Only the jacobian
function Bijectors.jacobian(t::typeof(log1pexp), x::AbstractArray, xt::AbstractArray)
    logdetjac = sum(x .- xt)
    return logdetjac
end

function Bijectors.jacobian(t::typeof(log1pexp), x::Real, xt::Real)
    logdetjac = x - xt
    return logdetjac
end

function Bijectors.jacobian(t::typeof(identity), x::AbstractArray, xt::AbstractArray)
    logdetjac = zero(eltype(x))
    return logdetjac
end

function Bijectors.jacobian(t::typeof(identity), x::Real, xt::Real)
    logdetjac = zero(eltype(x))
    return logdetjac
end


#
function rand_with_logjacobian(q_dist_array::AbstractArray)
    x = rand_array(q_dist_array, from_base_dist=true, reduce_to_vec=false)
    x_t = [q_dist_array[ii].transform(x[ii]) for ii in eachindex(x)]
    abs_jacobian = [Bijectors.jacobian(q_dist_array[ii].transform, x[ii], x_t[ii]) for ii in eachindex(x)]

    return x_t, sum(abs_jacobian)
end

# Entropy
function Distributions.entropy(d::Bijectors.TransformedDistribution)
    Distributions.entropy(d.dist)
end


function elbo(
    z::AbstractArray;
    q_family_array::AbstractArray,
    log_joint,
    n_samples::Int64=1
    )

    # get a specific distribution using the weights z, from the variational family 
    q_dist_array = get_variational_dist(z, q_family_array, range_z)

    # evaluate the log-joint
    res = zero(eltype(z))
    for mc = 1:n_samples
        theta, abs_jacobian = rand_with_logjacobian(q_dist_array)
        # evaluate the log-joint
        res += (log_joint(theta; y=y, X=X) + sum(abs_jacobian)) / n_samples
    end

    # add entropy
    for d in q_dist_array
        res += Distributions.entropy(d)
    end

    return -res
end

elbo(
    z;
    q_family_array=q_family_array,
    log_joint=log_joint,
    n_samples=5
)

# test loop
using Optimisers

z = randn(dim_z) * 0.2

opt = MyOptimisers.DecayedADAGrad()

opt = Optimisers.Adam()

opt = Optimisers.Descent(0.01)

state = Optimisers.setup(opt, z)

n_iter = 1000
z_trace = zeros(n_iter, length(z))
loss = []
for iter = 1:n_iter
    println(iter)

    train_loss, grads = Zygote.withgradient(z) do zp
        elbo(
            zp;
            q_family_array=q_family_array,
            log_joint=log_joint,
            n_samples=5
        )
    end

    push!(loss, train_loss)
    # z update
    Optimisers.update!(state, z, grads[1])
    # @. z += opt.eta * grads[1]
    z_trace[iter, :] = z
end

plot(z_trace)
plot(loss)

# Get VI distribution
q = get_variational_dist(z, q_family_array, range_z)
theta = rand_array(q; from_base_dist=false, reduce_to_vec=false)

log_joint(theta; y=y, X=X)

elbo(
    z;
    q_family_array=q_family_array,
    log_joint=log_joint,
    n_samples=10
)
