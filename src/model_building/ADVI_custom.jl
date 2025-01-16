# ADVI with custom variational distributions

module ADVI

using Zygote
using Bijectors
using Distributions
using LogExpFunctions


function LogExpFunctions.log1pexp(x::AbstractArray)
    LogExpFunctions.log1pexp.(x)
end


function log_joint(theta; y, X)
    pred = X * theta
    -sum((y .- pred).^2)
end
X = randn(100, 3)
y = X * [1, 2, 1] .+ randn(100)*0.5

function vi_theta1(z::AbstractArray)
    Bijectors.transformed(
        Distributions.product_distribution(Distributions.Normal.(z[1:2], LogExpFunctions.log1pexp.(z[3:4]))),
        identity
    )
end

dd = vi_theta1([0, 1, 1, 0.5])
z_1_dim = 4
z_1_pos = 1:4
rand(dd)
rand(dd.dist)
dd.transform

function vi_theta2(z::AbstractArray)
    Bijectors.transformed(
        Distributions.Normal(z[1], LogExpFunctions.log1pexp.(z[2])),
        LogExpFunctions.log1pexp
    )
end

dd2 = vi_theta2([0, -1])
z_2_dim = 2
z_2_pos = range(1, z_2_dim) .+ z_1_dim

rand(dd2)
rand(dd2.dist)
dd2.transform

range_z = [z_1_pos, z_2_pos]
q_family_array = [vi_theta1, vi_theta2]
z = randn(6)


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

    return reduce(vcat, x_t), sum(abs_jacobian)
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
    theta, abs_jacobian = rand_with_logjacobian(q_dist_array)

    # evaluate the log-joint
    res = log_joint(theta; y=y, X=X) + sum(abs_jacobian)

    # add entropy
    for d in q_dist_array
        res += Distributions.entropy(d)
    end

    return -res
end

z = randn(6)
elbo(
    z;
    q_family_array=q_family_array,
    log_joint=log_joint,
    n_samples=1
)


for iter = 1:100
    println(iter)

    train_loss, grads = Zygote.withgradient(z) do zp
        elbo(
            zp;
            q_family_array=q_family_array,
            log_joint=log_joint,
            n_samples=1
        )
    end

    # z update
    z = z .- 0.01 * grads[1]
end

# Get VI distribution
q = get_variational_dist(z, q_family_array, range_z)
theta = rand_array(q; from_base_dist=false, reduce_to_vec=true)


y, back = Zygote.pullback(f, θ)
dy = first(back(1.0))
DiffResults.value!(out, y)
DiffResults.gradient!(out, dy)

return out
