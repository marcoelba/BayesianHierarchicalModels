# Log Likelihood functions

module DistributionsLogPdf
using LogExpFunctions: log1pexp

function log_normal(
    x::AbstractArray{Float32},
    m::AbstractArray{Float32}=Float32.(zeros(size(x))),
    s::AbstractArray{Float32}=Float32.(ones(size(x)));
    mu::AbstractArray{Float32}=m,
    sigma::AbstractArray{Float32}=s
    )
    -0.5f0 * log.(2*Float32(pi)) .- log.(sigma) .- 0.5f0 * ((x .- mu) ./ sigma).^2f0
end

function log_normal(
    x::Float32,
    mu::Float32=0f0,
    sigma::Float32=1f0
    )
    -0.5f0 * log(2*Float32(pi)) - log(sigma) - 0.5f0 * ((x - mu) / sigma)^2f0
end


function log_half_cauchy(
    x::AbstractArray{Float32},
    s::AbstractArray{Float32}=Float32.(ones(size(x)));
    sigma::AbstractArray{Float32}=s
    )
    log(2f0) .- log.(Float32(pi) .* sigma) .- log.(1f0 .+ (x ./ sigma).^2f0)
end

function log_half_cauchy(
    x::Float32,
    s::Float32=1f0;
    sigma::Float32=s
    )
    log(2f0) .- log.(Float32(pi) .* sigma) .- log.(1f0 .+ (x ./ sigma).^2f0)
end

function log_bernoulli_from_logit(x::AbstractArray, logitp::AbstractArray)
    @. - (1 - x) * log1pexp(logitp) - x * log1pexp(-logitp)
end

function log_bernoulli_from_logit(x::Real, logitp::Real)
    x == 0 ? -log1pexp(logitp) : (x == 1 ? -log1pexp(-logitp) : oftype(float(logitp), -Inf))
end

"""
    Bernoulli likelihood
"""
eps_prob = 1e-6

function log_bernoulli(x::AbstractArray, prob::AbstractArray)
    @. x * log(prob + eps_prob) + (1. - x) * log(1. - prob + eps_prob)
end

function log_bernoulli(x::Real, prob::Real)
    x * log(prob + eps_prob) + (1. - x) * log(1. - prob + eps_prob)
end


"""
Log-pdf of a mixture of Normal distributions.
    x::Float32
    w::AbstractArray{<:Float32}
    mu::AbstractArray{<:Float32}
    sd::AbstractArray{<:Float32}
"""
function log_normal_mixture(
    x::Float32,
    w::AbstractArray{<:Float32}=Float32.(ones(1, 2)*0.5),
    m::AbstractArray{<:Float32}=Float32.(zeros(1, 2)),
    s::AbstractArray{<:Float32}=Float32.(ones(1, 2));
    weights::AbstractArray{<:Float32}=w,
    mu::AbstractArray{Float32}=m,
    sigma::AbstractArray{Float32}=s
    )

    xstd = -0.5f0 .* ((x .- mu) ./ sigma).^2f0
    wstd = weights ./ (sqrt(2f0 .* Float32(pi)) .* sigma)
    offset = maximum(xstd .* wstd, dims=2)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=2)
    log.(s) .+ offset
end

"""
Log-pdf of a mixture of Normal distributions.
    x::AbstractArray{<:Float32}
    w::AbstractArray{<:Float32}
    mu::AbstractArray{<:Float32}
    sd::AbstractArray{<:Float32}
"""
function log_normal_mixture(
    x::AbstractArray{<:Float32},
    w::AbstractArray{<:Float32},
    mu::AbstractArray{<:Float32},
    sd::AbstractArray{<:Float32}
    )
    f(x_array) = log_normal_mixture(x_array, w, mu, sd)
    # log_normal_mixture.(x, Ref(w), Ref(mu), Ref(sd))
    f.(x)
end

"""
Log-likelihood of a mixture of Normal distributions (TEMP).
    x::AbstractArray{<:Float32} (n x 1)
    w::AbstractArray{<:Float32} (1 x n_clusters)
    mu::AbstractArray{<:Float32} (n x n_clusters)
    sd::AbstractArray{<:Float32} (n x n_clusters)
"""
function loglik_normal_mixture(
    x::AbstractArray{<:Float32},
    weights::AbstractArray{<:Float32},
    mu::AbstractArray{<:Float32},
    sigma::AbstractArray{<:Float32}
    )

    xstd = -0.5f0 .* ((x .- mu) ./ sigma).^2f0
    wstd = weights ./ (sqrt(2f0 .* Float32(pi)) .* sigma)
    offset = maximum(xstd .* wstd, dims=2)
    xe = exp.(xstd .- offset)
    s = sum(xe .* wstd, dims=2)
    log.(s) .+ offset
end

end
