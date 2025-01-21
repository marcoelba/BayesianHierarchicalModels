# Log Likelihood functions

module DistributionsLogPdf
using LogExpFunctions: log1pexp


function log_normal(
    x::AbstractArray,
    m::AbstractArray=zeros(eltype(x), size(x)),
    s::AbstractArray=ones(eltype(x), size(x));
    mu::AbstractArray=m,
    sigma::AbstractArray=s
    )
    -0.5f0 * log.(2*Float32(pi)) .- log.(sigma) .- 0.5f0 * ((x .- mu) ./ sigma).^2f0
end

log_normal(
    x::AbstractArray;
    mu::AbstractArray=zeros(eltype(x), size(x)),
    sigma::AbstractArray=ones(eltype(x), size(x))
) = log_normal(
    x::AbstractArray,
    mu::AbstractArray,
    sigma::AbstractArray
)


function log_normal(
    x::Real,
    mu::Real=0f0,
    sigma::Real=1f0
    )
    -0.5f0 * log(2*Float32(pi)) - log(sigma) - 0.5f0 * ((x - mu) / sigma)^2f0
end


function log_half_cauchy(
    x::AbstractArray,
    sigma::AbstractArray=(eltype(x), ones(size(x)))
    )
    log(2f0) .- log.(Float32(pi) .* sigma) .- log.(1f0 .+ (x ./ sigma).^2f0)
end

log_half_cauchy(
    x::AbstractArray;
    sigma::AbstractArray
) = log_half_cauchy(
    x::AbstractArray,
    sigma::AbstractArray
)


function log_half_cauchy(
    x::Real,
    s::Real=one(eltype(x));
    sigma::Real=s
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


end
