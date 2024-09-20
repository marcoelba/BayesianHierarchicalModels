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

end
