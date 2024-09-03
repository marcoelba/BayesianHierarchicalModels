# Log Likelihood functions

module DistributionsLogPdf

function log_normal(
    x::AbstractArray{Float32},
    mu::AbstractArray{Float32}=Float32.(zeros(size(x))),
    sigma::AbstractArray{Float32}=Float32.(ones(size(x)))
    )
    -0.5f0 * log.(2*Float32(pi)) .- log.(sigma) .- 0.5f0 * ((x .- mu) ./ sigma).^2f0
end

function log_normal(
    x::Float32;
    sigma::Float32,
    mu::Float32=0f0
    )
    -0.5f0 * log(2*Float32(pi)) - log(sigma) - 0.5f0 * ((x - mu) / sigma)^2f0
end

end
