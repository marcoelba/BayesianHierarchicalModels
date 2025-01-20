# Bijectors extension
using Bijectors
using LogExpFunctions


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

function LogExpFunctions.log1pexp(x::AbstractArray)
    LogExpFunctions.log1pexp.(x)
end
