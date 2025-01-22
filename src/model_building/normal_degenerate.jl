using Distributions
using Random

"""
    Univariate Constant Distribution
"""
struct ConstantDistribution{T<:Real} <: ContinuousUnivariateDistribution
    m::T
    function ConstantDistribution(m::T) where {T<:Real}
        new{T}(m)
    end
end

#### Outer constructors
ConstantDistribution(m::Integer) = ConstantDistribution(float(m))

function (d::ConstantDistribution)(x::Real)
    return d.m
end

#### Statistics
Distributions.entropy(d::ConstantDistribution) = 0.

#### Sampling
Base.rand(rng::Random.AbstractRNG, d::ConstantDistribution{T}) where {T} = d.m


"""
    Multivariate Constant Distribution
"""

struct MultivariateConstantDistribution{T<:AbstractArray} <: ContinuousUnivariateDistribution
    m::T
    function MultivariateConstantDistribution(m::T) where {T<:AbstractArray}
        new{T}(m)
    end
end

function (d::MultivariateConstantDistribution)(x::MultivariateConstantDistribution)
    return d.m
end

#### Statistics
Distributions.entropy(d::MultivariateConstantDistribution) = 0.

#### Sampling
Base.rand(rng::Random.AbstractRNG, d::MultivariateConstantDistribution{T}) where {T} = d.m
