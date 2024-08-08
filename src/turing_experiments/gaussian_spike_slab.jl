import Distributions
import Turing
using LinearAlgebra


EPS = Float32(1e-16)


struct GaussianSpikeSlab{
    M<:Real,
    S<:Real,
    G<:Real,
    B
} <: Distributions.ContinuousUnivariateDistribution
    mu::M
    sigma::S
    gamma::G
    slab::B
end

GaussianSpikeSlab(mu, sigma, gamma, slab) = GaussianSpikeSlab(mu, sigma, gamma, slab)
GaussianSpikeSlab(mu, sigma, gamma) = GaussianSpikeSlab(mu, sigma, gamma, Distributions.Normal(mu, sigma))
GaussianSpikeSlab(mu, sigma) = GaussianSpikeSlab(mu, sigma, 1f0, Distributions.Normal(mu, sigma))

Distributions.rand(rng::Distributions.AbstractRNG, d::GaussianSpikeSlab) = Distributions.rand(d.slab)

Distributions.logpdf(
    d::GaussianSpikeSlab,
    x::Real
) = log(d.gamma * exp(Distributions.logpdf(d.slab, x)) + (1f0 - d.gamma) + EPS)

Distributions.minimum(d::GaussianSpikeSlab) = -Inf32
Distributions.maximum(d::GaussianSpikeSlab) = +Inf32


# Test
# mu = [0., 1.]
# sigma = [1., 2.]
# gamma = [0.0001, 0.9999]
# x = GaussianSpikeSlab.(mu, sigma, gamma)

# sam = Distributions.rand.(x)
# Distributions.logpdf.(x, sam)
