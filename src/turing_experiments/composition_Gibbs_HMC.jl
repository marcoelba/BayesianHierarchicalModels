# Composition of Gibbs and HMC
using Turing
using Distributions
using StatsPlots
using Random
using LinearAlgebra
using DataFrames
using FillArrays


# Mixture Model

Random.seed!(3)
# Define Gaussian mixture model.
w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ in μ], w)

# We draw the data points.
N = 60
x = rand(mixturemodel, N)


# Build model
@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    # w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model(x)

# Inference
sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
nsamples = 150
nchains = 4
burn = 10
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains, discard_initial = burn)

plot(chains[["μ[1]", "μ[2]"]]; legend=true)


# Marginalised Miture Model
# Using MixtureModel from the Distribution package

@model function gmm_marginalized(x)
    K = 2
    D, _ = size(x)
    μ ~ Bijectors.ordered(MvNormal(Zeros(K), I))
    w ~ Dirichlet(K, 1.0)
    x ~ MixtureModel([MvNormal(Fill(μₖ, D), I) for μₖ in μ], w)
end
model = gmm_marginalized(x)

sampler = NUTS()
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains; discard_initial = burn)

plot(chains[["μ[1]", "μ[2]"]], legend=true)

# recovering the cluster assignments
function sample_class(xi, dists, w)
    lvec = [(logpdf(d, xi) + log(w[i])) for (i, d) in enumerate(dists)]
    rand(Categorical(Turing.softmax(lvec)))
end


@model function gmm_recover(x)
    K = 2
    D, N =  size(x)
    μ ~ Bijectors.ordered(MvNormal(Zeros(K), I))
    w ~ Dirichlet(K, 1.0)
    dists = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]
    x ~ MixtureModel(dists, w)
    # Return assignment draws for each datapoint.
    return [sample_class(x[:, i], dists, w) for i in 1:N]
end

model = gmm_recover(x)
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains, discard_initial = burn)

assignments = mean(generated_quantities(gmm_recover(x), chains))

scatter(
    x[1, :],
    x[2, :];
    legend=false,
    title="Assignments on Synthetic Dataset - Recovered",
    zcolor=assignments,
)

