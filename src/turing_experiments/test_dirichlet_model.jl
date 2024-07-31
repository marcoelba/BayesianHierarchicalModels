
# Define Gaussian mixture model.
w = [0.25, 0.5, 0.25]
μ = [-3.5, 0.5, 3.5]
mixturemodel = MixtureModel([Normal(μₖ, 0.5) for μₖ in μ], w)

# We draw the data points.
N = 100
x = rand(mixturemodel, N)
density(x)

@model function gmm_marginalized(x)
    K = 3
    μ ~ Bijectors.ordered(MvNormal(zeros(K), I))
    w ~ Dirichlet(K, 1.0)
    x ~ MixtureModel([Normal(μₖ, 1.) for μₖ in μ], w)
end

model = gmm_marginalized(x)

sampler = NUTS()
chains = sample(model, sampler, 2000)

plot(chains, legend=true)
