# Variational Distributions

module VariationalDistributions
using Distributions
using StatsFuns

function meanfield(z::AbstractArray{Float32}; tot_params::Int64)
    Distributions.MultivariateNormal(
        z[1:tot_params],
        StatsFuns.softplus.(z[(tot_params + 1):(tot_params * 2)])
    )
end

end
