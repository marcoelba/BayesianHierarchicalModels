# Vectorised Bijectors
module VectorizedBijectors

using StatsFuns


function StatsFuns.softplus(x::AbstractArray)
    StatsFuns.softplus.(x)
end

function simplex(x::AbstractArray)
    StatsFuns.softmax(vcat(x, 0f0))
end


end
