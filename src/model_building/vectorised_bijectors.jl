# Vectorised Bijectors
module VectorizedBijectors

using StatsFuns
using LogExpFunctions

function simplex(x::AbstractArray)
    StatsFuns.softmax(vcat(x, 0f0))
end

end
