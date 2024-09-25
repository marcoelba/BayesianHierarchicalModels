# gradient descent experiments
using DiffResults
using StatsFuns
using ADTypes
using Zygote
using AdvancedVI
using StatsPlots
using Flux

abs_project_path = normpath(joinpath(@__FILE__, "..", "..", ".."))
include(joinpath(abs_project_path, "src", "utils", "decayed_ada_grad.jl"))

function polynomial_decay(t::Int64; a::Float32=1f0, b::Float32=0.01f0, gamma::Float32=0.75f0)
    a * (b + t)^(-gamma)
end

function cyclical_polynomial_decay(n_iter::Int64, n_cycles::Int64=2)
    steps_per_cycle = Int(n_iter / n_cycles)
    lr_schedule = []
    for cycle = 1:n_cycles
        push!(lr_schedule, polynomial_decay.(range(1, steps_per_cycle))...)
    end
    return lr_schedule
end


# generate some simple optimisation problem

# convex
n = 100
x1 = collect(range(-2, 2, length=n))
x2 = collect(range(-2, 2, length=n))
f(x1, x2) = x1.^2 .+ x2.^2
y = @. f(x1, x2)
yf = @. f(x1', x2)
contour(x1, x2, yf, fill=false, color=:turbo, levels=30)

# NON convex
n = 100
x_range = 6
x1 = collect(range(-x_range, x_range, length=n))
x2 = collect(range(-x_range, x_range, length=n))
f(x1, x2) = −sin(x1) + cos(2*x1) − sin(x2) + cos(2*x2)
y = @. f(x1, x2)
yf = @. f(x1', x2)
contour(x1, x2, yf, fill=true, color=:turbo, levels=30)
scatter(x1, y)

# the loss function is the function f

# choose optimiser
optimizer = MyDecayedADAGrad()
optimizer = Flux.Descent(0.1)
optimizer = Flux.RMSProp(0.01)

# initial params (only 2 here)
theta = [-1.5, -1.5]

loss_trace = []
n_iter = 400
theta_trace = zeros(n_iter+1, 2)
theta_trace[1, :] = theta

use_noisy_grads = true
lr_schedule = cyclical_polynomial_decay(n_iter, 4)
n2 = Int(n_iter / 2)

for iter = 1:n_iter
    # calculate gradient
    loss, grads = Zygote.withgradient(theta) do x
        f(x...)
    end

    # update params (and optimizer)
    # Flux.update!(optimizer, theta, grads[1])
    if typeof(optimizer) == MyDecayedADAGrad
        diff_grads = apply!(optimizer, theta, grads[1])
    else
        diff_grads = Flux.Optimise.apply!(optimizer, theta, grads[1])
    end

    # update params
    if use_noisy_grads
        grad_noise = randn(size(theta)) .* lr_schedule[iter]
    else
        grad_noise = 0f0
    end
    
    @. theta = theta - diff_grads + grad_noise

    push!(loss_trace, loss)
    theta_trace[iter+1, :] = copy(theta)
end

plt = contour(x1, x2, yf, fill=false, color=:turbo, levels=30)
scatter!(theta_trace[1:n2, 1], theta_trace[1:n2, 2], label=false, markersize=3, color=2)
scatter!(theta_trace[n2+1:n_iter, 1], theta_trace[n2+1:n_iter, 2], label=false, markersize=3, color=3)
display(plt)



