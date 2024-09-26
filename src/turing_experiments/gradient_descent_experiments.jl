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
x_range = 3
fx(xmin, xmax) = collect(range(xmin, xmax, length=n))
x1 = fx(-x_range, x_range)
x2 = fx(-x_range, x_range)
f(x1, x2) = −sin(x1) + cos(5*x1) − sin(x2) + cos(2*x2)
y = @. f(x1, x2)
yf = @. f(x1', x2)
contour(x1, x2, yf, fill=true, color=:turbo, levels=30)
scatter(x1, y)

# the loss function is the function f

# choose optimiser
optimizer = MyDecayedADAGrad()
optimizer = Flux.Descent()
optimizer = Flux.RMSProp(0.01)

# initial params (only 2 here)
theta = [-2., -2.]
theta_init = deepcopy(theta)

n_iter = 1000
n_iter_post = 100
n_iter_tot = n_iter + n_iter_post

theta_trace = zeros(n_iter_tot, 2)
loss_trace = []
min_loss = Inf64
min_loss_theta = theta_init

use_noisy_grads = true
n_cycles = 4
n_per_cycle = Int(n_iter / n_cycles)
sections = []
for cycle = 1:n_cycles
    push!(sections, collect(range(1, n_per_cycle)) .+ n_per_cycle * (cycle-1))
end
lr_schedule = cyclical_polynomial_decay(n_iter, n_cycles)
plot(lr_schedule)
final_section = collect(range(n_iter+1, n_iter_tot))


for iter = 1:n_iter_tot
    
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
    if iter <= n_iter
        if use_noisy_grads
            grad_noise = randn(size(theta)) .* lr_schedule[iter]
        else
            grad_noise = 0f0
        end
    else
        grad_noise = 0f0
    end

    @. theta = theta - diff_grads + grad_noise

    # store the minimum
    if loss < min_loss
        min_loss = loss
        min_loss_theta = copy(theta)
    end

    # store trace
    push!(loss_trace, loss)
    theta_trace[iter, :] = copy(theta)
end

x1 = fx(minimum(theta_trace[:, 1]) .- 0.5, maximum(theta_trace[:, 1]) .+ 0.5)
x2 = fx(minimum(theta_trace[:, 2]) .- 0.5, maximum(theta_trace[:, 2]) .+ 0.5)
yf = @. f(x1', x2)
plt = contour(
    x1, x2, yf, fill=false, color=:turbo, levels=30
)
scatter!([theta_init[1]], [theta_init[2]], label="init")
for cycle = 1:n_cycles
    scatter!(theta_trace[sections[cycle], 1], theta_trace[sections[cycle], 2],
    label=cycle, markersize=3, color=cycle)
end
scatter!(theta_trace[final_section, 1], theta_trace[final_section, 2],
    label="final", markersize=3, color="black")
display(plt)

plot(loss_trace)
