# test

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))
include(joinpath(abs_project_path, "src", "model_building", "model_prediction_functions.jl"))
include(joinpath(abs_project_path, "src", "model_building", "distributions_logpdf.jl"))

dict = OrderedDict()

update_parameters_dict(dict; name="ciao", size=10, bij=identity)
update_parameters_dict(dict; name="blabla", size=100, bij=identity)


mu = [-1f0, 1f0]
s = [2f0, 0.5f0]
p = (mu, s)
p = (mu,)

DistributionsLogPdf.log_normal([1f0, 2f0])
DistributionsLogPdf.log_normal([1f0, 2f0], mu, s)
DistributionsLogPdf.log_normal([1f0, 2f0], mu=mu, sigma=s)
DistributionsLogPdf.log_normal([1f0, 2f0], p...)

g = x -> DistributionsLogPdf.log_normal(x)
g([1f0, 2f0])

g = (x, sigma) -> DistributionsLogPdf.log_normal(x, sigma=sigma)
g([1f0, 2f0], s)
DistributionsLogPdf.log_normal([1f0, 2f0], sigma=s)

# logit
using Distributions
using LogExpFunctions: log1pexp

dd = Distributions.Bernoulli(0.2)
Distributions.logpdf(dd, 1)

dd_logit = Distributions.BernoulliLogit(log(0.2/0.8))
Distributions.logpdf.(dd_logit, [0, 1])

1 * log(0.2/0.8) + log(1 - 0.2)
0 * log(0.2/0.8) + log(1 - 0.2)

-log1pexp(-log(0.2/0.8))
-log1pexp(log(0.2/0.8))

function b_logpdf(logp, x)
    x == 0 ? -log1pexp(logp) : (x == 1 ? -log1pexp(-logp) : oftype(float(logp), -Inf))
end
b_logpdf(log(0.2/0.8), 1)

function log_bernoulli_from_logit(x::AbstractArray, logitp::AbstractArray)
    @. - (1 - x) * log1pexp(logitp) - x * log1pexp(-logitp)
end

function log_bernoulli_from_logit(x, logitp)
    x == 0 ? -log1pexp(logitp) : (x == 1 ? -log1pexp(-logitp) : oftype(float(logitp), -Inf))
end

logitp = [log(0.2/0.8), log(0.2/0.8)]
x = [0, 1]
log_bernoulli_from_logit.(x, logitp)
log_bernoulli_from_logit(x, logitp)


# Bayesian
n = 100
x = rand(Bernoulli(0.3), n)
plt = density(rand(Beta(1. + sum(x), 1. + (n - sum(x))), 1000), label="Posterior n=100", trim=true, color="blue")
vline!([sum(x)/n], label="MLE n=100", color="blue", linestyle=3, linewidth=2)
n = 10
x = rand(Bernoulli(0.3), n)
density!(rand(Beta(1. + sum(x), 1. + (n - sum(x))), 1000), label="Posterior n=10", trim=true, color="green")
vline!([sum(x)/n], label="MLE n=10", color="green", linestyle=3, linewidth=2)
hline!([1.], label="Prior", color="red")
savefig(plt, "/home/marco_ocbe/post.pdf")


#
using GLM
using Distributions
using DataFrames
using StatsPlots

x = Uniform(0, 1)
pdf(x, 0.4)
cdf(x, 0.4)

n = 1000
m = 100
beta_true = vcat(ones(50), zeros(50))

X_dist = Normal(0, 1)
X = rand(X_dist, (n, m))

f(X) = X * beta_true .+ randn(n)*0.5
y = f(X)

res = GLM.lm(X, y)
p_values = DataFrame(GLM.coeftable(res))[:, "Pr(>|t|)"]
histogram(p_values[51:m], bins=5, normalize=true)

#
X = Normal(-2, 0.2)
x = rand(X, 2000)
density(x)

cdf(X, -1.8)
sum(x .< -1.8) / length(x)

abs_x = abs.(x)
mean(abs_x)
var(abs_x)

density!(abs_x)
sum(abs_x .> 1.8) / length(x)

logit_x = logistic.(abs_x)
density!(logit_x)
sum(logit_x .> logistic(1.8)) / length(x)
