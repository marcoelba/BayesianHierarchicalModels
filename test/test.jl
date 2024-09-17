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
