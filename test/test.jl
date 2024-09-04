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