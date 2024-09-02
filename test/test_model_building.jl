# Test model building
using OrderedCollections

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))

# Normal model with parameters:
# beta0
# beta
# sigma_y

p = 10
dict = OrderedDict()

update_parameters_dict(dict; name="beta0", size=1)
update_parameters_dict(dict; name="beta", size=p)
update_parameters_dict(dict; name="sigma_y", size=1, bij=exp)
dict["priors"]
