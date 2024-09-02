# test

abs_project_path = normpath(joinpath(@__FILE__, "..", ".."))

include(joinpath(abs_project_path, "src", "model_building", "utils.jl"))

dict = OrderedDict()

update_parameters_dict(dict; name="ciao", size=10, bij=identity)
update_parameters_dict(dict; name="blabla", size=100, bij=identity)
