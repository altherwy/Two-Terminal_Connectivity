#%%
import physical_model_simulation as pms
number_of_nodes = 6
number_of_localities = 3
phys_model = pms.PhysicalModel(number_of_nodes = number_of_nodes, loc_set_max=number_of_localities)
loc, links,loc_links, nodes = phys_model.get_data()
print(loc)
print(links)
print(loc_links)
print(nodes)
# %%
import ExhaustiveAlgorithm as ex_algthm
ea = ex_algthm.ExhaustiveAlgorithm(nodes = nodes, loc = loc, links = links, loc_links = loc_links)
ea.main()

# %%
ea.paths[ea.paths['Connected'] == False]
# %%
