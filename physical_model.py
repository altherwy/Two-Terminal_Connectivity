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
#%%
loc_links.head()
# %%
import TwoTerminalConn as ttc
_,_,loc_links = ttc.dummy_data()
loc_links
# %%
ttc.TwoTerminal(links=links, loc=loc, loc_links=loc_links).main()
# %%
import ExhaustiveAlgorithm as ea
ea.ExhaustiveAlgorithm(nodes=nodes, loc=loc, loc_links=loc_links).main()
# %%
nodes
# %%
loc['S']
# %%
4*3*4*3*2*2*3*2
# %%
