#%%
import time
import physical_model_simulation as pms
start_time = time.time()
number_of_nodes = 50
number_of_localities = 3
phys_model = pms.PhysicalModel(number_of_nodes = number_of_nodes, loc_set_max=number_of_localities)
loc, links,loc_links, nodes = phys_model.get_data()
print(loc)
print(links)
print(loc_links)
print(nodes)
print("---total running time  %s minutes ---" % ((time.time() - start_time)/60))
# %%
import time
start_time = time.time()
import ExhaustiveAlgorithm as ex_algthm
ea = ex_algthm.ExhaustiveAlgorithm(nodes = nodes, loc = loc, links = links, loc_links = loc_links)
ea.main()
print("--- %s seconds ---" % (time.time() - start_time))
# %%
import TwoTerminalConn as ttc
paths = ea.paths.copy()
two_ter_conn = ttc.TwoTerminal(links=links, loc=loc, loc_links=loc_links,paths= paths)
two_ter_conn.main()
# %%
