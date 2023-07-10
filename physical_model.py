#%%
import physical_model_simulation as pms
import time

if __name__ == '__main__':
    start_time = time.time()
    sim = pms.PhysicalModel(number_of_nodes=10, loc_set_max=3)
    sim.main()
    print("--- total running time  %s minutes ---" % (round((time.time() - start_time)/60,2)))

# %%
import ExhaustiveAlgorithm as ex_algthm
#ex_algthm.input('20230710053057')
loc, links, loc_links, nodes = ex_algthm.input('20230710181523')
ea = ex_algthm.ExhaustiveAlgorithm(nodes = nodes, loc = loc, links = links, loc_links = loc_links)
ea.main()
#print("--- %s seconds ---" % (time.time() - start_time))
# %%
import TwoTerminalConn as ttc
paths = ea.paths.copy()
two_ter_conn = ttc.TwoTerminal(links=links, loc=loc, loc_links=loc_links,paths= paths)
two_ter_conn.main()
# %%
loc_links.head()
# %%
loc
# %%
