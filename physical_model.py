#%%
import physical_model_simulation as pms
number_of_nodes = 8
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
ea.paths[ea.paths['Connected'] == True]
# %%
import TwoTerminalConn as ttc
paths = ea.paths.copy()
two_ter_conn = ttc.TwoTerminal(links=links, loc=loc, loc_links=loc_links,paths= paths)
conn = two_ter_conn.main()
print('Two Terminal Conn: ', conn)
# %%
two_ter_conn.dps
# %%
phys_model.node_positions_filtered.head()
# %%
# sum pos_prob based on node_id
phys_model.node_positions_filtered.groupby('node_id')['pos_prob'].sum()


# %%
loc_links
# %%
