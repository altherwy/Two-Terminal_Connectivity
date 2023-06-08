#%%
import physical_model_simulation as pms
import pandas as pd
#%%
sim = pms.PhysicalModel(loc_set_max=5)
loc, links, loc_links = sim.main()
# %%
from jaal import Jaal
from jaal.datasets import load_got

edge_df,node_df = load_got()
#Jaal(edge_df,node_df).plot()
# %%
edge_df.head()
# %%
from_list = []
to_list = []
for key in links.keys():
    for to in links[key]:
        from_list.append(key)
        to_list.append(to)

df_node = pd.DataFrame({'id':[i for i in range(sim.number_of_nodes)]})
df_edge = pd.DataFrame({'from':from_list,'to':to_list})


# %%
Jaal(df_edge,df_node).plot()

# %%
from_list = []
to_list = []
from_loc_list = []
to_loc_list = []
status_list = []
for key in loc_links.keys():
    key_list = list(key)
    node_from = key[0]
    node_to = key[1]
    dict_ = loc_links[key]
    for loc in dict_.keys():
        list_ = dict_[loc]
        counter = 0
        for item in list_:
            from_list.append(node_from)
            to_list.append(node_to)
            from_loc_list.append(loc)
            to_loc_list.append(counter)
            status_list.append(item)
            counter += 1

df_loc_links = pd.DataFrame({'from':from_list,'to':to_list,'from_loc':from_loc_list,'to_loc':to_loc_list,'status':status_list})           



# %%
sim.node_positions_filtered
# %%
df_loc_links[df_loc_links['status']==1]
# %%
