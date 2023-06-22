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
prob_list = []

for key in loc_links.keys():
    key_list = list(key)
    node_from = key[0]
    node_to = key[1]
    dict_ = loc_links[key]
    for locality in dict_.keys():
        list_ = dict_[locality]
        counter = 0
        for item in list_:
            from_list.append(node_from)
            to_list.append(node_to)
            from_loc_list.append(locality)
            to_loc_list.append(counter)
            status_list.append(item)
            if item == 1: # if link is active
                prob = loc[node_from][locality]*loc[node_to][counter]
            else:
                prob = 0 # if link is inactive
            prob_list.append(prob)
            counter += 1

df_loc_links = pd.DataFrame({'from':from_list,'to':to_list,'from_loc':from_loc_list,'to_loc':to_loc_list,'prob':prob_list,'status':status_list})           

# %%
df_ = pd.DataFrame()
df_['from'] = df_loc_links['from'].astype(int)
# change to string starting from A, B, ...
df_['from'] = df_['from'].apply(lambda x: chr(x+65))
df_['to'] = df_loc_links['from_loc']
# add from char before number. For example A0, A1, ...
df_['to'] = df_['from'] + df_['to'].astype(str)
df_ = df_.drop_duplicates()
# restart the index from 0
df_ = df_.reset_index(drop=True)
df_
# %%
df_node = pd.DataFrame({'id':[i for i in range(sim.number_of_nodes-1)]})
df_node['id'] = df_node['id'].apply(lambda x: chr(x+65))
Jaal(df_).plot()
# %%
links

# %%
loc
# %%
loc_links

# %%
coordinate_1 = (890, -202, -255)
coordinate_2 = (-19, -296, -257)
x1, y1, z1 = coordinate_1
x2, y2, z2 = coordinate_2
distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
print(distance)
# %%
import physical_model_simulation as pms
phys_model = pms.PhysicalModel(number_of_nodes=10, loc_set_max=5)
loc, links, loc_links  = phys_model.get_data()
print(loc)
print(links)
print(loc_links)
import TwoTerminalConn as ttc
ttc.TwoTerminal(links=links, loc=loc, loc_links=loc_links).main()
# %%

