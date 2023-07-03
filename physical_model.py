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
nodes, loc, loc_links, links = ea.dummy_data()
ex_algthm = ea.ExhaustiveAlgorithm(nodes=nodes,loc=loc,loc_links=loc_links, links=links)
ex_algthm.main()
df = ex_algthm.paths.copy()


# %% Two Terminal Connectivity approach
import DisjointPaths as dis_p
algorithm = 'MaxFlow'
dis_paths = dis_p.DisjointPaths(links)# type: ignore
dps = dis_paths.runMaxFlow() if algorithm == 'MaxFlow' else dis_paths.runSSSP() 
df_paths = df[df['Connected'] == True].copy()
df_paths['prob'] = [1] * len(df_paths)
#%%
ConnectedPathException = type('ConnectedPathException', (Exception,), {})
NotConnectedPathException = type('NotConnectedPathException', (Exception,), {})
two_terminal_data = {}
for dp in dps:
    dp_copy = dp.copy()
    dp_copy.append('Connected')
    dp_copy.append('prob')
    df_path = df_paths[dp_copy]
    df_path = df_path.drop_duplicates(subset=dp)
    df_path.reset_index(drop=True,inplace=True)
    for i in range(len(df_path)):
        node_pos = int(df_path.loc[i,'S'])
        df_path.loc[i,'prob'] *= ex_algthm.loc['S'][node_pos]
        try:
            two_terminal(0,i,df_path,dp)
        except ConnectedPathException as e:
            df_path.loc[i,'Connected'] = True
        except NotConnectedPathException as e:
            df_path.loc[i,'Connected'] = False
    two_terminal_data[tuple(dp)] = df_path

    
#%%
def two_terminal(node_id:int,path_index:int,df_path,dp:list):
    node = dp[node_id]
    neighbour = dp[node_id+1]
    node_pos = int(df_path.loc[i,node])
    neighbour_pos = int(df_path.loc[i,neighbour])
    if ex_algthm.isConnected(node,neighbour,node_pos,neighbour_pos):
        df_path.loc[path_index,'prob'] *= ex_algthm.loc[neighbour][neighbour_pos]
        if neighbour == 'T':
            raise ConnectedPathException('The path is connected')
        else:
            two_terminal(node_id+1,path_index,df_path,dp)
    else:
        raise NotConnectedPathException('The path is not connected')

# %%
conn = 0
for dp in dps:
    connected_df = two_terminal_data[tuple(dp)]
    connected_df = connected_df[connected_df['Connected'] == True]
    conn += connected_df['prob'].sum()
# %%
two_terminal_data[tuple(dps[0])]['prob'].sum()
# %%
