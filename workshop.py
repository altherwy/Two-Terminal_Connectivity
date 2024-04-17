#%%
from ExhaustiveAlgorithm import input
import pandas as pd
import DisjointPaths as dis_p
from numpy import prod
# %%
file_name = '20240414170600'
loc, links, loc_links, nodes = input(file_name)

def _get_disjoint_paths(links):
        dis_paths = dis_p.DisjointPaths(links)
        dps = dis_paths.runMaxFlow()
        return dps

disjoint_paths = _get_disjoint_paths(links)
# %%
def generate_paths(dp)->list:
    nodes = dp
    num_nodes = len(nodes)
    num_locs = [len(loc[node]) for node in nodes]
    num_paths = prod(num_locs)
    paths = []
    for i in range(num_paths):
        path = []
        for j in range(num_nodes):
            path.append(i % num_locs[j])
            i //= num_locs[j]   
        paths.append(path)
    paths = pd.DataFrame(paths, columns=nodes)
    # set prob column to 1
    paths['prob'] = 1
    return paths


def multiply_probabilities(paths, dp):
    num_paths = len(paths)
    for i in range(num_paths):
        prob = 1
        path = paths.iloc[[i]]
        dp.remove('S')
        dp.remove('T')
        for node in dp:
            loc_index = path[node].values[0]
            prob *= loc[node][loc_index]

        paths.loc[i,'prob'] = prob
    return paths
def isConnected(node:str,neighbour:str,node_pos:int,neighbour_pos:int):

        connections = loc_links[(node,neighbour)]
        connection = connections[node_pos][neighbour_pos]
        if connection == 1:
            return True
        return False

def connected_paths(paths, dp):
    paths['Connected'] = 'Not Processed'
    for i in range(len(paths)):
        path = paths.iloc[[i]]
        if path['Connected'].values[0] == 'Not Processed':
            # change path status to True
            paths.loc[i,'Connected'] = True
            paths =  _is_path_connected(path,dp, paths)

    return paths    


def _is_path_connected(path,dp, paths):
    for i in range(len(dp)-1):
        node = dp[i]
        neighbour = dp[i+1]
        node_pos = path[node].values[0]
        neighbour_pos = path[neighbour].values[0]
        if not isConnected(node,neighbour,node_pos,neighbour_pos):
            paths = _flag_paths(paths, node, neighbour, node_pos, neighbour_pos, False)
    
            return paths
    return paths
    

def _flag_paths(paths, node, neighbour, node_pos, neighbour_pos, flag):
    paths.loc[(paths[node] == node_pos) & (paths[neighbour] == neighbour_pos), 'Connected'] = flag
    return paths
# %% 
all_paths = pd.DataFrame()
for dp in disjoint_paths:
    paths = generate_paths(dp)
    prod_paths = multiply_probabilities(paths,dp)
    processed_paths = connected_paths(prod_paths, dp)
    all_paths = pd.concat([all_paths, processed_paths])
    all_paths.to_csv('results/'+file_name + '.csv', index=False)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DONE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
print(loc_links[('S','21')][1][0])
print(loc_links[('21','30')][0][1])
print(loc_links[('30','T')][1][1])

# %% import all_paths from csv
file_name = 'results/20240414170600.csv'
df = pd.read_csv(file_name)

# %% return Connected true paths
def get_connectivity(paths):
        '''
        Computes the connectivity between two nodes (terminals) on a node disjoint path graph
        Args:
            None
        Returns:
            conn (float): the connectivity between two nodes (terminals) on a node disjoint path graph
        '''
        
        conn = 0
        for i in range(len(loc['S'])):
            s_prob = loc['S'][i]
            for j in range(len(loc['T'])):
                j_prob = loc['T'][j]
                temp = 1
                for dp in disjoint_paths:
                    connected_df = get_df_for_dp(paths,dp)
                    connected_df = connected_df[(connected_df['S'] == i) & (connected_df['T'] == j)]
                    sum_prob = connected_df['prob'].sum()
                    temp *= 1 - sum_prob
                

                conn += s_prob*j_prob*(1-temp)
                
        return conn
# %%
get_connectivity(df)
# %% get df for dp 
def get_df_for_dp(df,dp):
    dp.append('prob')
    df_connected = df[df['Connected'] == True]
    dp_df = df_connected[df_connected.columns.intersection(dp)]
    df_ = dp_df[dp_df.notnull().all(axis=1)]
    return df_



# %%
loc
# %%
df[df['Connected'] == False]
# %%
count = 0
for i in range(len(loc['S'])):
    for j in range(len(loc['T'])):
        count += df[(df['S'] == 0) & (df['T'] == 0) & (df['Connected'] == False)]['prob'].sum()

count
# %% df change all connected to False
df['Connected'] = False
# change first path to connected
df.loc[0,'Connected'] = True
df


# %%
df_results = pd.DataFrame(columns=['V','Loc_max','Conn_Level','Connectivity','Running Time'])
# %%
df_results
# %%
ser = pd.Series([1,2,3,4,5], index = df_results.columns)
# %%
ser
# %%
df_results = pd.concat([df_results,ser.to_frame().T],axis=0)
df_results
# %%
