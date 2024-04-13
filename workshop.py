#%%
from ExhaustiveAlgorithm import input
import pandas as pd
import DisjointPaths as dis_p
from numpy import prod
# %%
file_name = '20240414003237'
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

# %%
print(loc_links[('S','36')][2][1])
print(loc_links[('36','38')][1][2])
print(loc_links[('38','T')][2][2])

# %% import all_paths from csv
file_name = 'results/20240414003237.csv'
df = pd.read_csv(file_name)
df

# %% return Connected true paths
df_paths = df[df['Connected'] == True]
df_paths

# %% 37 not None

df2 = df[(df['36'].notnull()) & (df['38'].notnull()) & (df['Connected'] == True)]
# %%
print(loc['S'])
print(loc['36'])
print(loc['38'])
print(loc['T'])
# %%
3*3*3*3
# %% show only S, T, 36, 38
df2[['S','36','38','T','prob']]

# %%
