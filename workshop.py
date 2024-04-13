#%%
from ExhaustiveAlgorithm import input
import pandas as pd
# %%
file_name = '20240412120240'
loc, links, loc_links, nodes = input(file_name)
disjoint_paths = [['S', '1', 'T']]
# %%
'''
for dp in disjoint_paths:
    paths = generate_paths(dp)
    multiply_pths = multiply_probabilities(paths,dp)
    
'''
def print_(node, neighbour, node_pos, neighbour_pos):
    print(node, neighbour, node_pos, neighbour_pos)
    print(isConnected(node,neighbour,node_pos,neighbour_pos))
# %%
from itertools import product
from numpy import prod
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
#%%

pths = generate_paths(disjoint_paths[0])
multiply_pths = multiply_probabilities(pths,disjoint_paths[0])
# call connected_paths
# %%
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
            #print_(node, neighbour, node_pos, neighbour_pos)
            return paths
    return paths
    #return _flag_paths(paths, node, neighbour, node_pos, neighbour_pos, True)

def _flag_paths(paths, node, neighbour, node_pos, neighbour_pos, flag):
    paths.loc[(paths[node] == node_pos) & (paths[neighbour] == neighbour_pos), 'Connected'] = flag
    return paths
# %% 

pt = connected_paths(multiply_pths, disjoint_paths[0])
# %%
# %%
print(loc_links[('S','1')][0][0])
print(loc_links[('1','T')][0][0])
# %%
pt
# %%
