#%%
#from TwoTerminalConn import *
import pandas as pd
from typing import Dict,List

nodes = ['S','A','B','T']
loc:Dict = {'S':[.6,.4], 'A':[.7,.3], 'B':[.5,.5], 'T':[.8,.2]}

loc_links = pd.DataFrame({('S','A'): {0:[1,0], 1:[0,1]},
                          ('S','B'): {0:[0,0], 1:[1,1]},
                          ('A','B'): {0:[1,0], 1:[1,1]},
                          ('A','T'): {0:[1,0], 1:[1,0]},
                          ('B','T'): {0:[1,1], 1:[0,1]},
                          })
prob= pd.DataFrame({}, columns=['S','A','B','T','Prob'])

def rec(node_ind:int, s_p:int, loc_ind:int, path:Dict):
    node:str = nodes[node_ind]
    prev_node:str = nodes[node_ind-1]
    path[prev_node] = loc_ind

    node_loc:List = loc[node]


    for j in range(len(node_loc)):
        p = node_loc[j]
        s_p *= p
        path:Dict = __check_connection(node,j,path)
        print(path)
        if node != 'T':
            node_ind += 1 # next node in nodes list
            node_ind= rec(node_ind,s_p,j,path)
            s_p /= p
        else:
            
            indices = [v for k,v in path.items()]
            indices.append(s_p)
            prob.loc[len(prob.index)] = indices
            path.popitem() # remove last node
            s_p /= p # remove the probability of the last node

    
    node_ind -= 1
    return node_ind

def __check_connection(node:str, index:int, path:Dict)->Dict:
    for k,v in path.items(): # {S:0,A:0}
        try:
            links:Dict = loc_links[(k,node)]
        except KeyError as err:
            continue

        if v == -1:
            continue
        e = links[v][index]
        if e == 1:
            path[node] = index
            return path
    
    path[node] = -1
    return path

#%%
node_loc:List = loc['S']
for i in range(len(node_loc)):
    path:Dict = {'S':i}
    rec(1,node_loc[i],i,path)
# %%
print(prob)
# %%
indices = prob.loc[0]
print(indices['S'])
# %%
