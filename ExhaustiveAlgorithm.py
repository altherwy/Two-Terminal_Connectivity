import pandas as pd
from typing import Dict,List
nodes = ['S','A','B','C','D','E','F','T']
loc:Dict = {'A':[.15,.25,.3,.3], 'B':[.4,.2,.4], 'C':[.2,.3,.4,.1], 'D':[.4,.3,.3],
                'E':[.5,.5], 'F':[.4,.6],
                'S':[.3,.5,.2], 'T':[.8,.2]}
loc_links = pd.DataFrame({('A','B'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                                ('B','C'): {0:[1,1,1,1],1:[1,1,1,1], 2:[1,1,1,1]},
                                ('C','D'): {0:[0,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                                ('E','F'): {0:[1,1],1:[1,1]},
                                ('S','A'): {0:[1,1,1,1], 1:[1,1,1,1], 2:[1,1,1,1]},
                                ('D','T'): {0:[1,1], 1:[0,1], 2:[1,1]},
                                ('S','E'): {0:[1,1],1:[0,1],2:[0,1]},
                                ('B','T'): {0:[1,0],1:[0,0],2:[0,0]},
                                ('F','T'): {0:[1,0],1:[1,1]}
                                })
columns:List = nodes
columns.append('Probability')

prob= pd.DataFrame({}, columns=columns)
conn_prob = pd.DataFrame({}, columns=columns)

def exhaustive_algorithm(node_ind:int, s_p:int, path:Dict, conn_path:Dict):
    node:str = nodes[node_ind]    
    node_loc:List = loc[node]


    for j in range(len(node_loc)):
        conn_path[node] = j
        p = node_loc[j]
        s_p *= p
        path:Dict = __check_connection(node,j,path)
        if node != 'T':
            node_ind += 1 # next node in nodes list
            node_ind= exhaustive_algorithm(node_ind,s_p,path, conn_path)
            s_p /= p
        else:
            
            indices = [v for k,v in path.items()]
            indices.append(s_p)
            prob.loc[len(prob.index)] = indices
  
            indices = [v for k,v in conn_path.items()]
            indices.append(s_p)
            conn_prob.loc[len(conn_prob.index)] = indices
            
            path.popitem() # remove last node
            conn_path.popitem()
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

tot_conn = 0
node_loc:List = loc['S']
for i in range(len(node_loc)):
    path:Dict = {'S':i}
    conn_path:Dict = {'S':i}
    exhaustive_algorithm(1,node_loc[i],path, conn_path)

tot_conn = round(sum(prob.loc[prob['T'] != -1]['Probability']),7) # the total connectivity of the graph
print(tot_conn)
'''
nodes = ['S','A','B','T']
loc:Dict = {'S':[.6,.4], 'A':[.7,.3], 'B':[.5,.5], 'T':[.8,.2]}

loc_links = pd.DataFrame({('S','A'): {0:[1,0], 1:[0,1]},
                          ('S','B'): {0:[0,0], 1:[1,1]},
                          ('A','B'): {0:[1,0], 1:[1,1]},
                          ('A','T'): {0:[1,0], 1:[1,0]},
                          ('B','T'): {0:[1,1], 1:[0,1]},
                          })
'''
