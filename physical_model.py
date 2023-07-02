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
ea.ExhaustiveAlgorithm(nodes=nodes, loc=loc, loc_links=loc_links).main()
# %%
import pandas as pd
nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'T']
#nodes = ['S', '1', '2', '3', '4', 'T']
#nodes = ['S','A','B','T']


loc: dict = {'A': [.15, .25, .3, .3], 'B': [.4, .2, .4], 'C': [.2, .3, .4, .1], 'D': [.4, .3, .3],
                'E': [.5, .5], 'F': [.4, .6],
                'S': [.3, .5, .2], 'T': [.8, .2]}
'''
loc: dict = {'S': [0.33, 0.66], 
        '1': [0.33], 
        '2': [0.33, 0.33], 
        '3': [0.33, 0.33], 
        '4': [0.33, 0.33], 
        'T': [0.33, 0.66]}

loc: dict = {
    'S':[0.2,0.3],
    'A':[0.5,0.5],
    'B':[0.4,0.6],
    'T':[0.3]
}

'''
loc_links = pd.DataFrame({('A', 'B'): {0: [1, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [0, 0, 0]},
                            ('B', 'C'): {0: [0, 0, 0, 1], 1: [0, 0, 0, 0], 2: [0, 0, 0, 1]},
                            ('C', 'D'): {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 0], 3: [0, 0, 0]},
                            ('E', 'F'): {0: [1, 1], 1: [0, 0]},
                            ('S', 'A'): {0: [0, 1, 0, 0], 1: [0, 0, 1, 0], 2: [0, 1, 0, 0]},
                            ('D', 'T'): {0: [0, 0], 1: [0, 1], 2: [0, 0]},
                            ('S', 'E'): {0: [1, 1], 1: [0, 1], 2: [0, 0]},
                            ('B', 'T'): {0: [1, 0], 1: [0, 0], 2: [0, 0]},
                            ('F', 'T'): {0: [1, 0], 1: [0, 0]}
                            })
'''
loc_links = pd.DataFrame({
    ('S', '2'): {0: [1,1], 1: [1,1]},
    ('S', '3'): {0: [1,1], 1: [1,1]},
    ('S', '4'): {0: [1,1], 1: [1,1]},
    ('1', '2'): {0: [1,1]},('1', '3'): {0: [1,1]},
    ('1', '4'): {0: [1,1]},('1', 'T'): {0: [1,1]},
    ('2', '3'): {0: [1,1], 1: [1,1]},
    ('2', '4'): {0: [1,1], 1: [1,1]},
    ('2', 'T'): {0: [1,1], 1: [1,1]},
    ('3', '4'): {0: [1,1], 1: [1,1]},
    ('3', 'T'): {0: [1,1], 1: [1,1]},
    ('4', 'T'): {0: [1,1], 1: [1,1]},
})

loc_links = pd.DataFrame({
    ('S', 'A'): {0: [1,1], 1: [0,0]},
    ('S', 'B'): {0: [0,0], 1: [1,1]},
    ('A', 'T'): {0: [0], 1: [1]},
    ('B', 'T'): {0: [1], 1: [0]}
    })
'''
links = {'S': ['A', 'B'], 'A': ['T'], 'B': ['T']}
links:dict = {
    'S':['A','E'],
    'A':['B'],
    'B':['C','T'],
    'C':['D'],
    'D':['T'],
    'E':['F'],
    'F':['T']}

# %%
columns = nodes.copy()
columns.append('prob')
paths = pd.DataFrame(columns=columns)
# %%

def exh_algthm(node_id:int,path:list,prob:int):
    node = nodes[node_id] # node such as 'S' and 'A'
    node_loc = loc[node] # node_loc such as [.3, .5, .2]
    for i in range(len(node_loc)):
        path.append(i)
        prob *= node_loc[i]
        if node != 'T':
            path, prob = exh_algthm(node_id+1,path,prob)
            path.pop()
            prob /= node_loc[i]
        else:
            path_prob = path.copy()
            path_prob.append(prob)
            paths.loc[len(paths)] = path_prob 
            path.pop()
            prob /= node_loc[i]
    
    return path, prob


# %%
exh_algthm(0,[],1)
# %%
def path_isConnected(node:str,path:pd.Series):
    node_pos = int(path[node])
    neighbours = links[node]
    for neighbour in neighbours:
        neighbour_pos = int(path[neighbour])
        if isConnected(node,neighbour,node_pos,neighbour_pos):
            if neighbour == 'T':
                raise ConnectedPathException('The path is connected')   
            else:
                path_isConnected(neighbour,path)
    

#%%
def isConnected(node:str,neighbour:str,node_pos:int,neighbour_pos:int):

    connections = loc_links[(node,neighbour)]
    connection = connections[node_pos][neighbour_pos]
    if connection == 1:
        return True
    return False


# %%
class ConnectedPathException(Exception):
    pass

paths['Connected'] = False
for i in range(len(paths)):
    path = paths.loc[i]
    try:
        path_isConnected('S',path)
    except ConnectedPathException as e:
        paths.loc[i,'Connected'] = True
        continue
    paths.loc[i,'Connected'] = False
# %%
prob_total = paths.loc[(paths['Connected'] == True)]['prob'].sum()
prob_total
# %%
