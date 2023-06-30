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
nodes
# %%
loc['S']
# %%
4*3*4*3*2*2*3*2
# %%
import pandas as pd
nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'T']
#nodes = ['S', '1', '2', '3', '4', 'T']
#nodes = ['S','A','B','T']

loc: dict = {'A': [.15, .25, .3, .3], 'B': [.4, .2, .4], 'C': [.2, .3, .4, .1], 'D': [.4, .3, .3],
                'E': [.5, .5], 'F': [.4, .6],
                'S': [.3, .5, .2], 'T': [.8, .2]}
'''

loc = {'S': [0.33, 0.66], 
        '1': [0.33], 
        '2': [0.33, 0.33], 
        '3': [0.33, 0.33], 
        '4': [0.33, 0.33], 
        'T': [0.33, 0.66]}

loc:dict = {
    'S':[0.2,0.3],
    'A':[0.5,0.5],
    'B':[0.4,0.6],
    'T':[0.3]
}
'''
loc_links = pd.DataFrame({('A', 'B'): {0: [1, 1, 1], 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1]},
                            ('B', 'C'): {0: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]},
                            ('C', 'D'): {0: [0, 1, 1], 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1]},
                            ('E', 'F'): {0: [1, 1], 1: [1, 1]},
                            ('S', 'A'): {0: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]},
                            ('D', 'T'): {0: [1, 1], 1: [0, 1], 2: [1, 1]},
                            ('S', 'E'): {0: [1, 1], 1: [0, 1], 2: [0, 1]},
                            ('B', 'T'): {0: [1, 0], 1: [0, 0], 2: [0, 0]},
                            ('F', 'T'): {0: [1, 0], 1: [1, 1]}
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
'''
# %%
columns = nodes
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
            #path_prob = build_path(path)
            #path_prob['prob'] = prob
            #paths.append(path_prob, ignore_index=True)
            path_prob = path.copy()
            path_prob.append(prob)
            paths.loc[len(paths)] = path_prob
            path.pop()
            prob /= node_loc[i]
    
    return path, prob


def build_path(path:list):
    dict_path = {}
    for node,loc in zip(nodes,path):
        dict_path[node] = loc
    return dict_path

# %%
exh_algthm(0,[],1)
# %%
paths
# %%
