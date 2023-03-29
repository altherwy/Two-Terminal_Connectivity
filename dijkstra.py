
#%%
import collections as coll

def dijkstra_paths(graph, start, goal):
    """Find all paths between start and goal vertices.

    Args:
        graph (dict): Graph represented as adjacency list.
        start (str): Start vertex.
        goal (str): Goal vertex.

    Returns:
        list: All paths between start and goal vertices.

    """
    # add the goal to the graph
    graph['T'] = set()
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph[vertex]) - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

def __get_disjoint_paths(paths:list, start:str = 'S', goal:str = 'T')->list:
    if not paths:
        return []
    paths.sort(key=len) # from the shortest path to the longest
    disjoint_paths = []
    for path in paths:
        # flag to check if 'path' should be added to 'disjoint_paths'
        add_flag = True 
        for disjoint_path in disjoint_paths:
            # the intersection between 'path' and 'disjoint_path'
            intersection = list(set(path) & set(disjoint_path))
            # the two terminal nodes 
            two_terminals = [start,goal]
            
            if not (coll.Counter(intersection) == coll.Counter(two_terminals)):
                # do not add the path to the disjoint_paths
                add_flag = False 
                break
        
        if add_flag:
            yield path



graph = {'S':['A','E'],
        'A':['B'],
        'B':['C','F'],
        'C':['D'],
        'D':['T'],
        'E':['F'],
        'F':['T']}

paths:list = list(dijkstra_paths(graph,'S','T'))
disjoint_paths:list =list(__get_disjoint_paths(paths))
print(disjoint_paths)

# %%
