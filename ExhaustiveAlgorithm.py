import argparse
import pandas as pd
class ExhaustiveAlgorithm:
    '''
    Computes the exact connectivity between two nodes (terminals)
    '''
    def __init__(self, nodes: list, loc: dict, loc_links: pd.DataFrame, links:dict) -> None:
        self.loc = loc  # the locality sets of all nodes
        self.nodes = nodes  # the nodes in the graph
        self.loc_links = loc_links  # the links between nodes. For example, for nodes x and y, the format is as follows
        self.links = links # the neighbours of each node
        self.columns = self.nodes.copy()
        self.columns.append('prob')
        self.paths = pd.DataFrame(columns=self.columns)
        self.ConnectedPathException = type('ConnectedPathException', (Exception,), {})
    
    def exhaustive_algorithm(self, node_id: int, path: list, prob: float) -> tuple:
        '''
        This method computes the exact connectivity between two nodes (terminals)
        Args:
            node_id (int): The index of the node in the nodes list
            path (list): The path from the source to the current node
            prob (float): The probability of the path
        Returns:
            paths (pd.DataFrame): The paths from the source to the destination
            prob (float): The probability of the path
        '''
        node = self.nodes[node_id]
        node_loc = self.loc[node] # node_loc such as [.3, .5, .2]
        for i in range(len(node_loc)):
            path.append(i)
            prob *= node_loc[i]
            if node != 'T':
                path, prob = self.exhaustive_algorithm(node_id+1,path,prob)
                path.pop()
                prob /= node_loc[i]
            else:
                path_prob = path.copy()
                path_prob.append(prob)
                self.paths.loc[len(self.paths)] = path_prob  # type: ignore
                path.pop()
                prob /= node_loc[i]
        
        return path, prob
    
    
    
    def path_isConnected(self,node:str,path:pd.Series):
        node_pos = int(path[node])
        neighbours = self.links[node]
        for neighbour in neighbours:
            neighbour_pos = int(path[neighbour]) # type: ignore
            if self.isConnected(node,neighbour,node_pos,neighbour_pos):
                if neighbour == 'T':
                    raise self.ConnectedPathException('The path is connected')   
                else:
                    self.path_isConnected(neighbour,path)
    
    def isConnected(self,node:str,neighbour:str,node_pos:int,neighbour_pos:int):

        connections = self.loc_links[(node,neighbour)]
        connection = connections[node_pos][neighbour_pos]
        if connection == 1:
            return True
        return False

  
    def main(self):
        _,_ = self.exhaustive_algorithm(0,[],1)
        self.paths['Connected'] = False
        for i in range(len(self.paths)):
            path = self.paths.loc[i]
            try:
                self.path_isConnected('S',path)
            except self.ConnectedPathException as e:
                self.paths.loc[i,'Connected'] = True
                continue
            self.paths.loc[i,'Connected'] = False
        
        print('Connectivity:',round(self.paths[self.paths['Connected'] == True]['prob'].sum(),2))


        
    

 
def dummy_data():
    '''
    This method creates a dummy data set for testing purposes.
    Args:
        None
    Returns:
        nodes (list): The nodes in the graph
        loc (dict): The locality sets of all nodes
        loc_links (pd.DataFrame): The links between nodes
    '''
    nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'T']
    #nodes = ['S', '1', '2', '3', '4', 'T']

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
    
    loc_links = pd.DataFrame({('A', 'B'): {0: [1, 1, 1], 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1]},
                                ('B', 'C'): {0: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]},
                                ('C', 'D'): {0: [0, 1, 1], 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1]},
                                ('E', 'F'): {0: [1, 1], 1: [1, 1]},
                                ('S', 'A'): {0: [1, 1, 1, 1], 1: [1, 1, 1, 1], 2: [1, 1, 1, 1]},
                                ('D', 'T'): {0: [1, 1], 1: [0, 1], 2: [1, 1]},
                                ('S', 'E'): {0: [1, 1], 1: [1, 1], 2: [1, 1]},
                                ('B', 'T'): {0: [1, 0], 1: [0, 0], 2: [0, 0]},
                                ('F', 'T'): {0: [1, 1], 1: [1, 1]}
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
    links:dict = {
    'S':['A','E'],
    'A':['B'],
    'B':['C','T'],
    'C':['D'],
    'D':['T'],
    'E':['F'],
    'F':['T']}
    return nodes, loc, loc_links, links


        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test",action="store_true")
    parser.add_argument("-n","--nodes")
    parser.add_argument("-l","--locality")
    parser.add_argument("-p","--plot",action="store_true")
    parser.add_argument("-r","--run",action="store_true")
    args = parser.parse_args()
    
    if args.test:
        nodes, loc, loc_links, links = dummy_data()
        ea = ExhaustiveAlgorithm(nodes=nodes,loc=loc,loc_links=loc_links, links=links)
        ea.main()
