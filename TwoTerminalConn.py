import DisjointPaths as dis_p
import physical_model_simulation as pms
import ExhaustiveAlgorithm as ex_algthm
import pandas as pd
import argparse

class TwoTerminal:
    '''
    Computes the connectivity between two nodes (terminals) on a node disjoint path graph
    '''
    def __init__(self, links, loc, loc_links, paths, algorithm:str = 'MaxFlow'):
        self.loc = loc  # the locality sets of all nodes
        self.loc_links = loc_links  # the links between nodes. For example, for nodes x and y, the format is as follows
        self.dis_paths = dis_p.DisjointPaths(links)# type: ignore
        self.dps = self.dis_paths.runMaxFlow() if algorithm == 'MaxFlow' else self.dis_paths.runSSSP() 
        self.df_paths = paths[paths['Connected'] == True].copy()
        self.df_paths['prob'] = [1] * len(self.df_paths) # reset the probability to 1
        self.ConnectedPathException = type('ConnectedPathException', (Exception,), {})
        self.NotConnectedPathException = type('NotConnectedPathException', (Exception,), {})
        self.two_terminal_data = {}

    def two_terminal(self,node_id:int,path_index:int,df_path,dp:list):
        node = dp[node_id]
        neighbour = dp[node_id+1]
        node_pos = int(df_path.loc[path_index,node]) # the locality set of the node in the path
        neighbour_pos = int(df_path.loc[path_index,neighbour]) # the locality set of the neighbour in the path
        if self.isConnected(node,neighbour,node_pos,neighbour_pos):
            df_path.loc[path_index,'prob'] *= self.loc[neighbour][neighbour_pos]
            if neighbour == 'T':
                raise self.ConnectedPathException('The path is connected')
            else:
                self.two_terminal(node_id+1,path_index,df_path,dp)
        else:
            raise self.NotConnectedPathException('The path is not connected')
        
    def isConnected(self,node:str,neighbour:str,node_pos:int,neighbour_pos:int):

        connections = self.loc_links[(node,neighbour)]
        connection = connections[node_pos][neighbour_pos]
        if connection == 1:
            return True
        return False
    
    def get_connectivity(self):
        '''
        Computes the connectivity between two nodes (terminals) on a node disjoint path graph
        Args:
            None
        Returns:
            conn (float): the connectivity between two nodes (terminals) on a node disjoint path graph
        '''
        
        conn = 0
        for i in range(len(self.loc['S'])):
            s_prob = self.loc['S'][i]
            for j in range(len(self.loc['T'])):
                j_prob = self.loc['T'][j]
                temp = 1
                for dp in self.dps:
                    connected_df = self.two_terminal_data[tuple(dp)]
                    connected_df = connected_df[(connected_df['Connected'] == True) & (connected_df['S'] == i) & (connected_df['T'] == j)]
                    temp *= 1 - (connected_df['prob'].sum() / (s_prob * j_prob))
                conn += s_prob*j_prob*(1-temp)
        return round(conn,2)

    
    
    def main(self):
        for dp in self.dps:
            dp_copy = dp.copy()
            dp_copy.append('Connected')
            dp_copy.append('prob')
            df_path = self.df_paths[dp_copy]
            df_path = df_path.drop_duplicates(subset=dp) # drop duplicate paths
            df_path.reset_index(drop=True,inplace=True) # reset the index starting from 0
            for i in range(len(df_path)):
                node_pos = int(df_path.loc[i,'S'])
                df_path.loc[i,'prob'] *= self.loc['S'][node_pos]
                try:
                    self.two_terminal(0,i,df_path,dp)
                except self.ConnectedPathException as e:
                    df_path.loc[i,'Connected'] = True
                except self.NotConnectedPathException as e:
                    df_path.loc[i,'Connected'] = False
            self.two_terminal_data[tuple(dp)] = df_path
        
        conn = self.get_connectivity()
        return conn


def dummy_data():
    loc:dict = {'A':[.15,.25,.3,.3], 'B':[.4,.2,.4], 'C':[.2,.3,.4,.1], 'D':[.4,.3,.3],
            'E':[.5,.5], 'F':[.4,.6],
            'S':[.3,.5,.2], 'T':[.8,.2]}
    loc = {'S': [.25, .25, .5], 'A': [.25, .25, .5], 'B': [.25, .25, .5], 'T': [.25, .25, .5]}
    
    links:dict = {
    'S':['A','E'],
    'A':['B'],
    'B':['C','F','T'],
    'C':['D'],
    'D':['T'],
    'E':['F'],
    'F':['T']}

    links = {'S': ['A', 'B', 'T'], 'A': ['B'], 'B': ['T']}
                
    loc_links = pd.DataFrame({('A','B'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                            ('B','C'): {0:[1,1,1,1],1:[1,1,1,1], 2:[1,1,1,1]},
                            ('C','D'): {0:[0,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                            ('E','F'): {0:[1,1],1:[1,1]},
                            ('S','A'): {0:[1,1,1,1], 1:[1,1,1,1], 2:[1,1,1,1]},
                            ('D','T'): {0:[1,1], 1:[0,1], 2:[1,1]},
                            ('S','E'): {0:[1,1],1:[0,1],2:[0,1]},
                            ('B','T'): {0:[1,0],1:[0,0],2:[0,0]},
                            ('F','T'): {0:[1,0],1:[1,1]}})
    
    loc_links = pd.DataFrame({('S','A'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1]},
                            ('S','B'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1]},
                            ('S','T'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1]},
                            ('A','B'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1]},
                            ('B','T'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1]}})
    

    return loc, links, loc_links
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--test",action="store_true")
    parser.add_argument("-n","--nodes")
    parser.add_argument("-l","--locality")
    parser.add_argument("-p","--plot",action="store_true")
    parser.add_argument("-r","--run",action="store_true")
    args = parser.parse_args()
    loc = {}
    links = {}
    loc_links = {}
    nodes = []
    if args.test:
        nodes, loc, loc_links, links = ex_algthm.dummy_data()
    elif args.run:
        if args.nodes and args.locality:
            num_nodes = int(args.nodes)
            num_locality = int(args.locality)
        else:
            num_nodes = 6
            num_locality = 3
        
        phys_model = pms.PhysicalModel(number_of_nodes=num_nodes, loc_set_max=num_locality)
        loc, links, loc_links,nodes  = phys_model.get_data()
        if args.plot:
            phys_model.plot_underlying_graph(links)
        print('loc: ', loc)
        print('links: ', links)
        print('loc_links: ', loc_links)
    else:
        print('Please enter the correct arguments')
        exit()
    
    ex_algthm = ex_algthm.ExhaustiveAlgorithm(nodes=nodes,loc=loc,loc_links=loc_links, links=links)
    ex_algthm.main()
    paths = ex_algthm.paths.copy()
    conn = TwoTerminal(links=links, loc=loc, loc_links=loc_links,paths= paths).main()
    print('Two Terminal Conn: ', conn)
    

    