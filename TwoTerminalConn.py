from DisjointPaths import DisjointPaths
import pandas as pd

class TwoTerminal:
    '''
    Computes the connectivity between two nodes (terminals) on a node disjoint path graph
    '''
    def __init__(self, dps:list=None, links:list = None, loc:dict=None, loc_links:pd.DataFrame = None, algorithm:str = 'MaxFlow') -> None:

         
        self.loc = loc # the locality sets of all nodes 
        self.loc_links = loc_links # the links between nodes. For example, for nodes x and y, the format is as follows
                                #{x_0:[y_0,y_1, ..., y_n], x_1:[y_0,y_1,..., y_n], ... , x_n:[y_0,y_1, ..., y_n]}
        self.algorithm = algorithm # the algorithm that returns the disjoint paths

        # dps is empty
        if dps is None:
            dis_paths = DisjointPaths(links)
            paths = dis_paths.runMaxFlow() if self.algorithm == 'MaxFlow' else dis_paths.runSSSP() 
            self.dps = [path[1:-1] for path in paths] # the disjoint paths without terminals (i.e., without S and T)
        else:
            self.dps = dps # the disjoint paths 
    
    def compute_T_prob(self)->dict:
        '''
        Compute the T_probability tables for the disjoint-paths

        Args:
            dps (list): Disjoint-paths
            loc (dict): Locality sets of all nodes 
            loc_links (Dataframe): The links between nodes. For example, for nodes x and y, the format is as follows
                                {x_0:[y_0,y_1, ..., y_n], x_1:[y_0,y_1,..., y_n], ... , x_n:[y_0,y_1, ..., y_n]} 
        
        Returns:
            The T_probability tables for the disjoint-paths
        '''
        T_prob_tables:dict = {}
        for dp in self.dps: 
            for i in range(len(dp) - 1):
                node:str = dp[i] # the current node i.e., 'A'
                if i == 0: # the first node in the disjoint-path
                    x = self.loc[node] # the locality set of the first node (e.g., [.4,.6])
                    first_rows_num = len(x) # the rows number of the first node in the disjoint-path
                next_node:str = dp[i+1]
                y:list = self.loc[next_node]

                rows_num = len(self.loc[node])
                columns_num = len(y)

                z = pd.DataFrame(0.0, index=range(first_rows_num), columns= range(columns_num)) # |Loc_x| * |Loc_y| 
                links:pd.DataFrame = self.get_links(node,next_node)
                for k in range(rows_num):
                    for l in range(columns_num):
                        if links.iat[k,l] == 1: # there is an edge
                            
                            if i == 0: # the first two nodes in a disjoint-path
                                z.iat[k,l] = z.iat[k,l] +  x[k]*y[l]
                            else:
                                
                                column_vals = x.iloc[:,k]
                                indices = range(len(column_vals)) # indices length for node i in [i,j] 
                                for val, index in zip(column_vals,indices):
                                    z.iat[index,l] += val * y[l]
                x = z
            T_prob_tables[tuple(dp)]= z
            

        return T_prob_tables

    def compute_prob_tables(self,T_prob_tables:dict)->dict:
        '''
        Compute the probability tables for the disjoint-paths

        Args:
            dps (list): Disjoint-paths
            T_prob_tables (dict): T_probability tables for the disjoint-paths

        Returns:
            The probability tables for the disjoint paths
        '''
        s:list = self.loc['S'] # the locality set of node 'S'
        t:list = self.loc['T'] # the locality set of node 'T'
        prob_tables:dict = {}
        rows_num:int = len(s)
        columns_num:int = len(t)
        
        for dp in self.dps: 
            prob = pd.DataFrame(0.0,index=range(rows_num), columns= range(columns_num)) # size of the table is |Loc_s| * |Loc_t|
            first_node:str = dp[0]
            last_node:str = dp[-1]
            
            links_s_first = self.get_links('S',first_node)
            links_last_t = self.get_links(last_node,'T')
            T_prob_table = T_prob_tables[tuple(dp)] # T_probability table for a disjoint-path

            first_node_indices = T_prob_table.index.array
            last_node_indices = T_prob_table.columns.array

            for i in range(rows_num):
                for j in range(columns_num):
                    for a in first_node_indices:
                        for b in last_node_indices:
                            try:
                                if links_s_first.iat[i,a] == 1 and links_last_t.iat[b,j] == 1:
                                    prob.iat[i,j] += T_prob_table.iat[a,b]
                            except IndexError as err:
                                    print('Error path:',i,'->',a,'->',b,'->',j)
                                    pass
                                
            prob_tables[tuple(dp)] = prob

        return prob_tables

    def compute_prob(self,prob_tables:dict)->pd.DataFrame:
        '''
        Compute the probability table for the disjoint-paths

        Args:
            dps (list): Disjoint-paths
            prob_tables (dict): The probability tables for the disjoint-paths

        Returns:
            A single probability table for all disjoint paths
        '''
        s:list = self.loc['S'] # the locality set of node 'S'
        t:list = self.loc['T'] # the locality set of node 'T'
        rows_num:int = len(s)
        columns_num:int = len(t)
        prob = pd.DataFrame(0.0, index= range(rows_num), columns=range(columns_num))
        for i in range(rows_num):
            for j in range(columns_num):
                temp:int = 1
                for dp in self.dps:
                    prob_table:pd.DataFrame = prob_tables[tuple(dp)]
                    temp *= 1 - prob_table.iat[i,j]
                prob.iat[i,j] = s[i]*t[j]*(1 - temp)
        return prob    

    def compute_2T_conn(self)->int:
        '''
        Compute the Two-Terminal connectivity

        Args:
            dps (list): Disjoint-paths
            loc (dict): The locality sets for all nodes
            loc_links (Dataframe): The links between nodes. For example, for nodes x and y, the format is as follows
                                    {x_0:[y_0,y_1, ..., y_n], x_1:[y_0,y_1,..., y_n], ... , x_n:[y_0,y_1, ..., y_n]}

        Returns:
            The Two-Terminal connectivity between S and T
        '''
        T_prob_tables:dict = self.compute_T_prob()
        prob_tables:dict = self.compute_prob_tables(T_prob_tables)
        prob:pd.DataFrame = self.compute_prob(prob_tables) 
        twoT_conn:int = 0
        s:list = self.loc['S'] # the locality set of node 'S'
        t:list = self.loc['T'] # the locality set of node 'T'
        rows_num:int = len(s)
        columns_num:int = len(t)
        for i in range(rows_num):
            for j in range(columns_num):
                twoT_conn += prob.iat[i,j]
        
        return twoT_conn
        
    def get_links(self,x:str, y:str)->pd.DataFrame:
        '''
        Return the links (i.e., edges) table between two nodes x and y

        Args:
            x (str): a node
            y (str): a neighbouring node to x 
        
        Returns:
            The links between nodes x and y as a pandas ``Dataframe``
        '''
        links_x_y = self.loc_links[(x,y)]      
        links_x_y = links_x_y.dropna() # remove NaN values
        rows_num = len(links_x_y)
        columns_num = len(links_x_y[0])
        links = pd.DataFrame(0.0,index=range(rows_num), columns=range(columns_num)) 
        for r in range(rows_num):
            for c in range(columns_num):
                links.iat[r,c] = links_x_y[r][c]
        return links



    def main(self) -> None:
        self.loc:dict = {'A':[.15,.25,.3,.3], 'B':[.4,.2,.4], 'C':[.2,.3,.4,.1], 'D':[.4,.3,.3],
                'E':[.5,.5], 'F':[.4,.6],
                'S':[.3,.5,.2], 'T':[.8,.2]}
        self.loc_links = pd.DataFrame({('A','B'): {0:[1,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                                ('B','C'): {0:[1,1,1,1],1:[1,1,1,1], 2:[1,1,1,1]},
                                ('C','D'): {0:[0,1,1], 1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                                ('E','F'): {0:[1,1],1:[1,1]},
                                ('S','A'): {0:[1,1,1,1], 1:[1,1,1,1], 2:[1,1,1,1]},
                                ('D','T'): {0:[1,1], 1:[0,1], 2:[1,1]},
                                ('S','E'): {0:[1,1],1:[0,1],2:[0,1]},
                                ('B','T'): {0:[1,0],1:[0,0],2:[0,0]},
                                ('F','T'): {0:[1,0],1:[1,1]}
                                })
        twoT_conn = self.compute_2T_conn()
        print(twoT_conn)

if __name__ == '__main__':
    links:dict = {
        'S':['A','E'],
        'A':['B'],
        'B':['C','F','T'],
        'C':['D'],
        'D':['T'],
        'E':['F'],
        'F':['T']
    }
    TwoTerminal(links=links).main()