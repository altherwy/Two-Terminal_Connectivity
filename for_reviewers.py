import pandas as pd
from ExhaustiveAlgorithm import input
from ExhaustiveAlgorithm import run_physical_model
import DisjointPaths as dis_p
from numpy import prod
from datetime import datetime

class ConnectivityAnalyzer:
    def __init__(self, number_nodes:int, loc_set_max:int, connection_level:int):
        self.number_nodes = number_nodes
        self.loc_set_max = loc_set_max
        self.connection_level = connection_level

        self.disjoint_paths, self.loc, self.loc_links, self.loc_links, self.nodes, self.file_name = self._preprocessing()
        self.all_paths = pd.DataFrame()
        

    def _preprocessing(self):
        

            file_name, number_cores = run_physical_model(number_of_nodes=self.number_nodes, loc_set_max=self.loc_set_max, conn_level=self.connection_level)
            loc, links, loc_links, nodes = input(file_name)
            disjoint_paths = self._get_disjoint_paths(links)
            return disjoint_paths, loc, loc_links, loc_links, nodes, file_name
    
    def _get_disjoint_paths(self, links):
        dis_paths = dis_p.DisjointPaths(links)
        dps = dis_paths.runMaxFlow()
        return dps
    
    def generate_paths(self,dp)->list:
        
        num_nodes = len(dp)
        num_locs = [len(self.loc[node]) for node in dp]
        num_paths = prod(num_locs)
        paths = []
        for i in range(num_paths):
            path = []
            for j in range(num_nodes):
                path.append(i % num_locs[j])
                i //= num_locs[j]   
            paths.append(path)
        paths = pd.DataFrame(paths, columns=dp)
        # set prob column to 1
        paths['prob'] = 1
        return paths


    def multiply_probabilities(self,paths, dp):
        
        num_paths = len(paths)
        for i in range(num_paths):
            prob = 1
            path = paths.iloc[[i]]
            for node in dp:
                if node != 'S' and node != 'T':
                    loc_index = path[node].values[0]
                    prob *= self.loc[node][loc_index]

            paths.loc[i,'prob'] = prob
        return paths
    
    def isConnected(self,node:str,neighbour:str,node_pos:int,neighbour_pos:int):

            connections = self.loc_links[(node,neighbour)]
            connection = connections[node_pos][neighbour_pos]
            if connection == 1:
                return True
            return False
    
    def connected_paths(self,paths, dp):
        paths['Connected'] = 'Not Processed'
        for i in range(len(paths)):
            path = paths.iloc[[i]]
            if path['Connected'].values[0] == 'Not Processed':
                # change path status to True
                paths.loc[i,'Connected'] = True
                paths =  self._is_path_connected(path,dp, paths)

        return paths    


    def _is_path_connected(self,path,dp, paths):
        for i in range(len(dp)-1):
            node = dp[i]
            neighbour = dp[i+1]
            node_pos = path[node].values[0]
            neighbour_pos = path[neighbour].values[0]
            if not self.isConnected(node,neighbour,node_pos,neighbour_pos):
                paths = self._flag_paths(paths, node, neighbour, node_pos, neighbour_pos, False)
        
                return paths
        return paths
        

    def _flag_paths(self, paths, node, neighbour, node_pos, neighbour_pos, flag):
        paths.loc[(paths[node] == node_pos) & (paths[neighbour] == neighbour_pos), 'Connected'] = flag
        return paths  

    def _get_df_for_dp(self,dp):
        df_connected = self.all_paths[self.all_paths['Connected'] == True]
        dp_df = df_connected[df_connected.columns.intersection(dp+['prob'])]
        return dp_df[dp_df.notnull().all(axis=1)] # remove rows with NaN values
        
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
                for dp in self.disjoint_paths:
                    connected_df = self._get_df_for_dp(dp)
                    connected_df = connected_df[(connected_df['S'] == i) & (connected_df['T'] == j)]
                    sum_prob = connected_df['prob'].sum()
                    temp *= 1 - sum_prob

                conn += s_prob*j_prob*(1-temp)
                
        return conn
    

def main():
    # run the physical model and get the disjoint paths
    analyzer = ConnectivityAnalyzer(number_nodes, loc_set_max, connection_level)
    # start timer
    import time
    start = time.time()

    
    for dp in analyzer.disjoint_paths:
        paths = analyzer.generate_paths(dp) # generate all possible paths
        prod_paths = analyzer.multiply_probabilities(paths,dp) # multiply probabilities
        processed_paths = analyzer.connected_paths(prod_paths, dp) # get connected paths
        analyzer.all_paths = pd.concat([analyzer.all_paths, processed_paths]) # append to all paths
        
    analyzer.all_paths.to_csv('results/'+analyzer.file_name + '.csv', index=False)

    conn = analyzer.get_connectivity()
    end = time.time()
    time_taken = end-start
    print(analyzer.disjoint_paths)
    print('Time taken:', time_taken)   
    return number_nodes, loc_set_max, connection_level, conn, time_taken

if __name__ == "__main__":

    df_experiment_list = pd.read_csv('experiment_list.csv')
    df_results = pd.DataFrame(columns=['V','Loc_max','Conn_Level','Connectivity','Running Time'])
    for i in range(len(df_experiment_list)):
        number_nodes = int(df_experiment_list.iloc[i]['number_nodes'])
        loc_set_max = int(df_experiment_list.iloc[i]['loc_set_max'])
        connection_level = int(df_experiment_list.iloc[i]['connection_level'])
        v,loc_max,conn_level,conn, runn_time =  main()
        ser = pd.Series([v,loc_max,conn_level,conn,runn_time],index=df_results.columns)
        df_results = pd.concat([df_results,ser.to_frame().T],axis=0)
    
    dt = datetime.now()
    ts = datetime.strftime(dt,'%Y%m%d%H%M%S')
    file_name  = 'results/results_'+ts+'.csv'
    df_results.to_csv(file_name,index=False)

     

   


