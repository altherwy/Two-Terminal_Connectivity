import pandas as pd
from ExhaustiveAlgorithm import input
from ExhaustiveAlgorithm import run_physical_model
import DisjointPaths as dis_p
from numpy import prod

class ConnectivityAnalyzer:
    def __init__(self, experiment_list_file):
        self.experiment_list_file = experiment_list_file
        self.disjoint_paths, self.loc, self.loc_links, self.loc_links, self.nodes, self.file_name = self._preprocessing()
        self.all_paths = pd.DataFrame()
        

    def _preprocessing(self):
        df_experiment_list = pd.read_csv(self.experiment_list_file)
        for i in range(len(df_experiment_list)):
            number_nodes = int(df_experiment_list.iloc[i]['number_nodes'])
            loc_set_max = int(df_experiment_list.iloc[i]['loc_set_max'])
            connection_level = int(df_experiment_list.iloc[i]['connection_level'])

            file_name, number_cores = run_physical_model(number_of_nodes=number_nodes, loc_set_max=loc_set_max, conn_level=connection_level)
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
    

if __name__ == "__main__":
    analyzer = ConnectivityAnalyzer('experiment_list.csv')
    for dp in analyzer.disjoint_paths:
        paths = analyzer.generate_paths(dp)
        prod_paths = analyzer.multiply_probabilities(paths,dp)
        processed_paths = analyzer.connected_paths(prod_paths, dp)
        analyzer.all_paths = pd.concat([analyzer.all_paths, processed_paths])
        analyzer.all_paths.to_csv('.csv', index=False)
        

   


