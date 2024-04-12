import pandas as pd
from ExhaustiveAlgorithm import input
from ExhaustiveAlgorithm import run_physical_model
import DisjointPaths as dis_p

class ConnectivityAnalyzer:
    def __init__(self, experiment_list_file):
        self.experiment_list_file = experiment_list_file
        self.disjoint_paths, self.loc, self.loc_links, self.loc_links, self.nodes = self._preprocessing()
        self.paths = []

    def _preprocessing(self):
        df_experiment_list = pd.read_csv(self.experiment_list_file)
        for i in range(len(df_experiment_list)):
            number_nodes = int(df_experiment_list.iloc[i]['number_nodes'])
            loc_set_max = int(df_experiment_list.iloc[i]['loc_set_max'])
            connection_level = int(df_experiment_list.iloc[i]['connection_level'])

            file_name, number_cores = run_physical_model(number_of_nodes=number_nodes, loc_set_max=loc_set_max, conn_level=connection_level)
            loc, links, loc_links, nodes = input(file_name)
            disjoint_paths = self._get_disjoint_paths(loc, links, loc_links, nodes)
            return disjoint_paths, loc, loc_links, loc_links, nodes
    
    def _get_disjoint_paths(self, loc, links, loc_links, nodes):
        dis_paths = dis_p.DisjointPaths(links)
        dps = dis_paths.runMaxFlow()
        return dps
    


    
    def genrate_all_paths(self):
        for dp in self.disjoint_paths:
         list_probs:list = self.loc[dp[0]] # get the probabilities of the locations of node S
         for i in range(len(list_probs)):
            prob = list_probs[i]
            path = {'S':i}
            self._generate_path(dp -> list,node_index = 0 -> int,loc_index = i -> int, prob = prob -> float, path =path -> dict)   
                
    def _generate_path(self,dp,index,prob,path):
        list_probs:list = self.loc[dp[index]] # get the probabilities of the locations of the current node
        prob *=  
        if dp[index] == 'T':
            
            
        paths = []
        for location, location_prob in self.loc[dp[index]].items():
            if location not in dp:
                new_path = dp + [location]
                new_prob = prob * location_prob
                paths += self._generate_path(dp,location, new_prob,new_path)
        return paths
          
    def analyze_connectivity(self):
        disjoint_paths = self._preprocessing()
        print(self.loc)
            


if __name__ == "__main__":
    analyzer = ConnectivityAnalyzer('experiment_list.csv')
    analyzer.analyze_connectivity()


