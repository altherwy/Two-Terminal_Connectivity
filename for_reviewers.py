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
    


    
    

if __name__ == "__main__":
    analyzer = ConnectivityAnalyzer('experiment_list.csv')
    print(analyzer.disjoint_paths)
    print('-----------------')
    print(analyzer.loc)
    print('-----------------')
    print(analyzer.loc_links)
    print('-----------------')
    print(analyzer.nodes)


