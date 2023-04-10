import pandas as pd
class ExhaustiveAlgorithm:
    '''
    Computes the exact connectivity between two nodes (terminals)
    '''
    def __init__(self, nodes: list = None, loc: dict = None, loc_links: pd.DataFrame = None) -> None:
        self.loc = loc  # the locality sets of all nodes
        self.nodes = nodes  # the nodes in the graph
        self.loc_links = loc_links  # the links between nodes. For example, for nodes x and y, the format is as follows
        # {x_0:[y_0,y_1, ..., y_n], x_1:[y_0,y_1,..., y_n], ... , x_n:[y_0,y_1, ..., y_n]}
        self.columns = nodes + ['Probability'];
        self.prob = pd.DataFrame({}, columns=self.columns)
        self.conn_prob = pd.DataFrame({}, columns=self.columns)   

    def exhaustive_algorithm(self, node_ind: int, s_p: int, path: dict, conn_path: dict) -> int:
        """
        This method performs an exhaustive search of all possible paths from the source node to the target node.

        Args:
            node_ind (int): The index of the current node in the nodes list.
            s_p (int): The probability of the path so far.
            path (dict): A dictionary containing the nodes in the current path.
            conn_path (dict): A dictionary containing the connections in the current path.

        Returns:
            int: The index of the current node in the nodes list.
        """
        node: str = self.nodes[node_ind]  # get the current node
        node_loc: list = self.loc[node]  # get the location of the current node

        for j in range(len(node_loc)):
            conn_path[node] = j  # add connection to conn_path
            p = node_loc[j]  # get probability of connection
            s_p *= p  # update probability of path so far
            path: dict = self.__check_connection(node, j, path)  # add node to path
            if node != 'T':
                node_ind += 1  # next node in nodes list
                node_ind = self.exhaustive_algorithm(node_ind, s_p, path, conn_path)  # recursive call
                s_p /= p  # update probability of path so far
            else:

                indices = [v for k, v in path.items()]
                indices.append(s_p)
                self.prob.loc[len(self.prob.index)] = indices

                indices = [v for k, v in conn_path.items()]
                indices.append(s_p)
                self.conn_prob.loc[len(self.conn_prob.index)] = indices

                path.popitem()  # remove last node
                conn_path.popitem()  # remove last connection
                s_p /= p  # update probability of path so far
                
                node_ind -= 1  # go back to previous node     
                return node_ind
        node_ind -= 1  # go back to previous node     
        return node_ind
    def __check_connection(self,node: str, index: int, path: dict) -> dict:
        for k, v in path.items():  # {S:0,A:0}
            try:
                links: dict = self.loc_links[(k, node)]
            except KeyError as err:
                continue
            if v == -1:
                continue
            e = links[v][index]
            if e == 1:
                path[node] = index
                return path
        path[node] = -1
        return path
    


    def main(self):
        tot_conn = 0
        node_loc: list = self.loc['S']
        for i in range(len(node_loc)):
            path: dict = {'S': i}
            conn_path: dict = {'S': i}
            self.exhaustive_algorithm(1, node_loc[i], path, conn_path)
        # the total connectivity of the graph
        tot_conn = round(sum(self.prob.loc[self.prob['T'] != -1]['Probability']), 7)
        print(tot_conn)
        

if __name__ == '__main__':
    nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'T']
    loc: dict = {'A': [.15, .25, .3, .3], 'B': [.4, .2, .4], 'C': [.2, .3, .4, .1], 'D': [.4, .3, .3],
                        'E': [.5, .5], 'F': [.4, .6],
                        'S': [.3, .5, .2], 'T': [.8, .2]}
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
    
    ExhaustiveAlgorithm(nodes=nodes,loc=loc,loc_links=loc_links).main()
