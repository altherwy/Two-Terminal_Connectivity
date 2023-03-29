

from Graph import *
from collections import Counter
class DisjointPaths:
    '''
    Returns the disjoint paths from a given graph using Max flow and SSSP algorithms
    '''

    def __init__(self, graph:dict = None, start:str = 'S', goal:str = 'T') -> None:
        self.graph:dict = graph # the graph
        # the start node, usually 'S'
        self.start = start
        # the end node, usually 'T'
        self.goal = goal
        self.numNodes:int = len(self.graph) + 1 # the number of nodes (+1 for T)
        self.nodeLetters:list= list(self.graph.keys()) # the nodes (e.g., S,A,B,C,...,T)
   

    def mainBuildGraph(self)->list:
        '''
        Build a 2D array to represent the Graph.

        Args:
             graph: Dictionary of each node and its neighbours as letters (e.g., 'S':['A','B'])
        
        Returns:
            The graph as 2D array of 0s and 1s, where the 1s represent the edges between the nodes
        '''
        

        indexes:dict = {'T':len(self.nodeLetters)} # {'S':0,'A':1,'B':2,...,'T':len(nodeLetters)}
        indexesValues:dict = {len(self.nodeLetters):'T'}
        for letter,indexValue in zip(self.nodeLetters,range(len(self.nodeLetters))):
            indexes[letter] = indexValue
            indexesValues[indexValue] = letter

        # Build the graph
        def buildGraph()-> np.array:
            graph:list= []
            for node in self.nodeLetters: # S
                neighbours:list= self.graph[node] # The neighbours for each node (e.g., ['A','B','C'])
                indexesList = nodeLettersToIndexes(neighbours) # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
                graph.append(indexesList) # [[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0] ]
            graph.append([0]*self.numNodes) # add [0,0,0, ..., 0] for T
            return np.array(graph)

        # Convert the letters to 0s and 1s
        def nodeLettersToIndexes(neighbours:list) -> list:
            indexesList:list= [0]*self.numNodes # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for neighbour in neighbours: # ['A','B','C']
                indexesList[indexes[neighbour]] = 1 # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
            return indexesList
        
        return buildGraph(),indexesValues
        


    def runMaxFlow(self)->list:
        '''
        Run the Max Flow algorithm 

        Args:
            None
        
        Returns:
            The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
        '''
        
        graph,indexesValues = self.mainBuildGraph()
        EdmondKarp_graph = Graph(graph)
        dict_ = EdmondKarp_graph.EdmondKarp(data=True)
        list_ = dict_['iteration']
        disjointPaths:list= []
        for path in list_:
            pathList = list(path.values())[0]
            temp:list= []
            for node in pathList:
                letter = indexesValues[node]
                temp.append(letter)
            disjointPaths.append(temp)

        # Return the disjoint paths without the two terminal nodes (i.e., S and T)
        def filterDisjointPaths(disjointPaths:list)->list:
            disjointPathsNoTerminal:list= []
            for disjointPath in disjointPaths:
                if not (len(disjointPath) == 2):
                    disjointPathsNoTerminal.append(disjointPath[1:len(disjointPath)-1])
            return disjointPathsNoTerminal        

        # Remove all but one of the paths that have similar nodes
        # The retained path is the longest
        def removeOverlapPaths(disjointPaths:list)-> list:
            filteredList:list= filterDisjointPaths(disjointPaths)
            filteredList.sort(key=len)
            filteredList.reverse
            outputList:list= [filteredList[0]]
            for i in range(1,len(filteredList)):
                temp:list= filteredList[i]
                flag:bool = True # Permission to Add
                for lst in outputList:

                    if  list(set(temp) & set(lst)): # Not Empty
                        flag = False
                        break
                if flag:
                    outputList.append(temp)
            return outputList



        # Return the final disjoint paths 
        def addTerminalsToDisjointPaths(outputList:list):
            disjointPathTerminals:list= []
            for disjointPath in outputList:
                disjointPath.insert(0,'S')
                disjointPath.insert(len(disjointPath),'T')
                disjointPathTerminals.append(disjointPath)
            return disjointPathTerminals
            
        disjointPaths = addTerminalsToDisjointPaths(removeOverlapPaths(disjointPaths))    
        return disjointPaths
            
    def runSSSP(self)->list:
        '''
        Run the dijkstra algorithm

        Args:
            None
        
        Returns:
            The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
        '''

        def __dijkstra_paths()->list:
            """Find all paths between start and goal vertices.

            Args:
                graph (dict): Graph represented as adjacency list.
                start (str): Start vertex.
                goal (str): Goal vertex.

            Returns:
                list: All paths between start and goal vertices.

            """
            # add the goal to the graph
            self.graph['T'] = set()
            stack = [(self.start, [self.start])]
            while stack:
                (vertex, path) = stack.pop()
                for next in set(graph[vertex]) - set(path):
                    if next == self.goal:
                        yield path + [next]
                    else:
                        stack.append((next, path + [next]))

        def __get_disjoint_paths(paths:list)->list:
            """
            This method takes a list of paths and returns a list of disjoint paths. 
            A path is considered disjoint if it does not share any nodes with any other path in the list (except the two terminal nodes).

            Args:
                paths (list): A list of paths.

            Returns:
                list: A list of disjoint paths.

            """
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
                    two_terminals = [self.start,self.goal]

                    if not (Counter(intersection) == Counter(two_terminals)):
                        # do not add the path to the disjoint_paths
                        add_flag = False 
                        break

                if add_flag:
                    disjoint_paths.append(path)
                    yield path
                    


        paths:list = list(__dijkstra_paths())
        disjoint_paths:list =list(__get_disjoint_paths(paths))
        return disjoint_paths
       
    def printPaths(self,disjoint_paths:list, algorithm = 'Max Flow'):
        '''
        Print the disjoint paths

        Args:
            disjointPaths: The disjoint paths
            indexesValues: The node letter values (i.e., [S,A,....,T])
            Algorithm: Max Flow or SSSP
        
        Returns: 
            None
        '''
        print_statement = 'Paths from Max Flow Algorithm' if algorithm == 'Max Flow' else 'Paths from SSSP Algorithm'
        print(print_statement)
        for path in disjoint_paths:
            for node in path:
                if not node == 'T':
                    print(node,end='->')
                else:
                    print(node)
    def main(self) -> None:

        disjoint_paths = self.runMaxFlow()
        self.printPaths(disjoint_paths)

        disjoint_paths = self.runSSSP()
        self.printPaths(disjoint_paths,'SSSP') 
if __name__ == '__main__':
    '''
    graph:dict = { 
                'S':['A','B','C','D','E'],
                'A':['B','C','D','E','T'],
                'B':['C','D','E','T'],
                'C':['D','E','T'],
                'D':['E','T'],
                'E':['T']}
    '''
    graph:dict = {
        'S':['A','E'],
            'A':['B'],
            'B':['C','F'],
            'C':['D'],
            'D':['T'],
            'E':['F'],
            'F':['T']
    }
    DisjointPaths(graph).main()