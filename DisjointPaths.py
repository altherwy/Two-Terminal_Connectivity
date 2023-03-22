
from Graph import *
from typing import List,Dict,Tuple
import GraphSSSP as sssp

class DisjointPaths:

    def __init__(self, links:Dict = None) -> None:
        self.links:Dict = links # the graph
        self.numNodes:int = len(self.links) + 1 # the number of nodes (+1 for T)
        self.nodeLetters:List = list(self.links.keys()) # the nodes (e.g., S,A,B,C,...,T)
   

    def mainBuildGraph(self)->List:
        '''
        Build a 2D array to represent the Graph.

        Args:
            links: Dictionary of each node and its neighbours as letters (e.g., 'S':['A','B'])
        
        Returns:
            The graph as 2D array of 0s and 1s, where the 1s represent the edges between the nodes
        '''
        

        indexes:Dict = {'T':len(self.nodeLetters)} # {'S':0,'A':1,'B':2,...,'T':len(nodeLetters)}
        indexesValues:Dict = {len(self.nodeLetters):'T'}
        for letter,indexValue in zip(self.nodeLetters,range(len(self.nodeLetters))):
            indexes[letter] = indexValue
            indexesValues[indexValue] = letter

        # Build the graph
        def buildGraph()-> np.array:
            graph:List = []
            for node in self.nodeLetters: # S
                neighbours:List = self.links[node] # The neighbours for each node (e.g., ['A','B','C'])
                indexesList = nodeLettersToIndexes(neighbours) # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
                graph.append(indexesList) # [[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0] ]
            graph.append([0]*self.numNodes) # add [0,0,0, ..., 0] for T
            return np.array(graph)

        # Convert the letters to 0s and 1s
        def nodeLettersToIndexes(neighbours:List) -> List:
            indexesList:List = [0]*self.numNodes # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for neighbour in neighbours: # ['A','B','C']
                indexesList[indexes[neighbour]] = 1 # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
            return indexesList
        
        return buildGraph(),indexesValues
        


    def runMaxFlow(self)->List:
        '''
        Run the Max Flow algorithm 

        Args:
            links: The Graph 
        
        Returns:
            The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
        '''
        
        graph,indexesValues = self.mainBuildGraph()
        EdmondKarp_graph = Graph(graph)
        dict_ = EdmondKarp_graph.EdmondKarp(data=True)
        list_ = dict_['iteration']
        disjointPaths:List = []
        for path in list_:
            pathList = list(path.values())[0]
            temp:List = []
            for node in pathList:
                letter = indexesValues[node]
                temp.append(letter)
            disjointPaths.append(temp)

        # Return the disjoint paths without the two terminal nodes (i.e., S and T)
        def filterDisjointPaths(disjointPaths:List)->List:
            disjointPathsNoTerminal:List = []
            for disjointPath in disjointPaths:
                if not (len(disjointPath) == 2):
                    disjointPathsNoTerminal.append(disjointPath[1:len(disjointPath)-1])
            return disjointPathsNoTerminal        

        # Remove all but one of the paths that have similar nodes
        # The retained path is the longest
        def removeOverlapPaths(disjointPaths:List)-> List:
            filteredList:List = filterDisjointPaths(disjointPaths)
            filteredList.sort(key=len)
            filteredList.reverse
            outputList:List = [filteredList[0]]
            for i in range(1,len(filteredList)):
                temp:List = filteredList[i]
                flag:bool = True # Permission to Add
                for lst in outputList:

                    if  list(set(temp) & set(lst)): # Not Empty
                        flag = False
                        break
                if flag:
                    outputList.append(temp)
            return outputList



        # Return the final disjoint paths 
        def addTerminalsToDisjointPaths(outputList:List):
            disjointPathTerminals:List = []
            for disjointPath in outputList:
                disjointPath.insert(0,'S')
                disjointPath.insert(len(disjointPath),'T')
                disjointPathTerminals.append(disjointPath)
            return disjointPathTerminals
            
        disjointPaths = addTerminalsToDisjointPaths(removeOverlapPaths(disjointPaths))    
        return disjointPaths
            
    def runSSSP(self)->Tuple[List,List]:
        '''
        Run the Single Source Shortest Path (SSSP) algorithm

        Args:
            links: The Graph
        
        Returns:
            The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
            The node letter values (i.e., [S,A,....,T])
        '''
        graph, indexesValues = self.mainBuildGraph()
        disjointPaths:List = []
        while True:
            g = sssp.GraphSSSP(len(graph),graph)
            try:
                g.dijkstra(0)
            except:
                break
            sourceNode:int = 0
            sinkNode:int = len(graph) - 1

            targetNode:int = sinkNode
            relayNodes:List = []
            if  not sinkNode in g.results:
                break
            disjointPath = []
            while True:
                node:int = g.results[targetNode]
                disjointPath.append(targetNode)
                if not node == sinkNode and not node == sourceNode:
                    relayNodes.append(node)
                    targetNode = node
                else:
                    disjointPath.append(0)
                    disjointPaths.append(disjointPath)
                    break
                
                    
            
            for nodeIndex in relayNodes:
                g.graph[nodeIndex] = len(g.graph[0]) * [0]
            graph = g.graph

            # turns the SSSP paths from number to letter (e.g., 0 to S, 1 to A, ...., n to S)    
            def formatSSSP()->List:
                final_paths = []
                for path in disjointPaths:
                    path.reverse()
                    formated_path = []
                    for node in path:
                        letter:str = indexesValues[node]
                        formated_path.append(letter)
                    final_paths.append(formated_path)
                return final_paths
            
        return formatSSSP()
    # Print the disjoint paths
    def printPaths(self,disjointPaths:List, algorithm = 'Max Flow'):
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
        for path in disjointPaths:
            for node in path:
                if not node == 'T':
                    print(node,end='->')
                else:
                    print(node)
    def main(self) -> None:

        disjointPaths = self.runMaxFlow()
        self.printPaths(disjointPaths)

        disjointPaths = self.runSSSP()
        self.printPaths(disjointPaths,'SSSP') 
if __name__ == '__main__':
    '''
    links:Dict = { 
                'S':['A','B','C','D','E'],
                'A':['B','C','D','E','T'],
                'B':['C','D','E','T'],
                'C':['D','E','T'],
                'D':['E','T'],
                'E':['T']}
    '''
    links:Dict = {
        'S':['A','E'],
            'A':['B'],
            'B':['C','F'],
            'C':['D'],
            'D':['T'],
            'E':['F'],
            'F':['T']
    }
    DisjointPaths(links).main()