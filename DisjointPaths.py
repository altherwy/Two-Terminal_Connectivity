#%% Build Graph
from Graph import *
from typing import List,Dict,Tuple
import GraphSSSP as sssp


def mainBuildGraph(links:Dict)->List[List[int]]:
    '''
    Objective: build a 2D array to represent the Graph.

    Parameters:
        links: Dictionary of each node and its neighbours as letters (e.g., 'S':['A','B'])
    
    Output:
        The graph as 2D array of 0s and 1s, where the 1s represent the edges between the nodes
    '''
    nodeLetters:List[str] = list(links.keys()) # The nodes (e.g., S,A,B,C,...,T)
    numNodes:int = len(links) + 1 # The number of nodes (+1 for T)

    indexes:Dict = {'T':len(nodeLetters)} # {'S':0,'A':1,'B':2,...,'T':len(nodeLetters)}
    indexesValues:Dict = {len(nodeLetters):'T'}
    for letter,indexValue in zip(nodeLetters,range(len(nodeLetters))):
        indexes[letter] = indexValue
        indexesValues[indexValue] = letter

    # Build the graph
    def buildGraph(links:Dict, numNodes:int)-> np.array:
        graph:List[List] = []
        for node in nodeLetters: # S
            neighbours:List[str] = links[node] # The neighbours for each node (e.g., ['A','B','C'])
            indexesList = nodeLettersToIndexes(neighbours) # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
            graph.append(indexesList) # [[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0] ]
        graph.append([0]*numNodes) # add [0,0,0, ..., 0] for T
        return np.array(graph)

    # Convert the letters to 0s and 1s
    def nodeLettersToIndexes(neighbours:List[str]) -> List[int]:
        indexesList:List[int] = [0]*numNodes # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for neighbour in neighbours: # ['A','B','C']
            indexesList[indexes[neighbour]] = 1 # [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
        return indexesList
    
    return buildGraph(links,numNodes),indexesValues
    


def runMaxFlow(links:Dict)->List[List[str]]:
    '''
    Objective: run the Max Flow algorithm 

    Parameters:
        links: The Graph 
    
    Output:
        The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
    '''
    
    graph,indexesValues = mainBuildGraph(links)
    EdmondKarp_graph = Graph(graph)
    dict_ = EdmondKarp_graph.EdmondKarp(data=True)
    list_ = dict_['iteration']
    disjointPaths:List[List[str]] = []
    for path in list_:
        pathList = list(path.values())[0]
        temp:List = []
        for node in pathList:
            letter = indexesValues[node]
            temp.append(letter)
        disjointPaths.append(temp)

    # Return the disjoint paths without the two terminal nodes (i.e., S and T)
    def filterDisjointPaths(disjointPaths:List[List[str]])->List[List[str]]:
        disjointPathsNoTerminal:List[List[str]] = []
        for disjointPath in disjointPaths:
            if not (len(disjointPath) == 2):
                disjointPathsNoTerminal.append(disjointPath[1:len(disjointPath)-1])
        return disjointPathsNoTerminal        

    # Remove all but one of the paths that have similar nodes
    # The retained path is the longest
    def removeOverlapPaths(disjointPaths:List[List[str]])-> List[List[str]]:
        filteredList:List[List[str]] = filterDisjointPaths(disjointPaths)
        filteredList.sort(key=len)
        filteredList.reverse
        outputList:List[List[str]] = [filteredList[0]]
        for i in range(1,len(filteredList)):
            temp:List[str] = filteredList[i]
            flag:bool = True # Permission to Add
            for lst in outputList:

                if  list(set(temp) & set(lst)): # Not Empty
                    flag = False
                    break
            if flag:
                outputList.append(temp)
        return outputList



    # Return the final disjoint paths 
    def addTerminalsToDisjointPaths(outputList:List[List[str]]):
        disjointPathTerminals:List[List[str]] = []
        for disjointPath in outputList:
            disjointPath.insert(0,'S')
            disjointPath.insert(len(disjointPath),'T')
            disjointPathTerminals.append(disjointPath)
        return disjointPathTerminals
        
    disjointPaths = addTerminalsToDisjointPaths(removeOverlapPaths(disjointPaths))    
    return disjointPaths
        
#%%
def runSSSP(links:Dict)->Tuple[List[List[str]],List[str]]:
    '''
    Objective: Run the Single Source Shortest Path (SSSP) algorithm

    Parameters:
        links: The Graph
    
    Output:
        The Disjoint paths as a 2D array of letters (e.g., [S,A,T],[S,C,D,T])
        The node letter values (i.e., [S,A,....,T])
    '''
    graph, indexesValues = mainBuildGraph(links)
    disjointPaths:List[List[str]] = []
    while True:
        g = sssp.GraphSSSP(len(graph),graph)
        try:
            g.dijkstra(0)
        except:
            break
        sourceNode:int = 0
        sinkNode:int = len(graph) - 1

        targetNode:int = sinkNode
        relayNodes:List[int] = []
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
    
    return disjointPaths,indexesValues


# Print the disjoint paths
def printPaths(disjointPaths:List[List[str]],indexesValues:List[str] = None,algorithm = 'Max Flow'):
    '''
    Objective: Print the disjoint paths

    Parameters:
        disjointPaths: The disjoint paths
        indexesValues: The node letter values (i.e., [S,A,....,T])
        Algorithm: Max Flow or SSSP
    
    Output: 
        None
    '''
    if algorithm == 'Max Flow':
        for path in disjointPaths:
            for node in path:
                if not node == 'T':
                    print(node,end='->')
                else:
                    print(node)
    else:
        for path in disjointPaths:
            path.reverse()
            for node in path:
                letter:str = indexesValues[node]
                if not letter == 'T':
                    print(letter, end='->')
                else:
                    print(letter)



#%%
links:Dict = { 
    'S':['A','B','C','D','E'],
    'A':['B','C','D','E','T'],
    'B':['C','D','E','T'],
    'C':['D','E','T'],
    'D':['E','T'],
    'E':['T']}
'''
    ,
    'F':['G','M'],
    'G':['J','M'],
    'H':['I','L'],
    'I':['L','N'],
    'J':['K','M','T'],
    'K':['L','T'],
    'L':['N'],
    'M':['T'],
    'N':['T'],
    }
'''
disjointPaths = runMaxFlow(links)
printPaths(disjointPaths)

#%%
disjointPaths, indexesValues = runSSSP(links)
printPaths(disjointPaths,indexesValues,'SSSP')
# %%
