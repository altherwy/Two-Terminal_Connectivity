# Two-Terminal_Connectivity
The Two-Terminal connectivity algorithm computes the connectivity between two nodes (terminals) where the node locations are probabilistic
# Example graphs 
## [Graph 1](pics/graph1.png)

    # links between nodes
    links:Dict  = {'S':['A','E'],'A':['B'],'B':
                    ['C','F'],'C':['D'],'D':['T'],'E':['F'],'F':['T']}
    
    # locality set of each node
    loc:Dict  = {'A':[.15,.25,.3,.3], 'B':[.4,.2,.4], 
                'C':[.2,.3,.4,.1], 'D':[.4,.3,.3],'E':[.5,.5],
                'F':[.4,.6],'S':[.3,.5,.2], 'T':[.8,.2]}

	# the links between the locality sets of nodes
    loc_links  =  pd.DataFrame({('A','B'): {0:[1,1,1],1:[1,1,1], 2:[1,1,1], 3:[1,1,1]},
                            ('B','C'): {0:[1,1,1,1],1:[1,1,1,1],2:[1,1,1,1]},
                            ('C','D'): {0:[0,1,1], 1:[1,1,1], 2:[1,1,1],3:[1,1,1]},
                            ('E','F'): {0:[1,1],1:[1,1]},('S','A'): {0:[1,1,1,1],1:[1,1,1,1], 2:[1,1,1,1]},
                            ('D','T'): {0:[1,1], 1:[0,1],2:[1,1]},
                            ('S','E'): {0:[0,0],1:[0,0],2:[0,0]},
                            ('F','T'): {0:[0,0],1:[0,0]}})
    

## [Graph 2](pics/graph2.png)

# links between nodes
    links:Dict  = {'S':['A','D'],'A':['B'],
                'B':['C'],'C':['T'],'D':['E'],'E':['T']}
    
    # locality set of each node
    loc:Dict  = {'A':[.2,.8], 'B':[.3,.2,.5], 'C':[.6,.4], 
                'D':[.5,.5], 'E':[1],
                'S':[.35,.65], 'T':[.1,.1,.8]}

	# the links between the locality sets of nodes
    loc_links  =  pd.DataFrame({('A','B'): {0:[1,1,0], 1:[1,1,1]},
                            ('B','C'): {0:[1,0],1:[1,1], 2:[0,1]},
                            ('S','A'): {0:[1,1], 1:[0,1]},
                            ('C','T'):{0:[0,1,1],1:[0,0,1]},
                            ('D','E'): {0:[1],1:[1]},
                            ('S','D'): {0:[1,1], 1:[1,1]},
                            ('E','T'): {0:[1,1,1]},
                            })
                    

# To DO
 -  Generate DPaths using loc, links, nodes .. files (&#9989;)
 -  Generate paths as DataFrame for all DPaths (&#9874;)
 -  Check the connectivity of each DPath (&#9874;)
 -  Run the TwoTerminal algorithm (&#9874;)
    - Epoch2020: Nodes = 36, Loc_max = 2, Conn_level = 1
    - Epoch2020: Nodes = 36, Loc_max = 2, Conn_level = 2
    - Epoch2020: Nodes = 36, Loc_max = 2, Conn_level = 3
    - Epoch2020: Nodes = 36, Loc_max = 3, Conn_level = 1
    - Epoch2020: Nodes = 36, Loc_max = 3, Conn_level = 2
    - Epoch2020: Nodes = 36, Loc_max = 3, Conn_level = 3 
##
- Epcoh2020: Ehab 2020 paper, they have 36 nodes, 2 and 3 loc_max, and 3 different connection levels
- Epoch2014: Islam 2013 papaer, they have 20 nodes max, 8 loc_max , and connevetivty levels range from 3.5 to 6.5
- Epoch2018: need the paper (&#9874;)