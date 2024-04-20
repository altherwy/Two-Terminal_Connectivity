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
 -  Generate paths as DataFrame for all DPaths (&#9989;)
 -  Check the connectivity of each DPath (&#9989;)
 # Experiments
 -  MSCW vs. 2Nodes connectivity (Category: Effect of increasing node transmission radius)  (&#9874;)
    
    | V  | L | Conn Level | Running Time (sec) MSCW | Running Time (sec) (2Nodes connectivity) |
    | -- | - | ---------- | ----------------------- | ---------------------------------------- |
    | 36 | 2 | 1          | 83.546                  | N/A                                      |
    | 36 | 2 | 2          | 440.188                 | N/A                                      |
    | 36 | 2 | 3          | 6753.187                | N/A                                      |
    
- A-Conn vs. 2Nodes connectivity (&#9989;)
    |Algorithm| Number of States | Transmission Range | Connectivity Difference |
    | ------- | ---------------- | ------------------ | ----------------------- |
    | A-Conn  | ~ 1.7 M          | 5 units            | ~ 38 %                  |
    | 2Nodes  | ~ 4.7 M          | 66%                | ~ 25 %                  |
    | 2Nodes  | ~ 1.6 M          | 66%                | ~ 25 %                  |

- P-COMP vs. 2Nodes connectivity (varying V) &#9874;
    | V  | L | K | Connectivity Average (P-Comp) | Connectivity Average (2Nodes) |
    | -- | - | - | ----------------------------- | ----------------------------- |
    | 9  | 2 | 9 | .345                          | N/A                           |
    | 10 | 2 | 9 | .4025                         | N/A                           |
    | 11 | 2 | 9 | .4475                         | N/A                           |
    | 12 | 2 | 9 | .4825                         | N/A                           |
    | 13 | 2 | 9 | .5275                         | N/A                           |
    | 14 | 2 | 9 | .58                           | N/A                           |
    | 15 | 2 | 9 | .6475                         | N/A                           |

- P-COMP vs. 2Nodes connectivity (varying Loc max) &#9874;
    | V | L | K | Connectivity Average (P-Comp) | Connectivity Average (2Nodes) |
    | - | - | - | ----------------------------- | ----------------------------- |
    | 8 | 2 | 9 | .83                           | N/A                           |
    | 8 | 3 | 9 | .63                           | N/A                           |
    | 8 | 4 | 9 | .36                           | N/A                           |
    | 8 | 5 | 9 | .14                           | N/A                           |
    | 8 | 6 | 9 | .16                           | N/A                           |
    
   

# Misc
- MSCW: Multistate Component Weight (MSCW) problem
 > "On Connected Components in Multistate Wireless Sensor Network Probabilistic Models"

they have 36 nodes, 2 and 3 loc_max, and 3 different connection levels. 
- A-Conn: 
> "Tree bound on probabilistic connectivity of Underwater Sensor Networks" 

they have 20 nodes max, 8 loc_max , and connevetivty levels range from 3.5 to 6.5

- P-COMP:
> "On_Probabilistic_Connected_Components_in_Underwater_Sensor_Networks"

They vary the node numbers to 15 and loc max to 6 

