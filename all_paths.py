import argparse
import time
import pandas as pd

def generate_paths(graph, start_node, end_node, path=[], prob=1.0):
    '''
    This function generates all possible paths between two nodes in a graph where each node can have multiple locations
    Args:
        graph (dict): The graph represented as a dictionary of nodes and their locations and probabilities
        start_node (str): The starting node
        end_node (str): The ending node
        path (list): The current path being explored
        prob (float): The probability of the current path
    Returns:
        paths (list): A list of tuples containing the path and the probability of the path
    '''
    if start_node == end_node:
        return [(path + [end_node], prob)]
    paths = []
    for location, location_prob in graph[start_node].items():
        if location not in path:
            new_path = path + [location]
            new_prob = prob * location_prob
            paths += generate_paths(graph, location, end_node, new_path, new_prob)
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all possible paths between two nodes in a graph')
    parser.add_argument('start_node', type=str, help='The starting node')
    parser.add_argument('end_node', type=str, help='The ending node')
    args = parser.parse_args()

    print('Generating paths...')
    time_start = time.time()
    graph = {'S': {'S1': 0.4, 'S2': 0.6}, 'T': {'T1': 1.0}}
    start_node = args.start_node
    end_node = args.end_node
    paths = generate_paths(graph, start_node, end_node)
    df = pd.DataFrame(paths, columns=['path', 'probability'])
    time_end = time.time()
    elapsed_time = time_end - time_start
    
    print(f'Elapsed time: {elapsed_time}')
    print(df)