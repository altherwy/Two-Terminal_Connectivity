import argparse
from math import sin, cos,exp, pi, inf, sqrt, floor
from jaal import Jaal
import random as rand
#import plotly.express as px
import pandas as pd
import multiprocessing as mp
import psutil
import time
from datetime import datetime
import json


class PhysicalModel:
    '''
    Simulates a physical model of a sensor network
    '''
    def __init__(self, days=1, loc_set_max=10, number_of_nodes=10, conn_level=2):
        self.k1 = 0.3*pi
        self.k2 = pi
        self.k3 = 2*pi
        self.k4 = 1
        self.k5 = 1
        self.lambda_ = 1
        self.v = 0.3
        self.Vs = 0.5
        self.C = 10
        self.Ac = 10
        self.alph = 10
        self.K = 10
        self.m = 1
        self.Ar = 1
        self.P0 = 1025
        self.B = 0.02
        self.g = 9.8
        self.ac = self.C * self.Ac * self.alph
        self.days = days
        self.days_in_seconds = 86400 * self.days
        self.loc_set_max = loc_set_max
        self.interval = self.days_in_seconds//self.loc_set_max # interval between two node's positions
        self.number_of_nodes = number_of_nodes
        cpus = psutil.cpu_count(logical=False) # only physical cores
        self.number_of_cores = cpus - 1 if cpus > 1 else 1  # no logical cores
        print('Number of cores: ', self.number_of_cores)
        self.node_positions = pd.DataFrame(index=range(number_of_nodes) ,columns=range(loc_set_max)) # dataframe to store the locations of the nodes at different times
        self.node_initial_positions = list(self._generate_random_positions()) # generate random positions for the nodes
        self._reorder_node_positions() # reorder the nodes so that the two furtherest nodes are at the first and last positions of the node_initial_positions list
        self.node_positions_filtered = pd.DataFrame(columns=['x','y','z','node_id','pos_prob'])

        self.links = {}
        self.loc_links = {}
        self.nodes = []
        self.loc = {}
        
        self.file_name = None

        self.conn_level = conn_level # medium connectivity level

    def _generate_random_positions(self, xspan=[-1*10**3, 1*10**3], yspan=[-0.5*10**3, 0.5*10**3], zspan=[-500, -100], ps_span=[1025, 1045]):
        '''
        Generates random positions for the nodes

        '''
        counter = 0
        while counter < self.number_of_nodes:
            x = rand.randint(xspan[0],xspan[1]) # x span for the sensor deployment
            y = rand.randint(yspan[0],yspan[1]) # y span for the sensor deployment
            z = rand.randint(zspan[0],zspan[1]) # depth of the sensor deployment
            ps = rand.randint(ps_span[0],ps_span[1]) # density of the sensor
            counter += 1
            yield (x,y,z,ps)   

    def _reorder_node_positions(self):
        '''
        Place the two furtherest nodes from each other at the first and last positions of the node_initial_positions list
        '''
        max_distance = 0
        fst_node_id = 0
        lst_node_id = 0
        for i in range(len(self.node_initial_positions)-1):
            for j in range(i+1,len(self.node_initial_positions)):

                coordinate_1 = self.node_initial_positions[i][0:3]
                coordinate_2 = self.node_initial_positions[j][0:3]
                distance = self._get_distance(coordinate_1, coordinate_2)
                if distance > max_distance:
                    max_distance = distance
                    fst_node_id = i
                    lst_node_id = j

        
        fst_node = self.node_initial_positions.pop(fst_node_id)
        self.node_initial_positions.insert(0,fst_node)
        lst_node = self.node_initial_positions.pop(lst_node_id)
        self.node_initial_positions.append(lst_node)



    def simulate(self):
        for pos in self.node_initial_positions:
            node_id = self.node_initial_positions.index(pos)
            self._simulate(node_id,pos)


    def _simulate(self, node_id, node_pos):
        '''
        Simulates the physical model of the sensor network
        '''
        x,y,z,ps = node_pos
        x_velocity = 0
        y_velocity = 0
        z_velocity = 0
        Cxy = self.ac/(ps*self.Vs)

        counter = 0

        for t in range(0,self.days_in_seconds,self.interval):
            pw = self.P0 + (self.B*z) # water density
            ar = self.K*pw*self.m*self.Ar
            Cz = ar/(ps*self.Vs)

            vx = (self.k1*self.lambda_*self.v*sin(self.k2*x)*cos(self.k3*y))+(self.k1*self.lambda_*cos(2*self.k1*t))+(self.k4)
            vy = (-self.lambda_*self.v*cos(self.k2*x)*sin(self.k3*y))+(self.k5)
            vz = (ps-pw)*self.g/(ps*Cz)

            x_velocity = (vx) + (x_velocity - vx)*exp(-Cxy)
            y_velocity = (vy) + (y_velocity - vy)*exp(-Cxy)
            z_velocity = (vz) + (z_velocity - vz)*exp(-Cz)

            x = x + (vx) + ((x_velocity - vx)/(Cxy))*(1 - exp(-Cxy))
            y = y + (vy) + ((y_velocity - vy)/(Cxy))*(1 - exp(-Cxy))
            z = z + (vz) + ((z_velocity - vz)/(Cz))*(1 - exp(-Cz))
            self.node_positions.iat[node_id,counter] = (x,y,z)
            counter += 1
    def is_reachable(self, coordinate_1, coordinate_2, dis_threshold):
        '''
        Checks if two coordinates are reachable by a sensor
        '''
        distance = self._get_distance(coordinate_1, coordinate_2)
        if distance > dis_threshold:
            return False
        else:
            return True

    def _get_distance(self, coordinate_1, coordinate_2):
        '''
        Calculates the Euclidean distance between two coordinates
        '''
        x1, y1, z1 = coordinate_1
        x2, y2, z2 = coordinate_2
        distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
        return distance
    
    def _get_dis_threshold(self):
        '''
        Calculates the average distance between consecutive positions for each node in the simulation
        Args:
            None
        Returns:
            None
        '''
        total_avg = 0
        for j in range(len(self.node_positions)):
            df = self.node_positions.iloc[j]
            avg = 0
            for i in range(len(df)-1):      
                avg += self._get_distance(df[i], df[i+1])
                
                
            
            total_avg += avg / (len(df)-1)
        return total_avg/len(self.node_positions)
            
    def build_location_probs(self, dis_threshold, min_prob , max_prob)->None:
        '''
        Builds the location probabilities for the nodes
        Args:
            dis_threshold: the distance threshold between two positions to be considered as the same node
            min_prob: the minimum probability of a node being at a specific position
            max_prob: the maximum probability of a node being at a specific position
        Returns:
            None

        '''
        counter = 0
        for node_id in range(self.number_of_nodes):
            #pos_prob = round(1/self.loc_set_max,2) # probability of the node being at a specific position
            pos_prob = round(rand.randint(min_prob,max_prob)/100,2)
            pos_prob_counter = pos_prob
            x,y,z = zip(*self.node_positions.loc[node_id])
            add_flag = False
            temp_counter = 0
            for x_pos,y_pos,z_pos in zip(x,y,z): # type: ignore
                if temp_counter == 0:
                    add_flag = True
                else:
                    last_row = self.node_positions_filtered.iloc[-1] # get the last row
                    coordinate_1 = last_row[['x', 'y', 'z']].values.tolist() # extract the coordinates as a list
                    coordinate_2 = [x_pos,y_pos,z_pos]
                    add_flag = not self.is_reachable(coordinate_1,coordinate_2,dis_threshold)

                # if the node is reachable and the random number is not equal to the counter (for inactivity purposes)
                # we can safely delete the last part of the condition to discard inactivity
                if add_flag: #and rand.randint(0,self.loc_set_max) != counter: 
                    self.node_positions_filtered.loc[counter] = [x_pos,y_pos,z_pos,node_id,pos_prob] # type: ignore
                    counter += 1
                    temp_counter += 1 
                    #pos_prob = round(1/self.loc_set_max,2) # reset the probability
                    pos_prob = round(rand.randint(min_prob,max_prob)/100,2)
                    pos_prob_counter += pos_prob
                else:
                    #pos_prob += round(1/self.loc_set_max,2) # increase the probability
                    pos_prob += round(rand.randint(0, int(pos_prob*100))/100)
                    pos_prob_counter += pos_prob
                if pos_prob_counter >= 1 or temp_counter == self.loc_set_max-1: 
                    # sum the probabilities of the same node
                    pos_prob = round(1- self.node_positions_filtered[self.node_positions_filtered['node_id'] == node_id]['pos_prob'].sum(),2)
                    self.node_positions_filtered.loc[counter] = [x_pos,y_pos,z_pos,node_id,pos_prob] # type: ignore
                    counter += 1
                    
                    break
    
    def _get_coordinates(self,node_id:int,time:int):
        '''
        Returns the coordinates of a node at a given time
        Args:
            node_id: the id of the node
            time: the time at which the coordinates are to be returned
        Returns:
            coordinates: the coordinates of the node at the given time
        '''
        df = self.node_positions_filtered[self.node_positions_filtered['node_id'] == node_id]
        if time >= len(df):
            time = len(df)-1
        coordinates = list(df.iloc[time][['x','y','z','node_id']])
        return coordinates

  
    
    def build_loc(self):
        '''
        Builds the locality set for each node as a dictionary
        Args:
            None
        Returns:
            None
        '''
        for node_id in range(self.number_of_nodes):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            self.loc[node_id] = df['pos_prob'].values.tolist()
    
    def get_avg_distance(self):
        '''
        Returns the average distance between any two nodes
        Args:
            None
        Returns:
            avg_distance: the average distance between any two nodes
        '''
        avg_distance = 0
        for node_id in range(self.number_of_nodes-1):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id+1]
            coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
            distance = self._get_distance(coordinate_1,coordinate_2)
            avg_distance += distance
        return round(avg_distance/(self.number_of_nodes-1),0)
    
    def get_standard_deviation_distance(self):
        '''
        Returns the standard deviation of the distance between the nodes
        Args:
            None
        Returns:
            std_distance: the standard deviation of the distance between the nodes
        '''
        avg_distance = self.get_avg_distance()
        std_distance = 0
        for node_id in range(self.number_of_nodes-1):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id+1]
            coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
            distance = self._get_distance(coordinate_1,coordinate_2)
            std_distance += (distance-avg_distance)**2
        return round(sqrt(std_distance/(self.number_of_nodes-1)),0) 

    def build_underlying_graph(self,dis_threshold = None):
        '''
        Builds the connection between nodes based on the distance threshold
        Args:
            dis_threshold: the distance threshold
        Returns:
            None
        '''
        
        if dis_threshold == None:
            dis_threshold = self.get_avg_distance() + self.get_standard_deviation_distance()
        for node_id in range(self.number_of_nodes-1):
            neighbor_list = [] # the list of neighbors of node i
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
            for node_id_2 in range(node_id+1,self.number_of_nodes):
                df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id_2]
                coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
                distance = self._get_distance(coordinate_1,coordinate_2)
                if distance < dis_threshold:
                    neighbor_list.append(node_id_2)
            self.links[node_id] = neighbor_list # type: ignore
            
    
    def build_loc_links(self):
        '''
        Builds the connection between nodes based on the distance threshold
        Args:
            None
        Returns:
            None
        '''
        keys = list(self.links.keys()) # type: ignore
        for key in keys:
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == key] # get the position of node i
            for neighbor in self.links[key]: # type: ignore
                df2 = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == neighbor] # get the position of neighbor node to node i
                a,b = self._build_connection(df,df2)
                self.loc_links[a] = b
    
    def build_loc_links_mp(self,key):
        conn_res = {}
        df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == key]
        for neighbor in self.links[key]: # type: ignore
            df2 = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == neighbor]
            a,b = self._build_connection(df,df2)
            conn_res[a] = b

        
        return conn_res
        
    
    def _build_connection(self,node_1:pd.DataFrame,node_2:pd.DataFrame, dis_threshold=None):
        node_id_1 = node_1.iloc[0]['node_id']
        node_id_2 = node_2.iloc[0]['node_id']
        conn_dict = {}
        if dis_threshold == None:
            dis_threshold = self.get_avg_distance() + self.get_standard_deviation_distance()

        for i in range(len(node_1)):
            coordinate_1 = node_1.iloc[i][['x','y','z']].values.tolist()
            conn_list = []
            for j in range(len(node_2)):
                coordinate_2 = node_2.iloc[j][['x','y','z']].values.tolist()
                if self.is_reachable(coordinate_1,coordinate_2,dis_threshold) and self._random_loc_set(self.conn_level):
                    conn_list.append(1) # 1 means connected
                else:
                    conn_list.append(0) # 0 means not connected
            conn_dict[i] = conn_list
        return (node_id_1, node_id_2),conn_dict
    
    def _random_loc_set(self, random_threshold=2):
        num = rand.randint(0,random_threshold)
        if num < random_threshold:
            return True
        return False
    
    def plot_underlying_graph(self,links):
        '''
        Plots the JAAL graph
        Args:
            links: the links between nodes
        Returns:
            None
        '''
        from_list = []
        to_list = []
        for key in links.keys():
            for to in links[key]:
                from_list.append(key)
                to_list.append(to)

        df_node = pd.DataFrame({'id':[i for i in range(self.number_of_nodes)]})
        df_edge = pd.DataFrame({'from':from_list,'to':to_list})
        Jaal(df_edge,df_node).plot()
    
    def name_nodes(self):
        '''
        Changes the node_id of nodes to alphabetic characters
        Args:
            loc: the location of nodes
            links: the links between nodes
            loc_links: the links between nodes in the same location
        Returns:
            loc_name: the location of nodes with alphabetic characters
            links_name: the links between nodes with alphabetic characters
            loc_links_name: the links between nodes in the same location with alphabetic characters
        '''
        
        
        links_name = {}
        for key in self.links.keys(): # type: ignore
            if key == 0:
                temp_list = []
                for i in self.links[key]: # type: ignore
                    temp_list.append(str(i)) if i != self.number_of_nodes - 1 else temp_list.append('T')
                links_name['S'] = temp_list

            else:
                temp_list = []
                for i in self.links[key]: # type: ignore
                    temp_list.append(str(i)) if i != self.number_of_nodes - 1 else temp_list.append('T')
                links_name[str(key)] = temp_list
    
        loc_links_name = {}    
        for key in self.loc_links.keys():
            if key[0] == 0:
                from_node = 'S'
            else:
                from_node = str(int(key[0]))
            
            if key[1] == self.number_of_nodes - 1:
                to_node = 'T'
            else:
                to_node = str(int(key[1]))

            loc_links_name[from_node,to_node] = self.loc_links[key]
        
        loc_name = {}
        for key in self.loc.keys():
            if key == 0:
                loc_name['S'] = self.loc[key]
            elif key == self.number_of_nodes - 1:
                loc_name['T'] = self.loc[key]
            else:
                loc_name[str(key)] = self.loc[key]
        df_loc_links = pd.DataFrame.from_dict(loc_links_name)
        return loc_name, links_name, df_loc_links
    
    def get_data(self):
        '''
        Gets the locality set of nodes, the links between the nodes, and the links between the nodes' locality set
        Args:
            None
        Returns:
            loc: the locality set of nodes
            links: the links between the nodes
            loc_links: the links between the nodes' locality set
            nodes: the nodes
        '''
        dict_min_prob = {
            1:60,
            2:50,
            3:20,
            4:10,
            5:10,
            6:0
        }

        dict_max_prob = {
            1:100,
            2:80,
            3:70,
            4:40,
            5:20,
            6:20
        }
        if self.loc_set_max in dict_min_prob.keys():
            min_prob = dict_min_prob[self.loc_set_max]
            max_prob = dict_max_prob[self.loc_set_max]
        else:
            min_prob = 0
            max_prob = 20

        self.simulate()
        dis_threshold = self._get_dis_threshold()
        self.build_location_probs(dis_threshold,min_prob=min_prob, max_prob=max_prob) # get the proabalities of being at each location for each node
        self.build_loc()
        self.build_underlying_graph()

        start_time = time.time()
        print('Building the links between nodes in the same location...')
        # start the multiprocessing pool
        pool = mp.Pool(self.number_of_cores)
        keys = list(self.links.keys())
        results = pool.map(self.build_loc_links_mp, keys)
        
        # results to self.loc_links
        for result in results:
            for key in result.keys():
                self.loc_links[key] = result[key]
        print("--- build_loc_links method runs in %s minutes ---" % ((time.time() - start_time)/60))

        
        loc_name,links_name,loc_links_name = self.name_nodes()
        self.nodes = list(loc_name.keys())
        

        return loc_name,links_name,loc_links_name

            
    def output(self, loc_name, links_name, loc_links_name):
        dt = datetime.now()
        ts = datetime.strftime(dt,'%Y%m%d%H%M%S')
        with open(r'node_data/%s.txt'%ts,'w') as f:
            for node in self.nodes:
                f.write(node + '\n')
        f.close()

        with open(r'links_data/%s.txt'%ts,'w') as f:
            json.dump(links_name,f)
        f.close()

        with open(r'loc_data/%s.txt'%ts,'w') as f:
            json.dump(loc_name,f)
        f.close()
        
        loc_links_dict = loc_links_name.to_dict()
        my_dict_str = {str(k): v for k, v in loc_links_dict.items()}
        with open(r'loc_links_data/%s.json'%ts,'w') as f:
            json.dump(my_dict_str,f)
        f.close()

        return ts
    
    def print_data(self, loc_name, links_name, loc_links_name):
        print('loc_name: ', loc_name)
        print('links_name: ', links_name)
        print('loc_links_name: ', loc_links_name)
        print('nodes: ', self.nodes)
        

    
    
    def main(self, print_flag=False):
        loc_name, links_name, loc_links_name = self.get_data()
        print('------------------ Done ------------------')
        self.file_name = self.output(loc_name, links_name, loc_links_name)
        
    

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("-n","--nodes")
    parser.add_argument("-l","--locality")
    parser.add_argument("-p","--print",action='store_true')
    parser.add_argument("-cl","--connection_level")

    args = parser.parse_args()
   
    if args.nodes and args.locality:
        start_time = time.time()
        sim = PhysicalModel(number_of_nodes=int(args.nodes), loc_set_max=int(args.locality))
        if args.connection_level:
            sim.conn_level = int(args.connection_level)
        if args.print:
            sim.main(print_flag=True)
        else:
            sim.main()
        print("--- total running time  %s minutes ---" % (round((time.time() - start_time)/60,2)))
    else:
        print('Please enter the number of nodes and the locality set max')
        print('Example: python physical_model_simulation.py -n 10 -l 3')
        exit()
