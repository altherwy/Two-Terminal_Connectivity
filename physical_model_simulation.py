from math import sin, cos,exp, pi, inf, sqrt
from jaal import Jaal
import random as rand
import plotly.express as px
import pandas as pd


class PhysicalModel:
    '''
    Simulates a physical model of a sensor network
    '''
    def __init__(self, days=1, loc_set_max=10, number_of_nodes=10):
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

        self.node_positions = pd.DataFrame(index=range(number_of_nodes) ,columns=range(loc_set_max)) # dataframe to store the locations of the nodes at different times
        self.node_initial_positions = list(self._generate_random_positions()) # generate random positions for the nodes
        self.node_positions_filtered = pd.DataFrame(columns=['x','y','z','node_id','pos_prob'])

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
            sim: an instance of the PhysicalModel class
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
            
    def build_location_probs(self, dis_threshold)->None:
        '''
        Builds the location probabilities for the nodes
        Args:
            dis_threshold: the distance threshold between two positions to be considered as the same node
        Returns:
            None

        '''
        counter = 0
        for node_id in range(self.number_of_nodes):
            pos_prob = 1/self.loc_set_max # probability of the node being at a specific position
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

                if add_flag:        
                    self.node_positions_filtered.loc[counter] = [x_pos,y_pos,z_pos,node_id,pos_prob] # type: ignore
                    counter += 1
                    temp_counter += 1 
                    pos_prob = 1/self.loc_set_max # reset the probability
                else:
                    pos_prob += 1/self.loc_set_max # increase the probability
    
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

    def plot_position_at_time(self,time:int):
        
        '''
        Plots the position of a node at a given time
        Args:
            _time: the time at which the position is to be plotted
        Returns:
            None
        '''
        df_temp = pd.DataFrame(columns=['x','y','z','node_id'])
        for i in range(self.number_of_nodes):
            x,y,z,node_id = self._get_coordinates(i,time)
            df_temp.loc[i] = [x,y,z,node_id] # type: ignore
        fig = px.scatter_3d(df_temp,x='x',y='y',z='z',color='node_id')
        fig.show()
    
    def build_loc(self) -> dict[str, list[float]]:
        '''
        Builds the locality set for each node as a dictionary
        Args:
            None
        Returns:
            loc: the locality set for each node
        '''
        loc:dict = {}
        for node_id in range(self.number_of_nodes):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            loc[node_id] = df['pos_prob'].values.tolist()
        return loc
    def get_max_distance(self):
        '''
        Returns the maximum distance between any two nodes
        Args:
            None
        Returns:
            max_distance: the maximum distance between any two nodes
        '''
        max_distance = 0
        for node_id in range(self.number_of_nodes-1):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            coordinate_1 = df.iloc[0][['x','y','z']].values.tolist() # coordinates of node i
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id+1]
            coordinate_2 = df.iloc[0][['x','y','z']].values.tolist() # coordinates of node i+1
            distance = self._get_distance(coordinate_1,coordinate_2)
            if distance > max_distance:
                max_distance = distance
        return round(max_distance,0)
    
    def get_min_distance(self):
        '''
        Returns the minimum distance between any two nodes
        Args:
            None
        Returns:
            min_distance: the minimum distance between any two nodes
        '''
        min_distance = inf
        for node_id in range(self.number_of_nodes-1):
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id]
            coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == node_id+1]
            coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
            distance = self._get_distance(coordinate_1,coordinate_2)
            if distance < min_distance:
                
                min_distance = distance
        return round(min_distance,0)
    
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

    def build_underlying_graph(self,dis_threshold = None)->dict[str, list[str]]:
        '''
        Builds the connection between nodes based on the distance threshold
        Args:
            dis_threshold: the distance threshold
        Returns:
            None
        '''
        conn_list = {}
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
            conn_list[node_id] = neighbor_list
            
        return conn_list
    
    def build_loc_links(self,underlying_graph):
        '''
        Builds the connection between nodes based on the distance threshold
        Args:
            dis_threshold: the distance threshold
        Returns:
            None
        '''
        keys = list(underlying_graph.keys())
        loc_links = {}
        for key in keys:
            df = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == key] # get the position of node i
            for neighbor in underlying_graph[key]:
                df2 = self.node_positions_filtered.loc[self.node_positions_filtered['node_id'] == neighbor] # get the position of neighbor node to node i
                a,b = self._build_connection(df,df2)
                loc_links[a] = b
        return loc_links
    
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
                if self.is_reachable(coordinate_1,coordinate_2,dis_threshold):
                    conn_list.append(1) # 1 means connected
                else:
                    conn_list.append(0) # 0 means not connected
            conn_dict[i] = conn_list
        return (node_id_1, node_id_2),conn_dict
    
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
    
    def name_nodes(self, loc, links, loc_links):
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
        
        # get the last node_id
        first_node = chr(self.node_positions_filtered.iloc[0]['node_id'] + 65)
        last_node = chr(self.node_positions_filtered.iloc[-1]['node_id'] + 65)
        # change to character
        



            
        # chane the node_id of links to alphabetic characters
        links_name = {}
        for key in links.keys():
            links_name[chr(key+65)] = [chr(i+65) for i in links[key]]
        
        # replace the first node and last character to S and T, respectively
        links_name['S'] = links_name.pop('A')
        links_name['T'] = links_name.pop(chr(self.number_of_nodes-1+65))

        # change all the keys of loc_links to alphabetic characters
        loc_links_name = {}
        for key in loc_links.keys():
            from_node = chr(int(key[0])+65)
            to_node = chr(int(key[1])+65)
            loc_links_name[from_node,to_node] = loc_links[key]
        # replace the first node and last character of loc_links_name to S and T, respectively
        loc_links_name['S'] = loc_links_name.pop(('A',chr(self.number_of_nodes-1+65)))
        loc_links_name['T'] = loc_links_name.pop(('A',chr(self.number_of_nodes-1+65)))
        

       
        # change the node_id of loc to alphabetic characters
        loc_name = {}
        for key in loc.keys():
            loc_name[chr(key+65)] = loc[key]
        

        return loc_name, links_name, loc_links_name
    
    def get_data(self):
        '''
        Gets the locality set of nodes, the links between the nodes, and the links between the nodes' locality set
        Args:
            None
        Returns:
            loc: the locality set of nodes
            links: the links between the nodes
            loc_links: the links between the nodes' locality set
        '''
            
        self.simulate()
        dis_threshold = self._get_dis_threshold()
        self.build_location_probs(dis_threshold) # get the proabalities of being at each location for each node
        loc = self.build_loc()
        links = self.build_underlying_graph()
        loc_links = self.build_loc_links(links)
        loc_name,links_name,loc_links_name = self.name_nodes(loc,links,loc_links)

        return loc_name,links_name,loc_links_name

            

        

    def main(self):
        loc_name, links_name, loc_links_name = self.get_data()
        #self.plot_underlying_graph(links)
        
        print(loc_name)
        print('------------------')
        print(links_name)
        print('------------------')
        print(loc_links_name)

        

if __name__ == '__main__':
    sim = PhysicalModel(number_of_nodes=3, loc_set_max=3)
    sim.main()