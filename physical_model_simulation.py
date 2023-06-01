from math import sin, cos,exp, pi, inf
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
        self.C = 100
        self.Ac = 1
        self.alph = 1
        self.K = 100
        self.m = 1
        self.Ar = 1
        self.P0 = 1025
        self.B = 0.02
        self.g = 9.8
        self.ac = self.C * self.Ac * self.alph
        self.days = days
        self.days_in_seconds = 86400 * self.days
        self.loc_set_max = loc_set_max
        interval = self.days_in_seconds//self.loc_set_max # interval between two node's positions
        self.number_of_nodes = number_of_nodes

        self.node_positions = pd.DataFrame(index=range(number_of_nodes) ,columns=range(loc_set_max)) # dataframe to store the locations of the nodes at different times
        node_initial_positions = list(self._generate_random_positions()) # generate random positions for the nodes

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
        '''
        Simulates the physical model of the sensor network
        '''
        for i in range(self.days_in_seconds):
            for node_id in range(self.number_of_nodes):
                node_positions_filtered = self.node_positions[self.node_positions['node_id'] == node_id]
                if len(node_positions_filtered) == 0:
                    continue
                x_pos, y_pos, z_pos, node_id, pos_prob = node_positions_filtered.iloc[-1]
                temp_counter = 0
                add_flag = False
                while temp_counter < self.loc_set_max:
                    x_pos_new = x_pos + self.v * self.days * sin(self.k1 * x_pos + self.k2 * y_pos + self.k3 * z_pos + self.k4 * i + self.k5)
                    y_pos_new = y_pos + self.v * self.days * cos(self.k1 * x_pos + self.k2 * y_pos + self.k3 * z_pos + self.k4 * i + self.k5)
                    z_pos_new = z_pos + self.v * self.days * sin(self.k1 * x_pos + self.k2 * y_pos + self.k3 * z_pos + self.k4 * i + self.k5)
                    if z_pos_new > 0:
                        break
                    if temp_counter == 0:
                        add_flag = True
                    else:
                        last_row = node_positions_filtered.iloc[-1]
                        coordinate_1 = last_row[['x', 'y', 'z']].values.tolist()
                        coordinate_2 = [x_pos_new, y_pos_new, z_pos_new]
                        add_flag = not self.is_reachable(coordinate_1, coordinate_2, dis_threshold=100)
                    if add_flag:
                        self.node_positions.loc[len(self.node_positions)] = [x_pos_new, y_pos_new, z_pos_new, node_id, pos_prob]
                        temp_counter += 1
                        pos_prob = 1/self.loc_set_max
                    else:
                        pos_prob += 1/self.loc_set_max

    def is_reachable(self, coordinate_1, coordinate_2, dis_threshold):
        '''
        Checks if two coordinates are reachable by a sensor
        '''
        distance = self.get_distance(coordinate_1, coordinate_2)
        if distance > dis_threshold:
            return False
        else:
            return True

    def get_distance(self, coordinate_1, coordinate_2):
        '''
        Calculates the Euclidean distance between two coordinates
        '''
        x1, y1, z1 = coordinate_1
        x2, y2, z2 = coordinate_2
        distance = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
        return distance