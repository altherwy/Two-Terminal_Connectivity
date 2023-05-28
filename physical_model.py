#%%
from math import pi
from math import sin, cos,exp
import random as rand
import plotly.express as px
import pandas as pd

# %% generate random positions for the nodes
def _generate_random_positions():
    '''
    Generates random positions for the nodes
    '''
    counter = 0
    while counter < number_of_nodes:
        x = rand.randint(-500, 1*10**3)
        y = rand.randint(-500, 0.5*10**3)
        z = rand.randint(-500, -100)
        ps = rand.randint(1025, 1045) # density of the sensor
        counter += 1
        yield (x,y,z,ps)

# %% parameters
k1 = 0.3*pi
k2 = pi
k3 = 2*pi
k4 = 1
k5 = 1
lambda_ = 1
v = 0.3


Vs = 0.5 # volume of the sensor

C =100 # Constant
Ac = 1 # sensor cross section area facing the current
alph = 1# shape factor
K = 100 # constant
m = 1 # shape factor
Ar = 1 # sensor cross section area prependicular to the current
P0 = 1025
B = 0.02
g = 9.8

ac = C*Ac*alph

days = 1 # number of days
days_in_seconds = 86400 * days # Day(s) in seconds
loc_set_max = 6 # maximum number of locality sets
number_of_nodes = 10 # number of nodes
interval = days_in_seconds//loc_set_max # interval between two locality sets
node_positions = pd.DataFrame(index=range(number_of_nodes) ,columns=range(loc_set_max)) # dataframe to store the locations of the nodes at different times
node_positions_filtered = pd.DataFrame(columns=['x','y','z','node_id'])
node_initial_positions = list(_generate_random_positions()) # generator to generate random positions for the nodes

# %% find if two nodes are connected (or the same node has two locality sets)
is_reachable = lambda p1, p2, threshold: distance(p1, p2) <= threshold

# %% find the distance between two points in 3D space
from math import sqrt
def distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

# %% get the locations of the nodes
def __get_locations(node_id:int,node_pos:tuple):
    x,y,z,ps = node_pos

    x_velocity = 0
    y_velocity = 0
    z_velocity = 0
    Cxy = ac/(ps*Vs)

    counter = 0

    for t in range(0,days_in_seconds,interval):
        pw = P0 + (B*z) # water density
        ar = K*pw*m*Ar
        Cz = ar/(ps*Vs)

        vx = (k1*lambda_*v*sin(k2*x)*cos(k3*y))+(k1*lambda_*cos(2*k1*t))+(k4)
        vy = (-lambda_*v*cos(k2*x)*sin(k3*y))+(k5)
        vz = (ps-pw)*g/(ps*Cz)

        x_velocity = (vx) + (x_velocity - vx)*exp(-Cxy)
        y_velocity = (vy) + (y_velocity - vy)*exp(-Cxy)
        z_velocity = (vz) + (z_velocity - vz)*exp(-Cz)

        x = x + (vx) + ((x_velocity - vx)/(Cxy))*(1 - exp(-Cxy))
        y = y + (vy) + ((y_velocity - vy)/(Cxy))*(1 - exp(-Cxy))
        z = z + (vz) + ((z_velocity - vz)/(Cz))*(1 - exp(-Cz))
        node_positions.iat[node_id,counter] = (x,y,z)
        counter += 1


# %% get the locations of the nodes
def get_node_locations():
    for pos in node_initial_positions:
        node_id = node_initial_positions.index(pos)
        __get_locations(node_id,pos)

get_node_locations()
# %%
def build_locality_set(dis_threshold:float = 0.83):
    '''
    Plots the positions of the nodes
    Args:
        dis_threshold: the distance threshold between two nodes
    Returns:
        None
    '''
    counter = 0
    for node_id in range(number_of_nodes):
        x,y,z = zip(*node_positions.loc[node_id])
        add_flag = False
        temp_counter = 0
        for x_pos,y_pos,z_pos in zip(x,y,z):
            if temp_counter == 0:
                add_flag = True
            else:
                coordinate_1 =list(node_positions.loc[node_id][temp_counter-1])
                coordinate_2 = [x_pos,y_pos,z_pos]
                add_flag = not is_reachable(coordinate_1,coordinate_2,dis_threshold)

            if add_flag:        
                node_positions_filtered.loc[counter] = [x_pos,y_pos,z_pos,node_id]    
                counter += 1
                temp_counter += 1
        
        
    #fig = px.scatter_3d(node_positions_filtered,x='x',y='y',z='z',color='node_id')
    #fig.show()


build_locality_set(1) # plot the positions of the nodes

# %%
def plot_position_at_time(_time:int):
    
    '''
    Plots the position of a node at a given time
    Args:
        _time: the time at which the position is to be plotted
    Returns:
        None
    '''
    df_temp = pd.DataFrame(columns=['x','y','z','node_id'])
    for i in range(number_of_nodes):
        x,y,z,node_id = list(node_positions_filtered.loc[i*loc_set_max + _time])
        df_temp.loc[i] = [x,y,z,node_id]
    fig = px.scatter_3d(df_temp,x='x',y='y',z='z',color='node_id')
    fig.show()

plot_position_at_time(1) # plot the position of the nodes at time 1     
# %%

node_positions_filtered
# %%
