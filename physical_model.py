#%%
from math import sin, cos,exp, pi, inf
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
        x = rand.randint(-1*10**3, 1*10**3)
        y = rand.randint(-0.5*10**3, 0.5*10**3) # type: ignore
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
loc_set_max = 10 # maximum number of locality sets
number_of_nodes = 10 # number of nodes
interval = days_in_seconds//loc_set_max # interval between two locality sets
node_positions = pd.DataFrame(index=range(number_of_nodes) ,columns=range(loc_set_max)) # dataframe to store the locations of the nodes at different times
node_initial_positions = list(_generate_random_positions()) # generator to generate random positions for the nodes

# %% find if two nodes are connected (or the same node has two locality sets)
is_reachable = lambda p1, p2, threshold: get_distance(p1, p2) <= threshold

# %% find the distance between two points in 3D space
from math import sqrt
def get_distance(p1, p2):
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
def simulate():
    for pos in node_initial_positions:
        node_id = node_initial_positions.index(pos)
        __get_locations(node_id,pos)

simulate()
# %%
node_positions_filtered = pd.DataFrame(columns=['x','y','z','node_id','pos_prob'])
def build_location_probs(dis_threshold:float = 0.83):
    '''
    Builds the location probabilities for the nodes
    Args:
        dis_threshold: the distance threshold between two positions to be considered as the same node
    Returns:
        None

    '''
    counter = 0
    for node_id in range(number_of_nodes):
        pos_prob = 1/loc_set_max # probability of the node being at a specfic position
        x,y,z = zip(*node_positions.loc[node_id])
        add_flag = False
        temp_counter = 0
        for x_pos,y_pos,z_pos in zip(x,y,z): # type: ignore
            if temp_counter == 0:
                add_flag = True
            else:


                #coordinate_1 =list(node_positions.loc[node_id][temp_counter-1])
                last_row = node_positions_filtered.iloc[-1] # get the last row
                coordinate_1 = last_row[['x', 'y', 'z']].values.tolist() # extract the coordinates as a list
                coordinate_2 = [x_pos,y_pos,z_pos]
                add_flag = not is_reachable(coordinate_1,coordinate_2,dis_threshold)

            if add_flag:        
                node_positions_filtered.loc[counter] = [x_pos,y_pos,z_pos,node_id,pos_prob]     # type: ignore
                counter += 1
                temp_counter += 1 
                pos_prob = 1/loc_set_max # reset the probability
            else:
                pos_prob += 1/loc_set_max # increase the probability
        
        
    #fig = px.scatter_3d(node_positions_filtered,x='x',y='y',z='z',color='node_id')
    #fig.show()


build_location_probs(1.4) # build the locality set with a distance threshold of 1

# %%
def _get_coordinates(node_id:int,time:int):
    '''
    Returns the coordinates of a node at a given time
    Args:
        node_id: the id of the node
        time: the time at which the coordinates are to be returned
    Returns:
        coordinates: the coordinates of the node at the given time
    '''
    df = node_positions_filtered[node_positions_filtered['node_id'] == node_id]
    if time >= len(df):
        time = len(df)-1
    coordinates = list(df.iloc[time][['x','y','z','node_id']])
    return coordinates

#_get_coordinates(9,5) # get the coordinates of node 0 at time 1

def plot_position_at_time(time:int):
    
    '''
    Plots the position of a node at a given time
    Args:
        _time: the time at which the position is to be plotted
    Returns:
        None
    '''
    df_temp = pd.DataFrame(columns=['x','y','z','node_id'])
    for i in range(number_of_nodes):
        x,y,z,node_id = _get_coordinates(i,time)
        df_temp.loc[i] = [x,y,z,node_id] # type: ignore
    fig = px.scatter_3d(df_temp,x='x',y='y',z='z',color='node_id')
    fig.show()

#plot_position_at_time(1) # plot the position of the nodes at time 1     
# %%
def build_loc():
    '''
    Builds the locality set for each node as a dictionary
    Args:
        None
    Returns:
        loc: the locality set for each node
    '''
    loc:dict = {}
    for node_id in range(number_of_nodes):
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        loc[node_id] = df['pos_prob'].values.tolist()
    return loc

loc = build_loc()


# %%
def get_max_distance():
    max_distance = 0
    for node_id in range(number_of_nodes-1):
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        coordinate_1 = df.iloc[0][['x','y','z']].values.tolist() # coordinates of node i
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id+1]
        coordinate_2 = df.iloc[0][['x','y','z']].values.tolist() # coordinates of node i+1
        distance = get_distance(coordinate_1,coordinate_2)
        if distance > max_distance:
            print(node_id)
            max_distance = distance
    return round(max_distance,0)

get_max_distance()
# %%
def get_min_distance():
    min_distance = inf
    for node_id in range(number_of_nodes-1):
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id+1]
        coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
        distance = get_distance(coordinate_1,coordinate_2)
        if distance < min_distance:
            print(node_id)
            min_distance = distance
    return round(min_distance,0)

get_min_distance()
# %%
def get_avg_distance():
    avg_distance = 0
    for node_id in range(number_of_nodes-1):
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id+1]
        coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
        distance = get_distance(coordinate_1,coordinate_2)
        avg_distance += distance
    return round(avg_distance/number_of_nodes,0)
get_avg_distance()
# %%
def get_standard_deviation_distance():
    avg_distance = get_avg_distance()
    std_distance = 0
    for node_id in range(number_of_nodes-1):
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id+1]
        coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
        distance = get_distance(coordinate_1,coordinate_2)
        std_distance += (distance-avg_distance)**2
    return round(sqrt(std_distance/number_of_nodes),0) 

get_standard_deviation_distance()
# %%
def build_underlying_graph(dis_threshold:int=400):
    '''
    Builds the connection between nodes based on the distance threshold
    Args:
        dis_threshold: the distance threshold
    Returns:
        None
    '''
    conn_list = {}
    for node_id in range(number_of_nodes-1):
        neighbor_list = [] # the list of neighbors of node i
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id]
        coordinate_1 = df.iloc[0][['x','y','z']].values.tolist()
        for node_id_2 in range(node_id+1,number_of_nodes):
            df = node_positions_filtered.loc[node_positions_filtered['node_id'] == node_id_2]
            coordinate_2 = df.iloc[0][['x','y','z']].values.tolist()
            distance = get_distance(coordinate_1,coordinate_2)
            if distance < dis_threshold:
                neighbor_list.append(node_id_2)
        conn_list[node_id] = neighbor_list
    return conn_list

build_underlying_graph(1000)
# %%
def build_loc_links():
    underlying_graph = build_underlying_graph(1000)
    keys = list(underlying_graph.keys())
    loc_links = {}
    for key in keys:
        df = node_positions_filtered.loc[node_positions_filtered['node_id'] == key] # get the position of node i
        for neighbor in underlying_graph[key]:
            df2 = node_positions_filtered.loc[node_positions_filtered['node_id'] == neighbor] # get the position of neighbor node to node i
            a,b = _build_connection(df,df2,610)
            print(a)
            print(b)
            loc_links[a] = b
    return loc_links
             
    
build_loc_links()        
# %%
def _build_connection(node_1:pd.DataFrame,node_2:pd.DataFrame, dis_threshold:int=400):
    node_id_1 = node_1.iloc[0]['node_id']
    node_id_2 = node_2.iloc[0]['node_id']
    conn_dict = {}
    for i in range(len(node_1)):
        coordinate_1 = node_1.iloc[i][['x','y','z']].values.tolist()
        conn_list = []
        for j in range(len(node_2)):
            coordinate_2 = node_2.iloc[j][['x','y','z']].values.tolist()
            if is_reachable(coordinate_1,coordinate_2,dis_threshold):
                conn_list.append(1) # 1 means connected
            else:
                conn_list.append(0) # 0 means not connected
        conn_dict[i] = conn_list
    return (node_id_1, node_id_2),conn_dict



# %%
from math import sin, cos, exp, pi, inf
import random as rand
import plotly.express as px
import pandas as pd

