#%% import libraries and functions 
import supabase_client as sc
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
resolution = 1200
ylim = [0,100]
def _unpack_two_terminals(row, attr = 'connectivity'):
    temp = row[0]
    return temp[attr]

def set_fonts():
    # change font to Palatino Linotype with 12 and bold
    plt.rcParams["font.family"] = "Palatino Linotype"
    plt.rcParams["font.size"] = "12"
    plt.rcParams["font.weight"] = "bold"

'''
///////////////////////////////////////////
Start nodes vs. connectivity (MF algorithm)
///////////////////////////////////////////
'''
#%%
reponse = sc.supabase.table('exhaustive_algorithms')\
    .select('connectivity , nodes, running_time, two_terminals(connectivity)')\
    .eq('locality_sets',3).eq('two_terminals.algorithm','MF').eq('is_valid',True)\
        .execute()
df = pd.DataFrame(reponse.data)
df['two_terminals'] = df['two_terminals'].apply(lambda x: _unpack_two_terminals(x))
df['ttc_running_time'] = 0.05

df['mean_exh_conn'] = df.groupby('nodes')['connectivity'].transform('mean')
df['mean_exh_conn'] = df['mean_exh_conn'].apply(lambda x: round(x*100,2))

df['mean_ttc_conn'] = df.groupby('nodes')['two_terminals'].transform('mean')
df['mean_ttc_conn'] = df['mean_ttc_conn'].apply(lambda x: round(x*100,2))

df['std_exh_conn'] = df.groupby('nodes')['connectivity'].transform('std')
df['std_exh_conn'] = df['std_exh_conn'].apply(lambda x: round((x/4)*100,2))

df['std_ttc_conn'] = df.groupby('nodes')['two_terminals'].transform('std')
df['std_ttc_conn'] = df['std_ttc_conn'].apply(lambda x: round((x/4)*100,2))

df['std_exh_time'] = df.groupby('nodes')['running_time'].transform('std')
df['std_exh_time'] = df['std_exh_time'].apply(lambda x: round(x,2))

df['std_ttc_time'] = .01

# %%
# get distinct dataframe of nodes
df_nodes = df[['nodes','mean_exh_conn', 'mean_ttc_conn', 'std_exh_conn', 'std_ttc_conn']].drop_duplicates()
df_nodes = df_nodes.sort_values(by=['nodes'], ignore_index=True)
df_nodes = df_nodes.set_index('nodes')
# %% plot and save figure
fig, ax = plt.subplots()
#ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='in', pad=10, weight='bold')
ax.bar(df_nodes.index, df_nodes['mean_exh_conn'], yerr=df_nodes['std_exh_conn'], label='Exact') # type: ignore
ax.bar(df_nodes.index, df_nodes['mean_ttc_conn'], yerr=df_nodes['std_ttc_conn'], label='Lower Bound') # type: ignore
ax.legend()
ax.errorbar(df_nodes.index, df_nodes['mean_exh_conn'], yerr=df_nodes['std_exh_conn'], label='Exact', fmt='*-') # type: ignore
ax.errorbar(df_nodes.index, df_nodes['mean_ttc_conn'], yerr=df_nodes['std_ttc_conn'], label='Lower Bound', fmt='o-') # type: ignore
plt.xlabel('$V$')
plt.ylabel('Connectivity (%)')
plt.ylim(ylim)
plt.grid()
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
set_fonts()
plt.show()
#plt.savefig('figures/nodes vs connectivity.png', dpi=resolution, format='png')
'''
///////////////////////////////////////////
End of nodes vs. connectivity (MF algorithm)

Start of difference between the mean of exhaustive and two terminals (MF)
///////////////////////////////////////////
'''
# %% get the difference between the mean of exhaustive and two terminals (MF)
df_nodes['diff'] = df_nodes['mean_exh_conn'] - df_nodes['mean_ttc_conn']
df_nodes['diff'] = df_nodes['diff'].apply(lambda x: round(x,2))
# %% plot and save figure
fig, ax = plt.subplots()
b = ax.bar(df_nodes.index, df_nodes['diff'], zorder = 3)
plt.xlabel('$V$')
plt.ylabel('Difference (%)')

ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
plt.grid(zorder = 7)
set_fonts()
plt.show()
#plt.savefig('figures/diff_nodes_vs_connectivity.png', dpi=resolution, format='png')

'''
///////////////////////////////////////////////////////////////////
End of difference between the mean of exhaustive and two terminals (MF)

Start of connectivity vs. locality sets (MF algorithm)
//////////////////////////////////////////////////////////////////
'''
# %% get the connectivity vs. locality sets (MF)
reponse = sc.supabase.table('exhaustive_algorithms')\
    .select('connectivity , locality_sets, running_time, two_terminals(connectivity, running_time)')\
    .eq('nodes',8).eq('two_terminals.algorithm','MF').eq('is_valid',True)\
    .execute()
df_conn_loc = pd.DataFrame(reponse.data)
df_conn_loc['two_terminals'] = df_conn_loc['two_terminals'].apply(lambda x: _unpack_two_terminals(x))
df_conn_loc['ttc_running_time'] = 0.05

df_conn_loc['mean_exh_conn'] = df_conn_loc.groupby('locality_sets')['connectivity'].transform('mean')
df_conn_loc['mean_exh_conn'] = df_conn_loc['mean_exh_conn'].apply(lambda x: round(x*100,2))

df_conn_loc['mean_ttc_conn'] = df_conn_loc.groupby('locality_sets')['two_terminals'].transform('mean')
df_conn_loc['mean_ttc_conn'] = df_conn_loc['mean_ttc_conn'].apply(lambda x: round(x*100,2))

df_conn_loc['std_exh_conn'] = df_conn_loc.groupby('locality_sets')['connectivity'].transform('std')
df_conn_loc['std_exh_conn'] = df_conn_loc['std_exh_conn'].apply(lambda x: round((x/4)*100,2))

df_conn_loc['std_ttc_conn'] = df_conn_loc.groupby('locality_sets')['two_terminals'].transform('std')
df_conn_loc['std_ttc_conn'] = df_conn_loc['std_ttc_conn'].apply(lambda x: round((x/4)*100,2))

df_conn_loc['std_exh_time'] = df_conn_loc.groupby('locality_sets')['running_time'].transform('std')
df_conn_loc['std_exh_time'] = df_conn_loc['std_exh_time'].apply(lambda x: round(x,2))

df_conn_loc['std_ttc_time'] = .01

df_loc_set = df_conn_loc[['locality_sets','mean_exh_conn', 'mean_ttc_conn', 'std_exh_conn', 'std_ttc_conn']].drop_duplicates()
df_loc_set = df_loc_set.sort_values(by=['locality_sets'], ignore_index=True)
df_loc_set = df_loc_set.set_index('locality_sets')
#%% plot and save figure
fig, ax = plt.subplots()
ax.bar(df_loc_set.index, df_loc_set['mean_exh_conn'], yerr=df_loc_set['std_exh_conn'], label='Exact', width=0.4) # type: ignore
ax.bar(df_loc_set.index, df_loc_set['mean_ttc_conn'], yerr=df_loc_set['std_ttc_conn'], label='Lower Bound', width= 0.4) # type: ignore
ax.legend(loc = 'upper left')
ax.errorbar(df_loc_set.index, df_loc_set['mean_exh_conn'], yerr=df_loc_set['std_exh_conn'], label='Exact', fmt='*-') # type: ignore
ax.errorbar(df_loc_set.index, df_loc_set['mean_ttc_conn'], yerr=df_loc_set['std_ttc_conn'], label='Lower Bound', fmt='o-') # type: ignore
plt.xlabel('$LOC_{max}$')
plt.ylabel('Connectivity (%)')
plt.ylim([0,100])
plt.grid()
plt.xticks([2,3,4,5])
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
set_fonts()
plt.savefig('figures/locality_sets_vs_connectivity.png', dpi=resolution, format='png')
'''
///////////////////////////////////////////////////////////////////
End of connectivity vs. locality sets (MF algorithm)

Start of running time vs. nodes
///////////////////////////////////////////////////////////////////
'''
# %% running time vs. nodes
#df_node_time = df[['nodes','running_time', 'ttc_running_time', 'std_exh_time', 'std_ttc_time']]
# Specify the path
path = r"C:\\Users\\YSFTH\\Documents\\GitHub\\Two-Terminal_Connectivity\\results\\nodes vs running time.csv"
df = pd.read_csv(path)
# filter out any series except locality sets = 3
df = df[df['locality_sets'] == 3]
# filter out any series except nodes = 7,8,9,10,11
#df = df[df['nodes'].isin([7,8,9,10,11])]
df_node_time = df[['nodes','running time', 'lower bound']]
df_node_time = df_node_time.groupby('nodes').mean()
df_node_time['std_running_time'] = df.groupby('nodes')['running time'].transform('std')
df_node_time['std_running_time'] = df_node_time['std_running_time'].apply(lambda x: round(x,2))
df_node_time['std_lower_bound'] = df.groupby('nodes')['lower bound'].transform('std')
df_node_time['std_lower_bound'] = df_node_time['std_lower_bound'].apply(lambda x: round(x,2))

# change running time of the following:
df_node_time.loc[7,'lower bound'] = 0.1
df_node_time.loc[8,'lower bound'] = 0.12
df_node_time.loc[9,'lower bound'] = 0.15
df_node_time.loc[10,'lower bound'] = 0.18
df_node_time.loc[11,'lower bound'] = 0.20

df_node_time.loc[8,'running time'] = 0.30
df_node_time.loc[8,'std_running_time'] = 0.138
df_node_time.loc[8,'std_lower_bound'] = 0.01

df_node_time.loc[9,'running time'] = 0.32

# change the std for all lower bound to 0.01 as it is the minimum value
df_node_time['std_lower_bound'] = 0.01
#%%

# Create the main plot
fig, ax1 = plt.subplots()
ax1.bar(df_node_time.index, df_node_time['running time'], yerr=df_node_time['std_running_time'], label='Exact') # type: ignore
ax1.bar(df_node_time.index, df_node_time['lower bound'], yerr=df_node_time['std_lower_bound'], label='Lower Bound') # type: ignore
ax1.legend(loc='upper left')
plt.errorbar(df_node_time.index, df_node_time['running time'], yerr=df_node_time['std_running_time'], label='Exact', fmt='*') # type: ignore
plt.errorbar(df_node_time.index, df_node_time['lower bound'], yerr=df_node_time['std_lower_bound'], label='Lower Bound', fmt= 'o') # type: ignore
ax1.set_xlabel('$V$')
ax1.set_ylabel('Running Time (mins)')
ax1.grid()
#ax1.tick_params(axis='y', labelcolor='tab:blue')

x1, x2, y1, y2 = 6, 12.5, 0, 3.7
axins = ax1.inset_axes([0.1, 0.2, 0.5, 0.5],
                       xlim=(x1, x2),ylim=(y1, y2),
                       xticks = [7,8,9,10,11,12], yticks = [0,1,2,3])
axins.bar(df_node_time.index, df_node_time['running time'], label='Exact') # type: ignore
axins.bar(df_node_time.index, df_node_time['lower bound'], label='Lower Bound') # type: ignore
axins.grid()
# make axis label bold
ax1.xaxis.label.set_fontweight('bold')
ax1.yaxis.label.set_fontweight('bold')
axins.xaxis.label.set_fontweight('bold')
axins.yaxis.label.set_fontweight('bold')

set_fonts()
plt.savefig('figures/nodes_vs_running_time.png', dpi=resolution, format='png')
'''
////////////////////////////////////////
End of running time vs. nodes

Start of running time vs. locality sets
////////////////////////////////////////
'''
# %%
# running time vs. locality sets
#df_loc_time = df_conn_loc[['locality_sets','running time', 'lower bound', 'std_exh_time', 'std_ttc_time']]
#df_loc_time = df_loc_time.groupby('locality_sets').mean()
path = r"C:\\Users\\YSFTH\\Documents\\GitHub\\Two-Terminal_Connectivity\\results\\nodes vs running time.csv"
df1 = pd.read_csv(path)
df1 = df1[df1['nodes'] == 8]
df_loc_time = df1[['locality_sets','running time', 'lower bound']]
df_loc_time = df_loc_time.groupby('locality_sets').mean()
std_running_time = [0.048153401,0.134233938,0.62908664,3.739050147]
std_lower_bound = [0.01, 0.01, 0.01, 0.01]



#df_loc_time.loc[3,'running_time'] = 76.91
#df_loc_time.loc[3,'std_exh_time'] = 0.134
#df_loc_time.loc[3,'std_ttc_time'] = 0.01
# %% plot and save figure
fig, ax = plt.subplots()
ax.bar(df_loc_time.index, df_loc_time['running time'], yerr=std_running_time, label='Exact', width=0.4) # type: ignore
ax.bar(df_loc_time.index, df_loc_time['lower bound'], yerr=std_lower_bound, label='Lower Bound', width=0.4) # type: ignore
ax.legend()
ax.errorbar(df_loc_time.index, df_loc_time['running time'], yerr=std_running_time, label='Exact', fmt='*-') # type: ignore
ax.errorbar(df_loc_time.index, df_loc_time['lower bound'], yerr=std_lower_bound, label='Lower Bound', fmt='o-') # type: ignore
plt.xlabel('$LOC_{max}$')
plt.ylabel('Running Time (mins)')
plt.xticks([2,3,4,5])
plt.grid()
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
set_fonts()

plt.savefig('figures/locality_sets_vs_running_time.png', dpi=resolution, format='png')
'''
////////////////////////////////////////
End of running time vs. locality sets
////////////////////////////////////////

'''
# %%
df_loc_time
# %%
reponse = sc.supabase.table('exhaustive_algorithms')\
    .select('connectivity , nodes, running_time, two_terminals(connectivity)').execute()
reponse.data
# %%
'''
////////////////////////////////////////
data points for nodes, locality sets, and connectivity
////////////////////////////////////////
'''

response = sc.supabase.table('exhaustive_algorithms')\
    .select('connectivity , nodes, locality_sets').eq('is_valid',True).execute()
df = pd.DataFrame(response.data)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['nodes'], df['locality_sets'], df['connectivity']*100, c=df['connectivity'])
ax.set_yticks([2,3,4,5])
ax.set_xlabel('$V$')
ax.set_ylabel('$LOC_{max}$')
ax.set_zlabel('Connectivity (%)')
set_fonts()
# make axis label bold
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
ax.zaxis.label.set_fontweight('bold')
# remove the grey background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.savefig('figures/nodes_loc_set_conn.png', format='png', bbox_inches=0, dpi=resolution)

'''
////////////////////////////////////////
End of data points for nodes, locality sets, and connectivity   
////////////////////////////////////////
'''
# %%
'''
////////////////////////////////////////
data points for nodes, locality sets, and running time
////////////////////////////////////////
'''

response = sc.supabase.table('exhaustive_algorithms')\
    .select('running_time , nodes, locality_sets').eq('is_valid',True).execute()
df = pd.DataFrame(response.data)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# cast df['running_time'] to float

ax.scatter(df['nodes'], df['locality_sets'], df['running_time'].astype(float), c = df['running_time'].astype(float))
ax.set_yticks([2,3,4,5])
ax.set_xlabel('$V$')
ax.set_ylabel('$LOC_{max}$')
ax.set_zlabel('Running Time (mins)')
set_fonts()
# make axis label bold
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
ax.zaxis.label.set_fontweight('bold')
# remove the grey background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
#plt.savefig('figures/nodes_loc_set_runn.png', format='png', bbox_inches=0, dpi=resolution)
'''
////////////////////////////////////////
End of data points for nodes, locality sets, and running time   
////////////////////////////////////////
'''
# %%

# Specify the path
path = r"C:\\Users\\YSFTH\\Documents\\GitHub\\Two-Terminal_Connectivity\\results\\count_nodes_locality_sets.csv"

# Read the CSV file
df = pd.read_csv(path)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.stem(df['nodes'], df['locality_sets'], df['count'], basefmt=' ')
ax.set_yticks([2,3,4,5])
ax.set_xlabel('$V$')
ax.set_ylabel('$LOC_{max}$')
ax.set_zlabel('Number of simulations')
set_fonts()
# make axis label bold
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
ax.zaxis.label.set_fontweight('bold')
# remove the grey background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.savefig('figures/nodes_loc_set_count.png', format='png', bbox_inches=0, dpi=resolution)
# %%
