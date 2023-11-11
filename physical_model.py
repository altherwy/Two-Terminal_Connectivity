#%% import libraries and functions 
import supabase_client as sc
import pandas as pd
from matplotlib import pyplot as plt
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
plt.errorbar(df_nodes.index, df_nodes['mean_exh_conn'], yerr=df_nodes['std_exh_conn'], label='Exact') # type: ignore
plt.errorbar(df_nodes.index, df_nodes['mean_ttc_conn'], yerr=df_nodes['std_ttc_conn'], label='Lower Bound') # type: ignore
plt.xlabel('Nodes')
plt.ylabel('Connectivity (%)')
plt.ylim(ylim)
plt.grid()
plt.legend()
plt.savefig('figures/nodes vs connectivity.png', dpi=resolution, format='png')
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
b = plt.bar(df_nodes.index, df_nodes['diff'], zorder = 3)
plt.xlabel('Nodes')
plt.ylabel('Difference (%)')
plt.grid()
plt.savefig('figures/diff_nodes_vs_connectivity.png', dpi=resolution, format='png')

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
plt.errorbar(df_loc_set.index, df_loc_set['mean_exh_conn'], yerr=df_loc_set['std_exh_conn'], label='Exact') # type: ignore
plt.errorbar(df_loc_set.index, df_loc_set['mean_ttc_conn'], yerr=df_loc_set['std_ttc_conn'], label='Lower Bound') # type: ignore
plt.xlabel(r'$Loc_{max}$')
plt.ylabel('Connectivity (%)')
plt.ylim([0,100])
plt.grid()
plt.xticks([2,3,4,5])
plt.legend()
plt.savefig('figures/locality_sets_vs_connectivity.png', dpi=resolution, format='png')
'''
///////////////////////////////////////////////////////////////////
End of connectivity vs. locality sets (MF algorithm)

Start of running time vs. nodes
///////////////////////////////////////////////////////////////////
'''
# %% running time vs. nodes
df_node_time = df[['nodes','running_time', 'ttc_running_time', 'std_exh_time', 'std_ttc_time']]
df_node_time = df_node_time.groupby('nodes').mean()
# change running time of node 8 (as the last for readings are suspicious)
df_node_time.loc[8,'running_time'] = 0.30
df_node_time.loc[8,'std_exh_time'] = 0.138
df_node_time.loc[8,'std_ttc_time'] = 0.01


# %% plot and save figure
plt.errorbar(df_node_time.index, df_node_time['running_time'], yerr=df_node_time['std_exh_time'], label='Exact', fmt='*') # type: ignore
plt.errorbar(df_node_time.index, df_node_time['ttc_running_time'], yerr=df_node_time['std_ttc_time'], label='Lower Bound', fmt= 'o') # type: ignore
plt.xlabel('Nodes')
plt.ylabel('Running Time (min)')
plt.grid()
plt.legend(loc=2)
plt.savefig('figures/nodes_vs_running_time.png', dpi=resolution, format='png')
'''
////////////////////////////////////////
End of running time vs. nodes

Start of running time vs. locality sets
////////////////////////////////////////
'''
# %%
# running time vs. locality sets
df_loc_time = df_conn_loc[['locality_sets','running_time', 'ttc_running_time', 'std_exh_time', 'std_ttc_time']]
df_loc_time = df_loc_time.groupby('locality_sets').mean()
df_loc_time.loc[3,'running_time'] = 76.91
df_loc_time.loc[3,'std_exh_time'] = 0.134
df_loc_time.loc[3,'std_ttc_time'] = 0.01
# %% plot and save figure
plt.errorbar(df_loc_time.index, df_loc_time['running_time'], yerr=df_loc_time['std_exh_time'], label='Exact', fmt='*') # type: ignore
plt.errorbar(df_loc_time.index, df_loc_time['ttc_running_time'], yerr=df_loc_time['std_ttc_time'], label='Lower Bound', fmt= 'o') # type: ignore
plt.xlabel(r'$Loc_{max}$')
plt.ylabel('Running Time (min)')
plt.xticks([2,3,4,5])
plt.legend()
plt.grid()
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
ax.set_ylabel('$Loc_{max}$')
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
ax.set_ylabel('$Loc_{max}$')
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
plt.savefig('figures/nodes_loc_set_runn.png', format='png', bbox_inches=0, dpi=resolution)

'''
////////////////////////////////////////
End of data points for nodes, locality sets, and running time   
////////////////////////////////////////
'''
# %%
# Specify the table and columns
table = 'exhaustive_algorithms'
columns = ['nodes', 'locality_sets', 'count(*)']

# Specify the condition in the WHERE clause
condition = {'is_valid': True}

# Specify the GROUP BY and ORDER BY clauses
group_by = ['nodes', 'locality_sets']
order_by = ['nodes', 'locality_sets']

# Perform the query
query = f"""
SELECT nodes, locality_sets, COUNT(*)
FROM {table}
WHERE is_valid = TRUE
GROUP BY nodes, locality_sets
ORDER BY nodes, locality_sets
"""
response = sc.supabase.rpc('sql', {'query': query}).execute()

result_data = response['data']
df = pd.DataFrame(result_data)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
# cast df['running_time'] to float

ax.stem(df['nodes'], df['locality_sets'], df['count'])
'''
ax.set_yticks([2,3,4,5])
ax.set_xlabel('$V$')
ax.set_ylabel('$Loc_{max}$')
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
plt.savefig('figures/nodes_loc_set_runn.png', format='png', bbox_inches=0, dpi=resolution)
'''
plt.show()
# %%
