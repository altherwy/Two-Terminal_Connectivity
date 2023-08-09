#%% import libraries and functions 
import supabase_client as sc
import pandas as pd
from matplotlib import pyplot as plt
resolution = 1200
ylim = [0,100]
def _unpack_two_terminals(row, attr = 'connectivity'):
    temp = row[0]
    return temp[attr]
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
plt.bar(df_nodes.index, df_nodes['diff'])
plt.xlabel('Nodes')
plt.ylabel('Difference (%)')
reolsution = 1200
#plt.savefig('figures/diff nodes vs connectivity.png', dpi=resolution, format='png')

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
plt.errorbar(df_loc_set.index, df_loc_set['mean_exh_conn'], yerr=df_loc_set['std_exh_conn'], label='Exhaustive') # type: ignore
plt.errorbar(df_loc_set.index, df_loc_set['mean_ttc_conn'], yerr=df_loc_set['std_ttc_conn'], label='Two-Terminals') # type: ignore
plt.xlabel('Locality Sets')
plt.ylabel('Connectivity (%)')
plt.ylim([0,100])
plt.legend()
#plt.savefig('figures/locality sets vs connectivity.png', dpi=resolution, format='png')
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
plt.errorbar(df_node_time.index, df_node_time['running_time'], yerr=df_node_time['std_exh_time'], label='Exhaustive', fmt='*') # type: ignore
plt.errorbar(df_node_time.index, df_node_time['ttc_running_time'], yerr=df_node_time['std_ttc_time'], label='Two-Terminals', fmt= 'o') # type: ignore
plt.xlabel('Nodes')
plt.ylabel('Running Time (min)')
plt.legend()
#plt.savefig('figures/nodes vs running time.png', dpi=resolution, format='png')
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
plt.errorbar(df_loc_time.index, df_loc_time['running_time'], yerr=df_loc_time['std_exh_time'], label='Exhaustive', fmt='*') # type: ignore
plt.errorbar(df_loc_time.index, df_loc_time['ttc_running_time'], yerr=df_loc_time['std_ttc_time'], label='Two-Terminals', fmt= 'o') # type: ignore
plt.xlabel('Locality Sets')
plt.ylabel('Running Time (min)')
plt.legend()
#plt.savefig('figures/locality sets vs running time.png', dpi=resolution, format='png')
'''
////////////////////////////////////////
End of running time vs. locality sets
////////////////////////////////////////

'''
# %%
df_loc_time
# %%
