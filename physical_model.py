#%%
import physical_model_simulation as pms

#%%
sim = pms.PhysicalModel(loc_set_max=4)
sim.main()

# %%
# get the first row of the node positions

df = sim.node_positions.iloc[0]  
# %%
total_avg = 0
for j in range(len(sim.node_positions)):
    df = sim.node_positions.iloc[j]
    avg = 0
    for i in range(len(df)-1):
        temp = sim._get_distance(df[i], df[i+1])
        print(temp)
        avg += temp
    print('------AVG-------')
    print(avg / (len(df)-1))
    print('----------------')
    total_avg += avg / len(df)
    
total_avg /= len(sim.node_positions)

# %%
total_avg
# %%
