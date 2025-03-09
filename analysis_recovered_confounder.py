#%% load trained flow model
import json

experiment_name = 'train_flow_ray_2024-05-11_00-23-25' # linear 1d
experiment_name = 'train_flow_ray_2024-05-21_02-58-17' # nonlinear 5d


flow_results_folder = results_folder + f'ray/{experiment_name}/analysis/' 
os.makedirs(flow_results_folder, exist_ok=True)

keep_folder = 1
with open(results_folder + f'ray{("/keep" if keep_folder == 1 else "")}/{experiment_name}/ray_results_dataframe_{experiment_name}.pkl', 'rb') as file:
    resdf = pickle.load(file)


resdf.keys()
resdf[['deconf_dict/sim_Yint_slope', 'deconf_dict/flow_Yint_slope']]

resdf['flow_train_dict/sim_dict/df'][0]

# drop these because they are saved in an unwieldy format in resdf anyways
# if you want to access these df, load the metrics
resdf = resdf.drop(['deconf_dict/df_deconf'], axis = 1)
resdf = resdf.drop(['deconf_dict/sim_dict/df'], axis = 1)


def compute_mutual_information(M):
    M = np.array(M)
    P_L = M.sum(axis=1)
    P_H = M.sum(axis=0)
    MI = 0
    for l in range(M.shape[0]):
        for h in range(M.shape[1]):
            if M[l, h] > 0:  # To avoid division by zero and log of zero
                MI += M[l, h] * np.log(M[l, h] / (P_L[l] * P_H[h]))
    return MI


with open(results_folder + f'ray{("/keep" if keep_folder == 1 else "")}/{experiment_name}/ray_results_{experiment_name}.pkl', 'rb') as file:
    loaded_ray_results = pickle.load(file)

results_list = []
correlation_results = []
for i, result in enumerate(loaded_ray_results):
    cur_metrics = result.metrics
    
    M = cur_metrics['flow_train_dict']['sim_dict']['M']
    MI = compute_mutual_information(M)
    
    KL = cur_metrics['config']['dgp_params']['no_X_cluster']
    KQ = cur_metrics['config']['dgp_params']['no_U_cluster']
    
            
    #############
    # extract true and recovered confounder
    sim_dict_df = cur_metrics['deconf_dict']['sim_dict']['df'] # take U from here
    df_deconf = cur_metrics['deconf_dict']['df_deconf'] # take Z2 from here
    
    sim_dict_df.columns
    df_deconf.columns
    
    # Combine the two dataframes
    all_data = pd.concat([sim_dict_df[['U']], df_deconf[['Z6']]], axis=1)
    
    # Compute the correlation coefficient
    correlation = np.corrcoef(all_data['U'], all_data['Z6'])[0, 1]

    # Append the results to the list
    correlation_results.append({
        'Absolute Correlation': np.abs(correlation),
        'Mutual Information': MI,
        'KL': KL,
        'KQ': KQ
    })

# Create a DataFrame to store the correlation results
correlation_df = pd.DataFrame(correlation_results) 

# Plot a histogram of the correlation coefficients
plt.figure(figsize=(10, 6))
plt.hist(np.abs(correlation_df['Absolute Correlation']), bins=10, alpha=0.75)
plt.title('Histogram of Absolute Correlation Coefficients')
plt.xlabel('Absolute Correlation Coefficient')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Discretize the MI variable
max_MI = 0.3
bins = np.arange(0, max_MI + 0.1, 0.1)
correlation_df = correlation_df.loc[correlation_df["Mutual Information"] < max_MI]

# labels = [f'{round(b, 1)}-{round(b+0.1, 1)}' for b in bins[:-1]]
labels = [f'{b:.1f}'[1:] + '-' + f'{b + 0.1:.1f}'[1:] if b < 1 else f'{b:.1f}-{b + 0.1:.1f}' for b in bins[:-1]]
correlation_df['MI_category'] = pd.cut(correlation_df["Mutual Information"], bins=bins, labels=labels, include_lowest=True)

# Assuming correlation_df is already provided with MI_category
correlation_df = correlation_df.drop(columns=['KL'])
# Melt the DataFrame to long format for seaborn
df_melted = correlation_df.melt(id_vars=['MI_category', 'KQ'], value_vars=['Absolute Correlation'],
                           var_name='Metric', value_name='Value')

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

# Create the FacetGrid
g = sns.FacetGrid(df_melted, row='KQ', sharey=True, height=2, aspect=1.3)

# Define the plotting function
def boxplot_func(data, **kwargs):
    sns.boxplot(x='MI_category', y='Value', hue='Metric', data=data, 
                palette={'Absolute Correlation': 'orange'}, 
                linewidth=2.5, ax=plt.gca())
    ax = plt.gca()
    plt.xlabel('Mutual Information')
    plt.ylabel('Absolute Correlation')
    plt.legend(title='', loc='upper right')

# Apply the plotting function to each subset of the data
g.map_dataframe(boxplot_func)
g.set_titles(row_template=r'$K_L = K_Q = {row_name}$')
# g.add_legend(title='', bbox_to_anchor=(0.35, 0.05), loc='upper center', ncol=2)

# Adjust y-axis limits to start at 0
for ax in g.axes.flat:
    ax.set_ylim(0, ax.get_ylim()[1])
plt.savefig(f'{plot_folder}sim/conf_correlation_{experiment_name}.png', bbox_inches='tight', dpi=400)
plt.show()

# Save and show the plot
plt.show()

