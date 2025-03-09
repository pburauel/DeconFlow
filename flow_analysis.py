#%%

'''
The ray result dictionaries that form the basis of the plots in the
paper are large (on the order of 2GB per file) because they contain
a lot of information that is not shown in the plots.
These files are available upon request by the authors.
We include this script as it can be used to construct the same type of
plots for experiments that a user might want to run for themselves.
'''


import os
import socket


### this is for AWS use:
# # i.e. if we are on AWS server w home folder '/home/ec2-user'
# root_folder = '/home/ec2-user/gip/'
# results_folder = root_folder
# data_folder = root_folder + "data/"
# no_epochs = 300
# no_epochs_autoencoder = 100
# locat = 'AWS'


root_folder = "C:/Users/pfbur/Dropbox/acad_projects/deconflow/"
os.chdir(root_folder + 'code')
results_folder = root_folder + 'results/'
plot_folder = results_folder + 'plots/'
data_folder = "../data/"



from dependencies import *
from toy_data import *
# from toy_data_n_cluster import *

from get_toy_data import *
from flow_auxiliary_fns import *
from flow_architecture import *

#%%

# set ray experiment name

experiment_name = 'train_flow_ray_2024-05-11_00-23-25' # linear 1d
experiment_name = 'train_flow_ray_2024-05-21_02-58-17' # nonlinear 5d

# experiment_name = 'train_flow_ray_2024-05-21_02-12-01' # linear 5d (not in paper)

### rebuttal simulations
experiment_name = 'train_flow_ray_2024-08-02_22-05-21' # hi d
experiment_name = 'train_flow_ray_2024-08-02_21-57-59' # lo sample size
experiment_name = 'train_flow_ray_2024-08-04_15-54-37' # misspecified n_classes
experiment_name = 'train_flow_ray_2024-08-02_23-11-24' # lo sample size w less complex model (nlayers = 10)

# results with individual treatment effects
experiment_name = 'train_flow_ray_2024-09-26_17-32-38'


#%%

try: del loaded_ray_results
except NameError:
    pass
try: del resdf
except NameError:
    pass
try: del resdf_plot
except NameError:
    pass

flow_results_folder = results_folder + f'ray/{experiment_name}/analysis/' 
os.makedirs(flow_results_folder, exist_ok=True)

keep_folder = 0
with open(results_folder + f'ray{("/keep" if keep_folder == 1 else "")}/{experiment_name}/ray_results_dataframe_{experiment_name}.pkl', 'rb') as file:
    resdf = pickle.load(file)


resdf.keys()
resdf[['deconf_dict/sim_Yint_slope', 'deconf_dict/flow_Yint_slope']]


## compute mutual information

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

# Apply the function to each matrix in the dataframe
resdf['MI'] = resdf['flow_train_dict/sim_dict/M'].apply(compute_mutual_information)

resdf = resdf.rename(columns = {'deconf_dict/sim_Yint_slope': 'sim_Yint_slope',
                                        'deconf_dict/flow_Yint_slope': 'flow_Yint_slope', 
                                        'deconf_dict/sim_Y_naive': 'sim_Y_naive',
                                        'config/dgp_params/no_U_cluster': 'no_U_cluster',
                                        'config/dgp_params/no_X_cluster': 'no_X_cluster',
                                        'config/dgp_params/xdim': 'xdim',
                                        'config/dgp_params/n_samples': 'n_samples',
                                        'config/no_layers': 'no_layers',
                                        'config/n_classes': 'n_classes',
                                        'config/batch_size': 'batch_size',
                                        'config/num_epochs': 'num_epochs',
                                        'config/flow_type': 'flow_type',
                                        'config/lr_end' : 'lr_end',
                                        'config/lr_start': 'lr_start',
                                        'deconf_dict/rmse_Yint_Y': 'rmse_naive',
                                        'deconf_dict/rmse_Yint_Yint_flow': 'rmse_flow',
                                        'deconf_dict/rmse_cfac_flow': 'rmse_cfac_flow',
                                        'deconf_dict/rmse_cfac_naive': 'rmse_cfac_naive'
                                        })



resdf['time_total_s']

# drop these because they are saved in an unwieldy format in resdf anyways
# if you want to access these df, load the metrics
resdf = resdf.drop(['deconf_dict/df_deconf'], axis = 1)
resdf = resdf.drop(['deconf_dict/sim_dict/df'], axis = 1)

resdf['relative_RMSE'] = resdf['rmse_flow']/resdf['rmse_naive']

def print_unique_values(resdf):
    columns = [
        'n_samples',
        'no_U_cluster',
        'no_X_cluster',
        'xdim',
        'num_epochs',
        'lr_start',
        'lr_end',
        'batch_size',
        'no_layers',
        'n_classes'
    ]

    for col in columns:
        unique_values = resdf[col].unique()
        print(f"Unique values in '{col}': {unique_values}")
        # print()  # For better readability

print_unique_values(resdf)
# check dgp params

resdf['no_U_cluster'].unique()
resdf['no_X_cluster'].unique()
resdf['xdim'].unique()
resdf['num_epochs'].unique()
resdf['lr_start'].unique()
resdf['lr_end'].unique()
resdf['batch_size'].unique()
resdf['no_layers'].unique()
resdf['n_classes'].unique()


resdf["relative_RMSE_truncated"] = np.where(resdf['relative_RMSE'] > 1, 1, resdf['relative_RMSE'])


resdf_org = resdf.copy()

# add MI categories
resdf_plot = resdf.copy()

# Discretize the MI variable
max_MI = 0.3
bins = np.arange(0, max_MI + 0.1, 0.1)
resdf_plot = resdf_plot.loc[resdf_plot["MI"] < max_MI]

# labels = [f'{round(b, 1)}-{round(b+0.1, 1)}' for b in bins[:-1]]
labels = [f'{b:.1f}'[1:] + '-' + f'{b + 0.1:.1f}'[1:] if b < 1 else f'{b:.1f}-{b + 0.1:.1f}' for b in bins[:-1]]
resdf_plot['MI_category'] = pd.cut(resdf_plot['MI'], bins=bins, labels=labels, include_lowest=True)



#%%# make Figure 4 in original submission

# Make a copy and filter the DataFrame
resdf_2bp = resdf_plot.copy()
print_unique_values(resdf_2bp)

# resdf_2bp = resdf_2bp.loc[resdf_2bp["no_layers"] == 50]
# resdf_2bp = resdf_2bp.loc[resdf_2bp["num_epochs"] == 5000]
# resdf_2bp = resdf_2bp.loc[resdf_2bp["batch_size"] == 400]
# resdf_2bp = resdf_2bp.loc[resdf_2bp["lr_end"] == 1e-8]

print_unique_values(resdf_2bp)


# # for linear final
# if experiment_name == 'train_flow_ray_2024-05-11_00-23-25':
#     resdf_2bp = resdf_2bp.loc[resdf_2bp["no_layers"] == 10]
#     resdf_2bp = resdf_2bp.loc[resdf_2bp["num_epochs"] == 5000]
#     resdf_2bp = resdf_2bp.loc[resdf_2bp["batch_size"] == 400]
#     resdf_2bp = resdf_2bp.loc[resdf_2bp["lr_end"] == 1e-8]
#     resdf_2bp = resdf_2bp.loc[resdf_2bp["lr_start"] == 0.001]

# for nonlinear final
if experiment_name == 'train_flow_ray_2024-05-21_02-58-17':
    resdf_2bp = resdf_2bp.loc[resdf_2bp["no_layers"] == 50]
    resdf_2bp = resdf_2bp.loc[resdf_2bp["num_epochs"] == 8000]
    resdf_2bp = resdf_2bp.loc[resdf_2bp["batch_size"] == 200]
    resdf_2bp = resdf_2bp.loc[resdf_2bp["lr_end"] == 1e-8]
    resdf_2bp = resdf_2bp.loc[resdf_2bp["lr_start"] == 0.001]

# Select relevant columns and rename them
resdf_2bp = resdf_2bp[["MI", "MI_category", 'rmse_flow', 'rmse_naive', 'no_U_cluster']]
resdf_2bp = resdf_2bp.rename(columns={'rmse_flow': 'DeconFlow', 'rmse_naive': 'naive'})

# Melt the DataFrame to long format for seaborn
df_melted = resdf_2bp.melt(id_vars=['MI_category', 'no_U_cluster'], value_vars=['DeconFlow', 'naive'],
                           var_name='Metric', value_name='Value')

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

# Create the FacetGrid
g = sns.FacetGrid(df_melted, row='no_U_cluster', sharey=True, height=2, aspect=1.3)

# Define the plotting function
def boxplot_func(data, **kwargs):
    sns.boxplot(x='MI_category', y='Value', hue='Metric', data=data, 
                palette={'DeconFlow': 'orange', 'naive': 'red'}, 
                linewidth=2.5, ax=plt.gca())
    # sns.swarmplot(x='MI_category', y='Value', hue='Metric', dodge = False, data=data, 
    #             palette={'RMSE (DeconFlow)': 'orange', 'RMSE (naive)': 'red'}, 
    #             linewidth=1, ax=plt.gca())    
    ax = plt.gca()
    plt.xlabel('Mutual Information')
    plt.ylabel('RMSE')
    plt.legend(title='', loc='upper right')

# Apply the plotting function to each subset of the data
g.map_dataframe(boxplot_func)
g.set_titles(row_template=r'$K_L = K_Q = {row_name}$', y = .8, x = .7)
g.add_legend(title='', bbox_to_anchor=(0.35, 0.05), loc='upper center', ncol=2)

# Adjust y-axis limits to start at 0
for ax in g.axes.flat:
    ax.set_ylim(0, ax.get_ylim()[1])

# Save and show the plot
# plt.savefig(f'{plot_folder}sim/rmse_comparison_{experiment_name}.png', bbox_inches='tight', dpi=400)
plt.show()

#%%# make figure that shows individual treatment effects recovery, (basis: Figure 4 in original submission)


# plot individual effects
with open(results_folder + f'ray/{experiment_name}/ray_results_{experiment_name}.pkl', 'rb') as file:
    loaded_ray_results = pickle.load(file)


# pick cur_metrics
for i, result in enumerate(loaded_ray_results):
    cur_metrics = result.metrics
    

# extract dataframse
sim_dict_df = cur_metrics['deconf_dict']['sim_dict']['df'] # take U from here
df_deconf = cur_metrics['deconf_dict']['df_deconf'] # take Z2 from here


cur_metrics["config"].keys()

cur_metrics["config"]['dgp_params']


# Creating a DataFrame with the selected columns for the pair plot
data = pd.DataFrame({
    'Ycfac_naive': sim_dict_df['Ycfac_naive'],
    'Ycfac_true': sim_dict_df['Ycfac_true'],
    'Ycfac_flow': df_deconf['Ycfac_flow']
})


### make the plot for the paper

plt.figure(figsize=(3.5, 3.5))
plt.scatter(data['Ycfac_true'], data['Ycfac_naive'], color='red', marker='^', label='naive')
plt.scatter(data['Ycfac_true'], data['Ycfac_flow'], color='orange', marker='x', label='DeconFlow')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='45-degree line')

# Adding labels and title
plt.xlabel('true individual effects')
plt.ylabel('estimated/naive individual effects')
# plt.title('Comparison of Estimated vs True Individual Effects')
plt.legend()
# plt.grid(True)
plt.savefig(f'{plot_folder}sim/individual_treatment_{experiment_name}.png', bbox_inches='tight', dpi=400)

plt.show()





#%%# make figure that shows finite sample performance (basis: Figure 4)

# Make a copy and filter the DataFrame
resdf_2bp = resdf_plot.copy()
print_unique_values(resdf_2bp)

# Select relevant columns and rename them
resdf_2bp = resdf_2bp[["MI", "MI_category", 'rmse_flow', 'rmse_naive', 'no_U_cluster', 'n_samples']]
resdf_2bp = resdf_2bp.rename(columns={'rmse_flow': 'DeconFlow', 'rmse_naive': 'naive'})

# Melt the DataFrame to long format for seaborn
df_melted = resdf_2bp.melt(id_vars=['MI_category', 'no_U_cluster', 'n_samples'], value_vars=['DeconFlow', 'naive'],
                           var_name='Metric', value_name='Value')

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

# Create the FacetGrid
g = sns.FacetGrid(df_melted, row='no_U_cluster', col = 'n_samples', sharey=True, height=2, aspect=1.3)

# Define the plotting function
def boxplot_func(data, **kwargs):
    sns.boxplot(x='MI_category', y='Value', hue='Metric', data=data, 
                palette={'DeconFlow': 'orange', 'naive': 'red'}, 
                linewidth=2.5, ax=plt.gca())
    # sns.swarmplot(x='MI_category', y='Value', hue='Metric', dodge = False, data=data, 
    #             palette={'RMSE (DeconFlow)': 'orange', 'RMSE (naive)': 'red'}, 
    #             linewidth=1, ax=plt.gca())    
    ax = plt.gca()
    plt.xlabel('Mutual Information')
    plt.ylabel('RMSE')
    plt.legend(title='', loc='upper right')

# Apply the plotting function to each subset of the data
g.map_dataframe(boxplot_func)
g.set_titles(row_template=r'$K_L = K_Q = {row_name}$', col_template =r'$N={col_name}$', y = .8, x = .7)
g.add_legend(title='', bbox_to_anchor=(0.45, 0.02), loc='upper center', ncol=2)

# Adjust y-axis limits to start at 0
for ax in g.axes.flat:
    ax.set_ylim(0, ax.get_ylim()[1])

# Save and show the plot
plt.savefig(f'{plot_folder}sim/rmse_comparison_{experiment_name}.png', bbox_inches='tight', dpi=400)
plt.show()


#%%# make figure that shows hi d performance (basis: Figure 4)

# Make a copy and filter the DataFrame
resdf_2bp = resdf_plot.copy()
print_unique_values(resdf_2bp)

resdf_2bp = resdf_2bp.loc[resdf_2bp["xdim"] == 10]

# Select relevant columns and rename them
resdf_2bp = resdf_2bp[["MI", "MI_category", 'rmse_flow', 'rmse_naive', 'no_U_cluster', 'xdim']]
resdf_2bp = resdf_2bp.rename(columns={'rmse_flow': 'DeconFlow', 'rmse_naive': 'naive'})

# Melt the DataFrame to long format for seaborn
df_melted = resdf_2bp.melt(id_vars=['MI_category', 'no_U_cluster', 'xdim'], value_vars=['DeconFlow', 'naive'],
                           var_name='Metric', value_name='Value')

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

# Create the FacetGrid
g = sns.FacetGrid(df_melted, row='no_U_cluster', sharey=True, height=2, aspect=1.3)

# Define the plotting function
def boxplot_func(data, **kwargs):
    sns.boxplot(x='MI_category', y='Value', hue='Metric', data=data, 
                palette={'DeconFlow': 'orange', 'naive': 'red'}, 
                linewidth=2.5, ax=plt.gca())
    # sns.swarmplot(x='MI_category', y='Value', hue='Metric', dodge = False, data=data, 
    #             palette={'RMSE (DeconFlow)': 'orange', 'RMSE (naive)': 'red'}, 
    #             linewidth=1, ax=plt.gca())    
    ax = plt.gca()
    plt.xlabel('Mutual Information')
    plt.ylabel('RMSE')
    plt.legend(title='', loc='upper right')

# Apply the plotting function to each subset of the data
g.map_dataframe(boxplot_func)
g.set_titles(row_template=r'$K_L = K_Q = {row_name}$', y = .8, x = .7)
g.add_legend(title='', bbox_to_anchor=(0.35, 0.05), loc='upper center', ncol=2)

# Adjust y-axis limits to start at 0
for ax in g.axes.flat:
    ax.set_ylim(0, ax.get_ylim()[1])

# Save and show the plot
plt.savefig(f'{plot_folder}sim/rmse_comparison_{experiment_name}.png', bbox_inches='tight', dpi=400)
plt.show()




#%%# misspecified n_classes figure (basis: Figure 4)

# Make a copy and filter the DataFrame
resdf_2bp = resdf_plot.copy()
print_unique_values(resdf_2bp)


# Select relevant columns and rename them
resdf_2bp = resdf_2bp[["MI", "MI_category", 'rmse_flow', 'rmse_naive', 'no_U_cluster', 'n_classes']]
resdf_2bp = resdf_2bp.rename(columns={'rmse_flow': 'DeconFlow', 'rmse_naive': 'naive'})

# Melt the DataFrame to long format for seaborn
df_melted = resdf_2bp.melt(id_vars=['MI_category', 'no_U_cluster', 'n_classes'], value_vars=['DeconFlow', 'naive'],
                           var_name='Metric', value_name='Value')

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

# Create the FacetGrid with 2 rows
g = sns.FacetGrid(df_melted, col='n_classes', sharey=True, height=2, aspect=1.3, col_wrap=3)

# Define the plotting function
def boxplot_func(data, **kwargs):
    sns.boxplot(x='MI_category', y='Value', hue='Metric', data=data, 
                palette={'DeconFlow': 'orange', 'naive': 'red'}, 
                linewidth=2.5, ax=plt.gca())
    ax = plt.gca()
    plt.xlabel('Mutual Information')
    plt.ylabel('RMSE')
    plt.legend(title='', loc='upper right')

# Apply the plotting function to each subset of the data
g.map_dataframe(boxplot_func)
g.set_titles(col_template=r'$No. classes ={col_name}$', y=.8, x=.5)
g.add_legend(title='', bbox_to_anchor=(0.45, 0.02), loc='upper center', ncol=2)

# Adjust y-axis limits to start at 0
for ax in g.axes.flat:
    ax.set_ylim(0, ax.get_ylim()[1])

# Save and show the plot
plt.savefig(f'{plot_folder}sim/rmse_comparison_{experiment_name}.png', bbox_inches='tight', dpi=400)
plt.show()





#%% make Figure 3
### use 'train_flow_ray_2024-05-11_00-23-25'
import seaborn as sns
import matplotlib.pyplot as plt
print_unique_values(df_slopes)


df_slopes = resdf.rename(columns = {'deconf_dict/sim_Yint_slope': 'sim_Yint_slope',
                                        'deconf_dict/flow_Yint_slope': 'flow_Yint_slope', 
                                        'deconf_dict/sim_Y_naive': 'sim_Y_naive'})

# Sorting and indexing data
df_slopes_sorted = df_slopes.sort_values('sim_Yint_slope').reset_index(drop=True)
df_slopes_sorted["index"] = df_slopes_sorted.index

# Define variables for plotting
xvar = 'index'
hue_var = 'xdim'  # This variable will be used to color-code the data points

# Create the plot
plt.figure(figsize=(3,3))

# Plotting each variable with different markers and labels
sns.scatterplot(data=df_slopes_sorted, x=xvar, y='sim_Yint_slope', label='true', color='green', s=80)
sns.scatterplot(data=df_slopes_sorted, x=xvar, y='flow_Yint_slope', marker='x', label='DeconFlow', color='orange', s=100, linewidth = 2.5)
sns.scatterplot(data=df_slopes_sorted, x=xvar, y='sim_Y_naive', marker='^', label='naive', color='red', s=100)

# Setting labels and title
plt.xlabel('instance (ordered by true para.)')
plt.ylabel('parameter estimate')
plt.title('')
plt.legend(title="")  # Adding legend with a title

plt.savefig(f'{plot_folder}sim/linear_1d_para_estimates.png', bbox_inches='tight', dpi=400)
plt.show()

