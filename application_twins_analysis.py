# covid analysis
#%%
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
from get_toy_data import *
from flow_auxiliary_fns import *
from flow_architecture import *


#%%

# the file that contains the results for the empirical application (about 1.5GB)
# can be obtained from the authors
exp_time_str = 'train_flow_ray_2024-05-09_15-58-13' # twin results, that's in the paper

resdf_folder = root_folder + 'application/twins/results/'
with open(f'{save_folder}/ray_results_{exp_time_str}.pkl', 'rb') as file:
    print(file)
    loaded_resdf = pickle.load(file)

application_type = 'twins'
appl_results_folder = results_folder + f'../application/twins/results//{exp_time_str}//' 
os.makedirs(appl_results_folder, exist_ok=True)

resdf = loaded_ray_results.get_dataframe()


### resdf can be loaded directly
resdf_folder = root_folder + 'application/twins/results/'
with open(f'{resdf_folder}/ray_results_dataframe_{exp_time_str}.pkl', 'rb') as file:
    loaded_resdf = pickle.load(file)

resdf.keys()

ttt = resdf[["loss", 'config/no_layers']]

## print the settings used in the flow model:
config_columns = [col for col in resdf.columns if col.startswith('config/')]
config_columns = [col for col in config_columns if not col.startswith('config/dgp_params/')]

for col in config_columns[:-3]:
    print(f"{col}: {resdf[col].unique()}")

res_allslopes = pd.DataFrame(columns=['trial_id',
                                      'slope_naive', 
                                      'slope_full', 
                                      'slope_deconf', 
                                      'no_layers',
                                      'n_classes',
                                      'batch_size',
                                      'num_epochs',
                                      'loss',
                                      'filename'])



# prepare data for plotting
twins_folder = root_folder + "//application//twins//data"

all_slopes = []  # Initialize an empty list to store slopes
results_list = []
for i, result in enumerate(loaded_ray_results):
    # print(i)
    cur_metrics = result.metrics
    
    deconf_dict = cur_metrics["deconf_dict"]
    deconf_df = deconf_dict["df_deconf"]
    
    
    
    X = deconf_df[['X1', 'X2', 'X3']]
    X = sm.add_constant(X)  
    y = deconf_df['Yint_flow']
    
    model_on_Yintflow = sm.OLS(y, X)  
    model_on_Yintflow = model_on_Yintflow.fit()
    slopes = model_on_Yintflow.params[["X1", "X2", "X3"]]
    
    print(model_on_Yintflow.summary())
    
    trial_data = {
        "trial_id": cur_metrics["trial_id"],
        'flow_type': cur_metrics['config']['flow_type'],
        "X1": slopes["X1"],
        "X2": slopes["X2"],
        "X3": slopes["X3"],
    }
    
    all_slopes.append(trial_data)  

    file_path = os.path.join(twins_folder, '..', f'regressions_on_Yint_flow_{cur_metrics["trial_id"]}.txt')
    
    with open(file_path, 'w') as f:
        f.write("Regression of Yint_flow on X1:\n")
        f.write(model_on_Yintflow.summary().as_text())
        f.write("\n\n")  

df_slopes = pd.DataFrame(all_slopes)



# compute beta^* and \hat{beta}
sim_dict = cur_metrics['deconf_dict']['sim_dict']['df']

# compute the "target" slopes using the observed confounders
res_target_slopes = []

# Assuming 'deconf_df' is your DataFrame and it includes 'Yint_flow'
X = sim_dict[['X1', 'X2', 'X3']]
X = sm.add_constant(X)  
y = sim_dict['Y']

model_on_Y = sm.OLS(y, X)  
model_on_Y = model_on_Y.fit()
slopes = model_on_Y.params[["X1", "X2", "X3"]]

wt_conf_slopes = {
    "trial_id": 'wt_obs_conf',
    "X1": slopes["X1"],
    "X2": slopes["X2"],
    "X3": slopes["X3"],
}

res_target_slopes.append(wt_conf_slopes)

# ok now with observed confounders
X = sim_dict[['X1', 'X2', 'X3']]
X = sm.add_constant(X)  
y = sim_dict['Y']


X = sim_dict.drop(columns=['U', 'Yint', 'Y'])
X = sm.add_constant(X)  
y = sim_dict['Y']
model_on_Y_w_conf = sm.OLS(y, X)  
model_on_Y_w_conf = model_on_Y_w_conf.fit()
slopes = model_on_Y_w_conf.params[["X1", "X2", "X3"]]

w_conf_slopes = {
    "trial_id": 'w_obs_conf',
    "X1": slopes["X1"],
    "X2": slopes["X2"],
    "X3": slopes["X3"],
}

res_target_slopes.append(w_conf_slopes)


mean_estimated = {
    "trial_id": 'mean_flow',
    "X1": df_slopes["X1"].mean(),
    "X2": df_slopes["X2"].mean(),
    "X3": df_slopes["X3"].mean()
}

res_target_slopes.append(mean_estimated)

res_target_slopes = pd.DataFrame(res_target_slopes)



#%%
name_map = {'X1':"mother's age","X2": "gestation type","X3": "mother's education"}
df_slopes = df_slopes.rename(columns = name_map)
res_target_slopes = res_target_slopes.rename(columns = name_map)
# Prepare the plot
fig, axes = plt.subplots(3, 1, figsize=(2, 4), sharex=False)

# Plot boxplots
sns.boxplot(data=df_slopes.rename(columns = {"mother's age": "parameter estimate"}), x="parameter estimate", ax=axes[0], color='lightgray')
sns.boxplot(data=df_slopes.rename(columns = {"gestation type": "parameter estimate"}), x="parameter estimate", ax=axes[1], color='lightgray')
sns.boxplot(data=df_slopes.rename(columns = {"mother's education": "parameter estimate"}), x="parameter estimate", ax=axes[2], color='lightgray')

alpha_value = .8
# Overlay points
variables = ["mother's age", "gestation type", "mother's education"]
for i, col in enumerate(variables):
    axes[i].scatter(res_target_slopes[col].iloc[2], 0, color='orange', marker='x', linewidth = 4, s=100, alpha=alpha_value, zorder=6, label='DeconFlow')
    axes[i].scatter(res_target_slopes[col].iloc[0], 0, color='red', marker='^', s=100, alpha=alpha_value, zorder=5, label='w/t obs. conf.')
    axes[i].scatter(res_target_slopes[col].iloc[1], 0, color='green', marker='o', s=100, alpha=alpha_value, zorder=5, label='w/ obs. conf.')
    axes[i].set_title(col)
    
    # Extend the x-axis by 20% of the individual ranges
    data_range = df_slopes[col].max() - df_slopes[col].min()
    axes[i].set_xlim([df_slopes[col].min() - 0.3 * data_range, df_slopes[col].max() + 0.3 * data_range])
    
    
# Add legend only once
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=1, frameon=True)

# Adjust layout
plt.tight_layout(rect=[0, 0.15, 1, 1])
# Save the figure
plt.savefig(twins_folder + '/../plots/hist_with_Target_Slopes_compact.png', dpi=400)  # Save as PNG file with high resolution

plt.show()
