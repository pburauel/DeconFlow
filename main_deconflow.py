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

for folder in [results_folder, results_folder + 'model_paras', plot_folder, data_folder]:
    os.makedirs(folder, exist_ok=True)

import tempfile


from dependencies import *

from flow_auxiliary_fns import *




'''
there are two ways to start DeconFlow training:
    1: use a Ray Tuner object (this is useful for large-scale experimentation on e.g. AWS)
    2: call the train_flow_ray function directly



param_space:
    application: choose `sim` for simulation or `twins` for twins application
    device: choose if trained on cuda or cpu
    plot_during_training: if 1, plot intermediate results during training
    save_interim_results: if 1, saves interim results to disk
    lr_start: learning rate at the beginning of the training cycle
    lr_end: learning rate at the end of the learning cycle
    flow_type: choose type of flow model, can be `linear` or `nonlinear`
    datafile_name: choose the file that contains the input data for the flow model
        if `_generate_within_ray`, then it simulates data within one run
        if `twins_flow_cause3D`, then it uses twins data
    dgp_params: parameters for the simulated dataset
        'xdim': Dimension of X
        'n_samples': Number of samples
        'no_X_cluster': Number of clusters in X
        'no_U_cluster': Number of clusters in U
        'phiX_piecewise_affine': if 1, then phiX is a piecewise affine transformation, if 0, phiX is linear
        'phiY_piecewise_affine': if 1, then phiY is a piecewise affine transformation, if 0, phiY is linear
        'psiY_piecewise_affine': if 1, then psiY is a piecewise affine transformation, if 0, psiY is linear
        'no_resnet_blocks': number of resnet blocks for the invertible transformations
    no_layers: number of layers of the flow model
    n_classes: number of classes of the GMM model (can be `true` when application == `sim`, in that case it chooses the correct number of clusters)
    batch_size: batch size for flow learning
    num_epochs: number of epochs for flow learning
'''


#%% train flow aws

'''
Unique values in 'no_U_cluster': [3 2]
Unique values in 'no_X_cluster': [3 2]
Unique values in 'xdim': [5]
Unique values in 'num_epochs': [8000]
Unique values in 'lr_start': [0.001]
Unique values in 'lr_end': [1.e-08]
Unique values in 'batch_size': [200]
Unique values in 'no_layers': [50]
Unique values in 'n_classes': ['true']
'''

# choose hyperparameters and data files here
data_file_names = ["twins_flow_cause3D_allconf"]


dgp_params = {
    'xdim': 5,                       # Dimension of X
    'n_samples': 3000,                # Number of samples
    'no_X_cluster': 2,               # Number of clusters in X
    'no_U_cluster': 2,               # Number of clusters in U
    'phiX_piecewise_affine': 1,      # Piecewise affine parameter for phiX
    'phiY_piecewise_affine': 1,      # Piecewise affine parameter for phiY
    'psiY_piecewise_affine': 1,       # Piecewise affine parameter for psiY
    'no_resnet_blocks': 5 # number of resnet blocks for the invertible transformations
}





##########################################
# 1 training using a Ray tuner object
##########################################


l_plot_during_training = 0

param_space = {
        'application': 'sim', # this is important to choose the correct data frames etc
        "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"), #'cpu', # 
        "plot_during_training": l_plot_during_training,
        "save_interim_results": 0,
        "lr_start": tune.grid_search([1e-2]),
        "lr_end": tune.grid_search([1e-6]),
        "flow_type": tune.grid_search(["nonlinear"]),
        "datafile_name": "_generate_within_ray", # generates synthetic data
        # "datafile_name": tune.grid_search(data_file_names), # choose a specific dataset
        "dgp_params": tune.grid_search([dgp_params]),
        "no_layers": tune.grid_search([50]), 
        "n_classes": tune.grid_search(["true"]), # possible to choose true here, in sim mode it will then choose the true number of clusters
        "batch_size": tune.grid_search([50]),
        "num_epochs" : tune.grid_search([1000]*10)
    }




tuner = Tuner(
    trainable = train_flow_ray,
    param_space=param_space,
    run_config=RunConfig(storage_path = results_folder + 'ray'))

# run ray tuner
ray_results = tuner.fit()

resdf = ray_results.get_dataframe()

# save ray tune results
exp_time_str = ray_results.experiment_path.split('/')[-1]
with open(f'{ray_results.experiment_path}/ray_results_dataframe_{exp_time_str}.pkl', 'wb') as file:
    pickle.dump(resdf, file)
with open(f'{ray_results.experiment_path}/ray_results_{exp_time_str}.pkl', 'wb') as file:
    pickle.dump(ray_results, file)
if locat == 'local':
    excel_path = f'{ray_results.experiment_path}/resdf_{exp_time_str}.xlsx'
    resdf.to_excel(excel_path, index=False)





##########################################
# 1 training by calling train_flow_ray directly
##########################################
locat = 'remote'
if locat == 'local':

    
    dgp_params = {
        'xdim': 3,                       # Dimension of X
        'n_samples': 1000,                # Number of samples
        'no_X_cluster': 2,               # Number of clusters in X
        'no_U_cluster': 2,               # Number of clusters in U
        'phiX_piecewise_affine': 0,      # Piecewise affine parameter for phiX
        'phiY_piecewise_affine': 0,      # Piecewise affine parameter for phiY
        'psiY_piecewise_affine': 0       # Piecewise affine parameter for psiY
    }
    
    param_space = {
            'application': 'sim',#'twins', # this is important to choose the correct data frames etc
            # 'application': 'twins', # this is important to choose the correct data frames etc
            "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"), # or 'cpu'
            "plot_during_training": 0,
            "save_interim_results": 0,
            "lr_start": 1e-3,
            "lr_end": 1e-8,
            "flow_type": 'nonlinear',
            "datafile_name": "_generate_within_ray",
            # "datafile_name": "twins_flow_cause3D_allconf", # choose a specific dataset
            "dgp_params": dgp_params,
            "no_layers": 5,
            "n_classes": 4,
            "batch_size": 200,
            "num_epochs" : 20
        }
    
    metrics = train_flow_ray(param_space)
   
    # save results
    exp_time_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(f'{results_folder}/resdf_{exp_time_str}.pkl', 'wb') as file:
        pickle.dump(metrics, file)
    
    
    
    df_deconf = metrics["deconf_dict"]['df_deconf']
    
    X = sm.add_constant(df_deconf['X1'])
    y = df_deconf['Yint_flow']
    model = sm.OLS(y, X).fit()
    intercept, slope = model.params
    
    
    org_df = metrics["deconf_dict"]['sim_dict']["df"]
    X = sm.add_constant(org_df['X1'])
    y = org_df['Y']
    model_org = sm.OLS(y, X).fit()
    intercept, slope_org = model_org.params
    


    sim_Yint_slope = metrics["deconf_dict"]['sim_Yint_slope']#, 'rmse_cfac_naive']
    flow_Yint_slope = metrics["deconf_dict"]['flow_Yint_slope']
    print(f'{np.round(sim_Yint_slope,3)}: true slope\n{np.round(flow_Yint_slope,3)}: recovered slope')
    
    
