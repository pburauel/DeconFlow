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

import PyPDF2
import tempfile



from dependencies import *

from flow_auxiliary_fns import *

#%%

# load the twins csvs
os.getcwd()
import pandas as pd
# Path to the folder containing the CSV files
# data source is https://github.com/RickardKarl/detect-hidden-confounding

twins_folder = root_folder + "/application/twins/"


# Load each CSV file into a separate dataframe
twin_pairs_T = pd.read_csv(f'{twins_folder}data/files/twin_pairs_T_3years_samesex.csv')
twin_pairs_X = pd.read_csv(f'{twins_folder}data/files/twin_pairs_X_3years_samesex.csv')
twin_pairs_Y = pd.read_csv(f'{twins_folder}data/files/twin_pairs_Y_3years_samesex.csv')

# Display the first few rows of each dataframe to verify they're loaded correctly
print("Twin Pairs T Data:")
print(twin_pairs_T.head())
print("\nTwin Pairs X Data:")
print(twin_pairs_X.head())
print("\nTwin Pairs Y Data:")
print(twin_pairs_Y.head())

merged_df = pd.merge(twin_pairs_X, twin_pairs_T, on='Unnamed: 0')
df_flow = pd.DataFrame()
df_temp = pd.DataFrame()

# Drop rows with NA values
df = merged_df.dropna()
df = df.drop(columns = ["Unnamed: 0.1", 
                        "Unnamed: 0", 
                        "dbirwt_1", 
                        "infant_id_0", 
                        "infant_id_1",
                        "bord_0",
                        "bord_1"
                        ])



#%% generate the data set that is then fed to the flow model
snippet = "cause3D_allconf"


import json

# Load the text content from a file
with open(twins_folder + 'data/files/covar_desc.txt', 'r') as file:
    text_content = file.read()

# Convert the text content into a dictionary
var_names = json.loads(text_content)

# Print the dictionary
all_values = [var_names[key] for key in var_names.keys()]
# exclude variables that are clearly not relevant
not_relevant = ['data_year',
                'infant_id_0',
                'infant_id_1',
                'bord_1',
                'bord_0']

relevant_var_names = var_names.keys() - not_relevant

causes = ['mager8', 
          'gestat10', 
          'meduc6']
# plausible causes that are ordinal/cardinal
confounders = list(set(relevant_var_names) - set(causes))

confounder_values = [var_names[key] for key in confounders]


dfcc = df[list(relevant_var_names) + ['dbirwt_0']]
dfcc = dfcc.rename(columns = {'dbirwt_0': 'Y'})

# Regression of Y on U
U = sm.add_constant(dfcc[confounders])  # Add constant term to the predictors
model_Y_on_U = sm.OLS(dfcc['Y'], U)
results_Y_on_U = model_Y_on_U.fit()
print(results_Y_on_U.summary())

# Regression of Y on X1
X1 = sm.add_constant(dfcc[causes])  # Add a constant term to the predictor
model_Y_on_X1 = sm.OLS(dfcc['Y'], X1)
results_Y_on_X1 = model_Y_on_X1.fit()
print(results_Y_on_X1.summary())

# Regression of Y on X1 and U
X1_U = sm.add_constant(dfcc[[*causes, *confounders]])  # Add constant term to the predictors
model_Y_on_X1_U = sm.OLS(dfcc['Y'], X1_U)
results_Y_on_X1_U = model_Y_on_X1_U.fit()
print(results_Y_on_X1_U.summary())


# save as flow dataset
# clean df
df_flow = dfcc[[*causes, 'Y', *confounders]]

noise = np.random.uniform(0, 1, size=len(df)*3).reshape(-1,3)
df_flow[causes] = df_flow[causes] + noise
# add random noise to the discrete cause variables


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# List of columns to scale
columns_to_scale = [*causes, 'Y']

# Fit the scaler to the data and transform it
df_flow[columns_to_scale] = scaler.fit_transform(df_flow[columns_to_scale])
df_flow.head()

# run regression with rescaled data

# Regression of Y on X1
X1 = sm.add_constant(df_flow[causes])  # Add a constant term to the predictor
model_Y_on_X1 = sm.OLS(df_flow['Y'], X1)
results_Y_on_X1 = model_Y_on_X1.fit()
print(results_Y_on_X1.summary())
regression_file_path = os.path.join(twins_folder, f'regressions_{snippet}.txt')
with open(regression_file_path, 'w') as f:
    f.write("Regression of Y on X1:\n")
    f.write(results_Y_on_X1.summary().as_text())
    f.write("\n\n")  # Adding some space between the two models


# Regression of Y on X1 and U
X1_U = sm.add_constant(df_flow[[*causes, *confounders]])  # Add constant term to the predictors
model_Y_on_X1_U = sm.OLS(df_flow['Y'], X1_U)
results_Y_on_X1_U = model_Y_on_X1_U.fit()
print(results_Y_on_X1_U.summary())

# Appending the second regression results with a header
with open(regression_file_path, 'a') as f:
    f.write("Regression of Y on X1 and U:\n")
    f.write(results_Y_on_X1_U.summary().as_text())

sns.scatterplot(data = df_flow, x = 'gestat10', y = 'Y')

# save to applications folder
df_flow['U'] = pd.factorize(df_flow['hydra'])[0] 

new_names = ['X1', 'X2', 'X3']

# Create a dictionary from zipped lists
name_mapping = dict(zip(causes, new_names))

# Rename the columns using the dictionary
df_flow.rename(columns=name_mapping, inplace=True)


# List all columns that start with 'X'
x_columns = [col for col in df_flow.columns if col.startswith('X')]
df_flow[x_columns + ['Y']].to_csv(twins_folder + f"/data/twins_flow_{snippet}.csv", index=False)

sim_dict = {}
df_flow["Yint"] = df_flow["Y"]
sim_dict["df"] = df_flow
sim_dict["xdim"] = len(causes)
with open(f'{twins_folder}/data/twins_flow_{snippet}.pkl', 'wb') as f:
	pickle.dump(sim_dict, f)

