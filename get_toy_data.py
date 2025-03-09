import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

from dependencies import *



def get_toy_data(batch_size, train_share, filename):
    # Load the data from the CSV file
    df = pd.read_csv(filename + '.csv')
    
    # Convert the DataFrame to PyTorch tensors
    H = torch.tensor(df['H'].values, dtype=torch.long)  # H is a categorical variable
    L = torch.tensor(df['L'].values, dtype=torch.long)  # L is a categorical variable
    HL = torch.tensor(df['HL'].values, dtype=torch.long)  # L is a categorical variable
    
    # Create a list to hold the X tensors
    X_tensors = []
    
    for col in df.columns:
        if col.startswith('X'):
            X_tensors.append(torch.tensor(df[col].values, dtype=torch.float32))
    
    Y = torch.tensor(df['Y'].values, dtype=torch.float32)
    
    # Combine the X tensors and Y into a single tensor
    X = torch.stack(X_tensors + [Y], dim=1)
        
    # Combine the tensors into a single dataset
    dataset = TensorDataset(torch.tensor(X).float(), HL)
    
    # Define the size of the train and test datasets
    train_size = int(train_share * len(dataset))
    test_size = len(dataset) - train_size

    # Create the train and test datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Now you can create data loaders for the train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # put the test data in a dataframe
    all_batches = []
    for batch in test_loader:
        X, _ = batch  # Ignore labels for now
        df_batch = pd.DataFrame(X.numpy())  # Convert tensor to numpy if necessary
        all_batches.append(df_batch)
    test_data = pd.concat(all_batches, ignore_index=True)
    num_cols = test_data.shape[1]  # Get the number of columns
    new_col_names = [f"X{i}" for i in range(1, num_cols)] + ["Y"]  # Generate new column names
    test_data.columns = new_col_names


    # put the traindata in a dataframe
    all_batches = []
    for batch in train_loader:
        X, _ = batch  # Ignore labels for now
        df_batch = pd.DataFrame(X.numpy())  # Convert tensor to numpy if necessary
        all_batches.append(df_batch)
    train_data = pd.concat(all_batches, ignore_index=True)
    train_data.columns = new_col_names  # Rename the columns
    
    
    return train_loader, test_loader, train_data, test_data


def get_applicat_data(batch_size, train_share, filename):
    with open(filename + '.pkl', 'rb') as file:
        sim_dict = pickle.load(file)

    # df_obs = sim_dict["df"][["X1", "Y"]]
    df_obs = sim_dict["df"][sim_dict["df"].columns[sim_dict["df"].columns.str.startswith('X') | (sim_dict["df"].columns == 'Y')]]
    X = torch.tensor(df_obs.values, dtype=torch.float32).float()
    dummy_labels = torch.zeros(df_obs.shape[0], 0)  # Creates a tensor of shape [n_samples, 0]
    dataset = TensorDataset(X, dummy_labels)
    
    # Define the size of the train and test datasets
    train_size = int(train_share * len(dataset))
    test_size = len(dataset) - train_size

    # Create the train and test datasets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Now you can create data loaders for the train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # put the test data in a dataframe
    all_batches = []
    for batch in test_loader:
        X, _ = batch  # Ignore labels for now
        df_batch = pd.DataFrame(X.numpy())  # Convert tensor to numpy if necessary
        all_batches.append(df_batch)
    test_data = pd.concat(all_batches, ignore_index=True)
    test_data = test_data.rename(columns = {0: "X1", 1: "Y"})


    # put the traindata in a dataframe
    all_batches = []
    for batch in train_loader:
        X, _ = batch  # Ignore labels for now
        df_batch = pd.DataFrame(X.numpy())  # Convert tensor to numpy if necessary
        all_batches.append(df_batch)
    train_data = pd.concat(all_batches, ignore_index=True)
    train_data = train_data.rename(columns = {0: "X1", 1: "Y"})
    
    
    return train_loader, test_loader, train_data, test_data

