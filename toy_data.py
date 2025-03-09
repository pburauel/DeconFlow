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

# import PyPDF2
import tempfile


from dependencies import *




############### define iResNet as an invertible NN
class SimpleResNetBlock(nn.Module):
    def __init__(self, dim):
        super(SimpleResNetBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return identity + out

class SimpleResNet(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 1, num_blocks = 1):
        super(SimpleResNet, self).__init__()
        self.initial_linear = nn.Linear(input_dim, input_dim)
        self.blocks = nn.ModuleList([SimpleResNetBlock(input_dim) for _ in range(num_blocks)])
        self.final_linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.initial_linear(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)
        return x
    


def gen_toy_data(data_folder, string, dgp_params, save=1):
    # Extract parameters from dgp_params dictionary
    xdim = dgp_params.get('xdim', 1)  # Default to 1 if not specified
    n_samples = dgp_params.get('n_samples', 100)  # Default to 100 if not specified
    no_X_cluster = dgp_params.get('no_X_cluster', 2)  # Default to 2 if not specified
    no_U_cluster = dgp_params.get('no_U_cluster', 2)  # Default to 2 if not specified
    phiX_piecewise_affine = dgp_params.get('phiX_piecewise_affine', 0)  # Default to 0 if not specified
    phiY_piecewise_affine = dgp_params.get('phiY_piecewise_affine', 0)  # Default to 0 if not specified
    psiY_piecewise_affine = dgp_params.get('psiY_piecewise_affine', 0)  # Default to 0 if not specified
    no_resnet_blocks = dgp_params.get('no_resnet_blocks', 5)


    betaX = np.random.choice([-1,1]) # true causal effect of X on Y
    # betaX = 1
    
    ### define mu_l and sigma_l
    Lloc = np.random.uniform(-1,1,no_U_cluster)
    # alternative settings for additional experimentation
    # Lloc = np.linspace(1, -1, no_U_cluster)
    # Lloc = [3, -1]
    
    ### define mu_h and sigma_h
    Hloc = np.random.uniform(1, 4, no_X_cluster)
    # alternative settings for additional experimentation
    # Hloc = np.linspace(1, 4, no_X_cluster)
    # Hloc = [1, 4]
    
    # set covariances
    cov_l = 0.5
    cov_h = 0.5
    
    var_UY = .01
    mus = np.array(np.meshgrid(Lloc, Hloc)).T.reshape(-1, 2)
    sigmas = np.tile([cov_l, cov_h], (mus.shape[0], 1))
    
    def generalized_random_generation(no_X_cluster, no_U_cluster, equal_proba = 0):
        if equal_proba == -1:
            M = np.array([[0.45, 0.25],
                    [0.1, 0.2]])
        elif equal_proba == 1:
            equal_probability = 1 / (no_X_cluster * no_U_cluster)
            M = np.full((no_U_cluster, no_X_cluster), equal_probability)
        else:
            # The total number of segments is no_X_cluster * no_U_cluster
            total_segments = no_X_cluster * no_U_cluster
            
            # Generate cut points along with 0 and 1, ensuring we have total_segments + 1 points
            cut_points = np.sort(np.random.uniform(0, 1, total_segments - 1))
            cut_points = np.concatenate(([0], cut_points, [1]))
            
            # Compute differences between successive numbers to get segment lengths, i.e., probabilities
            probabilities = np.diff(cut_points)
            
            # Reshape these probabilities into a no_U_cluster x no_X_cluster matrix
            M = probabilities.reshape((no_U_cluster, no_X_cluster))
        
        # Check if M is a valid probability matrix
        assert np.all(M >= 0) and np.isclose(M.sum(), 1), "M is not a valid probability matrix"
        return M
    
    
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

    
    MI = 0
    while MI < .02:
        M = generalized_random_generation(no_X_cluster, no_U_cluster, equal_proba = 0)
        MI = compute_mutual_information(M)

    
    assert np.all(M >= 0) and np.isclose(np.sum(M), 1), "Invalid joint probability matrix"
    
    # Compute the marginal probabilities of L and H
    P_L = np.sum(M, axis=1)
    P_H = np.sum(M, axis=0)
    
    def sample_L_H_from_joint_distribution(M, n_samples):
        """
        Generalized function to sample L and H from a joint distribution represented by matrix M.
        
        Parameters:
        - M: A probability matrix of shape (no_U_cluster, no_X_cluster).
        - n_samples: Number of samples to generate.
        
        Returns:
        - L: An array of sampled low-level cluster indices.
        - H: An array of sampled high-level cluster indices.
        """
        # The total number of elements in M represents all possible combinations of L and H
        HL = np.random.choice(range(M.size), size=n_samples, p=M.flatten())
    
        # Calculate the number of clusters for H from the shape of M
        no_X_cluster = M.shape[1]
    
        # Decode L and H from HL
        L = HL // no_X_cluster  # Determines the cluster of L
        H = HL % no_X_cluster   # Determines the cluster of H
        
        return L, H, HL

    L, H, HL = sample_L_H_from_joint_distribution(M, n_samples = n_samples)
    
    
    MI = compute_mutual_information(M)
    
    def generate_Z1(n_samples, xdim, H, Hloc, cov_h): # Z1 dim is equal to xdim
        # Initialize X with zeros
        Z1 = np.zeros((n_samples, xdim))
        expanded_Hloc = np.repeat(Hloc, xdim)
        
        # Create a covariance matrix for the multivariate distribution
        cov_matrix = cov_h * np.eye(len(expanded_Hloc))
        # Generate XX with expanded Hloc and the covariance matrix
        Z1Z1 = np.random.multivariate_normal(mean=expanded_Hloc, cov=cov_matrix, size=n_samples)
        
        # Step 3: Allocate values to X based on H
        for i in range(len(Hloc)):
            for j in range(xdim):
                Z1[:, j][H == i] = Z1Z1[:, i*xdim + j][H == i]
        
        return Z1, Z1Z1
    
    Z1, Z1Z1 = generate_Z1(n_samples, xdim, H, Hloc, cov_h)
    num_rows = Z1.shape[0]
    
    '''
    generate a permuted version of Z1
    we will feed this through the same kind of transformation as Z1
    because we want to have a ground truth for the individual treatment
    effect estimation down the line
    '''
    permutation = np.random.permutation(num_rows)
    Z1cfac = Z1.copy()[permutation, :]
    Z1Z1cfac = Z1Z1.copy()[permutation, :]


    def generate_Z2(n_samples, L, Lloc, cov_l):
        """
        Generalizes the process of generating C, a confounder influenced by the cluster variable L.
        
        Parameters:
        - n_samples: The number of samples.
        - L: An array of cluster indices for L.
        - Lloc: The location parameters for each cluster in L.
        - cov_l: The common scale parameter for the normal distributions.
        
        Returns:
        - C: The generated confounder values.
        """
        # Initialize Z2
        Z2 = np.zeros((n_samples,1))
        Z2Z2 = np.random.multivariate_normal(mean = Lloc, cov = cov_l * np.eye(len(Lloc)), size = n_samples)

        for i in range(len(Lloc)):
            Z2[L == i] = Z2Z2[L == i,i].reshape(-1,1)#np.random.normal(loc=Hloc[i], scale=np.sqrt(cov_h), size=np.sum(H == i)).reshape(-1,1)            
        return Z2.reshape(-1,), Z2Z2
    
    
    
    Z2, Z2Z2 = generate_Z2(n_samples, L, Lloc, cov_l)    

    # error on Y
    UY = np.random.normal(0,var_UY, size=Z1.shape[0])
    
    expanded_betaX = np.repeat(betaX, xdim)

    Y = UY
    
    # add nonlinear transformation of Z2
    psiY = Z2
    psiYY = Z2Z2
    psiYlinear = Z2.copy()
    
    if psiY_piecewise_affine == 1:
        model = SimpleResNet(num_blocks = no_resnet_blocks)
        output = model(torch.tensor(psiY.reshape(-1,1)).float())
        psiY = output.detach().numpy().reshape(-1,)
        # Calculate variance of phiYlinear and phiY
        var_psiYlinear = np.var(psiYlinear)
        var_psiY = np.var(psiY)
    
        # Scale phiY to match the variance of phiYlinear
        scaling_factor = np.sqrt(var_psiYlinear / var_psiY)
        psiY = psiY * scaling_factor
    
        sns.jointplot(x=psiYlinear.reshape(-1,), y=psiY, kind='scatter', marginal_kws=dict(bins=30, fill=True))

        # feed (each dimension, i.e. for each L = l) Z2Z2 through the same network and scale similarly
        psiYY = Z2Z2.copy()

        for dim in range(Z2Z2.shape[1]):
            # Reshape the data and feed it through the network
            output = model(torch.tensor(Z2Z2[:, dim].reshape(-1,1)).float())
            # Detach and reshape the network output
            psiYtemp = output.detach().numpy().reshape(-1,)
            
            # Calculate variance of the output
            var_psiYtemp = np.var(psiYtemp)
            
            # Calculate the scaling factor and scale the output
            scaling_factor_temp = np.sqrt(var_psiYlinear / var_psiYtemp)
            
            # Store the scaled output
            psiYY[:, dim] = psiYtemp * scaling_factor


            
    Y += psiY

    phiY = Z1 @ expanded_betaX
    phiYlinear = phiY.copy()
    
    phiYcfac = Z1cfac @ expanded_betaX
    phiYlinearcfac = phiYcfac.copy()
    if phiY_piecewise_affine == 1: 
        model = SimpleResNet(input_dim = Z1.shape[1], 
                             num_blocks=no_resnet_blocks,
                             output_dim = 1)
        output = model(torch.tensor(Z1.copy().reshape(-1,Z1.shape[1])).float())
        phiY = output.detach().numpy().reshape(-1,)

        output_cfac = model(torch.tensor(Z1cfac.copy().reshape(-1,Z1.shape[1])).float())
        phiYcfac = output_cfac.detach().numpy().reshape(-1,)        # Calculate variance of Z1 and phiY
        
        var_phiYlinear = np.var(Z1)
        var_phiY = np.var(phiY)
        # Scale phiY to match the variance of Z1
        scaling_factor = np.sqrt(var_phiYlinear / var_phiY)
        phiY = phiY * scaling_factor
        phiYcfac = phiYcfac * scaling_factor

        
    Ycfac = Y.copy() # because Z1 has not been added to Y yet, we can just use Y to construct cfac from there
    Y += phiY
    Ycfac += phiYcfac
    
    # compute true interventional distribution
    Yint = psiY.mean() + phiY
    
    Ycfac_naive = Y[permutation] # the naive cfac estimate is the observed Y for the cfac X value

    # Create a DataFrame for easier plotting
    df_dict = {
        'H': H,
        'L': L,
        'HL': HL,
        'Y': Y,
        'Ycfac_true': Ycfac,
        'Ycfac_naive': Ycfac_naive, 
        'Yint': Yint,
        'U': Z2
    }
    
    phiX = Z1.copy()
    phiXcfac = phiX.copy()
    if phiX_piecewise_affine == 1:
        model = SimpleResNet(input_dim = Z1.shape[1], 
                             num_blocks = no_resnet_blocks,
                             output_dim = Z1.shape[1])
        output = model(torch.tensor(Z1.copy().reshape(-1,Z1.shape[1])).float())
        phiX = output.detach().numpy().reshape(-1,Z1.shape[1])
        output_cfac = model(torch.tensor(Z1cfac.copy().reshape(-1,Z1.shape[1])).float())
        phiXcfac = output_cfac.detach().numpy().reshape(-1,Z1.shape[1])        # Calculate variance of phiYlinear and phiY
        var_phiX = np.var(phiX)
        # Scale phiY to match the variance of phiXlinear        
        scaling_factor = np.sqrt(np.var(Z1) / var_phiX)
        phiX = phiX * scaling_factor
        phiXcfac = phiXcfac * scaling_factor
        sns.jointplot(x=phiX.reshape(-1,), y=Z1.reshape(-1,), kind='scatter', marginal_kws=dict(bins=30, fill=True))
    
    for i in range(xdim):
        df_dict[f'X{i+1}'] = phiX[:, i] 
        df_dict[f'cfacXcfac{i+1}'] = phiXcfac[:, i] 
    df = pd.DataFrame(df_dict)
    

    # if scaling = 1: scale observed X here, and save scaler
    scaling = 1
    if scaling == 1:
        columns_to_scale = [col for col in df.columns if col.startswith('X') or col == 'Y']
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        xcols = [col for col in df.columns if col.startswith('X')]
        df[xcols] = scaler.fit_transform(df[xcols])
        cfacxcols = [col for col in df.columns if col.startswith('cfacX')]
        df[cfacxcols] = scaler.fit_transform(df[cfacxcols])
        scalerY = MinMaxScaler(feature_range=(0, 1))
        scalerY.fit(np.array(df['Y']).reshape(-1, 1))
        df['Y'] = scalerY.transform(np.array(df['Y']).reshape(-1, 1))
        df['Ycfac_true'] = scalerY.transform(np.array(df['Ycfac_true']).reshape(-1, 1))
        df['Ycfac_naive'] = scalerY.transform(np.array(df['Ycfac_naive']).reshape(-1, 1))        
        df['Yint'] = scalerY.transform(np.array(df['Yint']).reshape(-1, 1))
    
    
    
    # Regress Y on X1 (Short regression)
    x_cols = [col for col in df.columns if col.startswith('X')]

    short_reg = LinearRegression().fit(df[x_cols], df['Y'])
    short_slope = short_reg.coef_[0]
    
    # Regress Y on X1 and C (Long regression)
    long_reg = LinearRegression().fit(df[x_cols + ['U']], df['Y'])
    long_slope = long_reg.coef_[0]  # Slope coefficient for X1
    
    print(f'short, long: {round(short_slope,2), round(long_slope,2)}')
    
    # save data
    print(df.head())
    
    filename = f'{data_folder}{string}'#toy_data_caus' + str(betaX) + '_conf' + str(confounding)
    print(filename)
    # save data including M matrix
    sim_dict = {'df': df,
                'M': M,
                'mus': mus,
                'sigmas': sigmas,
                'var_UY': var_UY,
                'xdim': xdim}
    if save == 1:
        # save it
        df.to_csv(filename + '.csv', index=False)
        with open(filename + '.pkl', 'wb') as file:
            # Serialize and save the dictionary
            pickle.dump(sim_dict, file)
    return df, sim_dict
