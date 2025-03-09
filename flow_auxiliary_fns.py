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

model_paras_save_folder = "C:/pb/DeconFlow/model_paras_by_trial/"
os.makedirs(model_paras_save_folder, exist_ok=True)


# import PyPDF2
import tempfile


from dependencies import *

from toy_data import *
from get_toy_data import *
from flow_architecture import *


def train_flow_ray(params):
    if socket.gethostname().endswith('compute.internal'):
        # i.e. if we are on AWS server w home folder '/home/ec2-user'
        root_folder = '/home/ec2-user/gip/'
        results_folder = root_folder
        data_folder = root_folder + "data/"
        locat = 'AWS'
    else:
        locat = 'local'
        root_folder = "C:/Users/pfbur/Dropbox/acad_projects/deconflow/"
        os.chdir(root_folder + 'code')
        results_folder = root_folder + 'results/'
        plot_folder = results_folder + 'plots/'
        data_folder = "../data/"
        import PyPDF2
        import tempfile
    plot_during_training = params["plot_during_training"]
    save_interim_results = params["save_interim_results"]
    
    if params["n_classes"] == "true": # note that this is not exactly correct, true number of classes need not be the product
        params["n_classes"] = params["dgp_params"]["no_X_cluster"] * params["dgp_params"]["no_U_cluster"]

    trial_name = ray_train.get_context().get_trial_name()
    if trial_name == None: # this happens when this function is called not within a ray environment, then generate another trial_name
        trial_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    device = params["device"]
    
    '''
    the datasource has to have at least a key 'df',
    which is the pandas dataframe that has columns "Y" (target) and "X1", "X2"... (features)
    '''
    
    if params["datafile_name"] == '_generate_within_ray':
        df, sim_dict = gen_toy_data(data_folder,
                                    string = trial_name,
                                    dgp_params = params["dgp_params"],
                                    save = 1)
        datafile = data_folder + trial_name
    else:
        if params["application"] != "sim": 
            l_data_filename = params["datafile_name"]
            if locat == "local":
                data_folder = f'../application/{params["application"]}/data/'
            if locat == "AWS":
                data_folder = data_folder # same datafolder as simulation data
            datafile = data_folder + l_data_filename
            print(f'datafile is {datafile}')
        else:
            l_data_filename = params["datafile_name"]
            datafile = data_folder + l_data_filename

    with open(datafile + '.pkl', 'rb') as file:
        sim_dict = pickle.load(file)
    
    
    
    if params["application"] == "sim":
        params["dim_XY"] = params["dgp_params"]["xdim"] + 1
    if params["application"] != "sim":
        params["dim_XY"] = sim_dict["xdim"] + 1


    # initialize base distribution
    pi_random = torch.rand(params["n_classes"])
    pi_random_normalized = pi_random / pi_random.sum()
    mu_random = torch.randn((params["n_classes"], params["dim_XY"]))
    sigmas_ones = torch.ones((params["n_classes"], params["dim_XY"])) * .5  
    params["pi_prior"] = torch.exp(pi_random_normalized).to(device)  
    params["mu_prior"] = mu_random.to(device)  
    params["log_var_prior"] = torch.log(sigmas_ones).to(device)  
    
    flow_train_dict = {}
    
    if params["application"] in ("twins"): 
        train_loader, test_loader, train_data, test_data = get_applicat_data(batch_size = params["batch_size"],
                                                                        train_share = .9,
                                                                        filename = datafile) # !!! could edit this so that is directly takes sim_dict as input
    else:
        train_loader, test_loader, train_data, test_data = get_toy_data(batch_size = params["batch_size"],
                                                                    train_share = .9,
                                                                    filename = datafile) # !!! could edit this so that is directly takes sim_dict as input

    flow = Flow(params).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=params["lr_start"]) #1e-5 is too small 
    scheduler = CosineAnnealingLR(optimizer, T_max=round(params["num_epochs"]*1/20), eta_min = params["lr_end"])
    flow = flow.float() # to make sure that the paras are also float32 just like the data
    losses = []
    lr_values = {}
    losses_test = []
    mu_prior_mean = []
    mu_prior_elem = []

    with torch.no_grad(): 
        torch.save(flow.state_dict(), f'{model_paras_save_folder}{trial_name}_epoch0_idx0.pth')
    
    # train flow model
    for epoch in range(params["num_epochs"]):  # Adding an epoch loop
        loss_sum = 0.0

        for idx_step, (X, HL) in enumerate(train_loader):
            X = torch.Tensor(X).to(device = device)
            optimizer.zero_grad()
            loss, _ = flow.forward_transform(X)
            loss = (-loss).mean()
            losses.append(loss.item())
            loss.backward()        
            optimizer.step()
            with torch.no_grad(): 
                loss_test, _ = flow.forward_transform(torch.from_numpy(np.array(test_data)))
                loss_test = (-loss_test).mean()
                losses_test.append(loss_test.item())
                # save intermediate flow parameters
                model_para_path = f'{model_paras_save_folder}{trial_name}'
                if not os.path.exists(model_para_path):
                    os.makedirs(model_para_path)
                torch.save(flow.state_dict(), f'{model_paras_save_folder}{trial_name}/{trial_name}_epoch{epoch+1}_idx{idx_step+1}.pth')
        lr_values[f"lr_epoch{epoch+1}"] = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        if (epoch) % 10 == 0 and plot_during_training == 1:
            print(f"epoch: {epoch:}, loss: {loss.item():.5f}")
            fn_make_plot(flow, 
                         epoch, 
                         loss.item(), 
                         train_data = train_data, 
                         test_data = test_data, 
                         datafile = datafile,
                         time_str = trial_name)
        if (epoch) % 10 == 0: # save info on mu_prior to check
            with torch.no_grad(): 
                mu_prior_mean.append(flow.mu_prior.mean().item())
                mu_prior_elem.append(flow.mu_prior[0,0].item())

    if save_interim_results == 1:        
        flow_train_dict["train_data"] = train_data
        flow_train_dict["test_data"] = test_data
    flow_train_dict["lr_values"] = lr_values
    flow_train_dict["losses"] = losses
    flow_train_dict["losses_test"] = losses_test
    flow_train_dict["mu_prior_mean"] = mu_prior_mean
    flow_train_dict["mu_prior_elem"] = mu_prior_elem
    flow_train_dict["sim_dict"] = sim_dict
    if locat != 'local': # doesnt pickle this when run locally
        flow_train_dict["flow_model"] = flow
        
    
    
    deconf_dict = deconf_permutation(sim_dict, flow)
    metrics = {"loss": loss.item(),
               "deconf_dict": deconf_dict,
               "flow_train_dict": flow_train_dict}

    # save flow object
    if locat != 'local': # doesnt pickle this when run locally
        with open(f'../{trial_name}_flow.pkl', 'wb') as f:
            pickle.dump(flow, f)
            torch.save(flow.state_dict(), f'../{trial_name}_resdict.pth')
        # save flow results dict
    with open(f'../results/{trial_name}_resdict.pkl', 'wb') as f:
        pickle.dump(flow_train_dict, f)
    if locat == 'local': # doesnt pickle this when run locally  
        print(os.getcwd())
        print(f'../results/model_paras')
        torch.save(flow.state_dict(), f'../results/model_paras/{trial_name}_resdict.pth')
    
    # delete data saved on disk
    if params["datafile_name"] == '_generate_within_ray':
        datafile = data_folder + trial_name
        os.remove(f"{datafile}.csv") 
        os.remove(f"{datafile}.pkl") 

    return metrics


def deconf_permutation(sim_dict, flow):
    '''permutation deconfounding: this is the function that implements our
    deconfounding logic
        '''
    df_org = sim_dict["df"]

    x_cols = [col for col in df_org.columns if col.startswith('X')]
    xcfac_cols = [col for col in df_org.columns if col.startswith('cfacXcfac')]
    xdim = len(x_cols)
    # go from df_obs to Z space
    with torch.no_grad():
        _, z = flow.forward_transform(torch.from_numpy(df_org[x_cols + ['Y']].values).float().to(device))
        x = flow.inverse_transform(z)
    zz = z.cpu().detach().numpy().copy()
    
    
    # shuffle Z2
    ally_data = {}
    
    no_draws = 5000 # this is the setting for all results in the paper
    for i in range(no_draws):
        # print(i)
        p = torch.randperm(z.shape[0])
        z[:,-1] = z[p,-1]
        x = flow.inverse_transform(z)
        ally_data[f'shuffled_col_{i}'] = x[:, -1].cpu().detach().numpy()  # Store each column's data in the dictionary
    
    ally = pd.DataFrame(ally_data)
    
    
    xdf = pd.concat([pd.DataFrame(x.cpu().detach().numpy()[:,0:xdim], columns = x_cols), 
                     ally.mean(axis = 1)], axis = 1).reset_index(drop = True)
    xdf = xdf.rename(columns = {0: "Yint_flow"})
    
    
    for i in range(zz.shape[1]):  # Iterate through each column in zz
        column_name = f"Z{i+1}"  # Generate column name dynamically
        xdf[column_name] = zz[:, i]  # Assign the column from zz to the new column in xdf


    
    # Create dataframes from the provided data
    # Linear regression for xdf
    X_xdf = sm.add_constant(xdf[x_cols])  # adding a constant
    model_xdf = sm.OLS(xdf['Yint_flow'], X_xdf).fit()
    model_xdf.summary()
    slope_xdf_statsmodel = model_xdf.params[1:]
    
    # statsmodels implementation for df_org
    if 'Yint' in df_org.columns:
        X_df_org = sm.add_constant(df_org[x_cols])
        model_df_org = sm.OLS(df_org['Yint'], X_df_org).fit()
        slope_df_org_statsmodel = model_df_org.params[1:]
    else:
        slope_df_org_statsmodel = None  

    X_df_org = sm.add_constant(df_org[x_cols])
    model_df_org_naive = sm.OLS(df_org['Y'], X_df_org).fit()
    slope_df_org_statsmodel_naive = model_df_org_naive.params[1:]
    
    # compute MSE between Yint_flow and true Yint
    rmseYYint = (((df_org["Yint"] - df_org["Y"])**2).mean())**.5
    rmse = (((df_org["Yint"] - xdf["Yint_flow"])**2).mean())**.5
    

    ### add individual treatment effects here
    # i) invert xcfac to get Zcfac
    # go from df_obs to Z space
    with torch.no_grad():
        _, zcfac = flow.forward_transform(torch.from_numpy(df_org[xcfac_cols + ['Y']].values).float().to(device))
        xcfac = flow.inverse_transform(zcfac)
    zzcfac = zcfac.cpu().detach().numpy().copy()
    #ii) create Zcfac_tilde: the composition of zzcfac[:,:-1] and z[:,-1]
    zzcfac_tilde = np.concatenate((zzcfac[:, :-1], zz[:, -1].reshape(-1, 1)), axis=1)
    
    # iii) feed zzcfac_tilde thru flow model
    zzcfac_tilde = torch.tensor(zzcfac_tilde, dtype=torch.float32)
    xcfac_tilde = flow.inverse_transform(zzcfac_tilde)
    Ycfac_flow = xcfac_tilde.cpu().detach().numpy().copy()[:,-1]
    
    xdf["Ycfac_flow"] = Ycfac_flow
    rmse_cfac = (((df_org["Ycfac_true"] - xdf["Ycfac_flow"])**2).mean())**.5
    rmse_cfac_naive = (((df_org["Ycfac_true"] - df_org["Ycfac_naive"])**2).mean())**.5


    deconf_dict = {}
    deconf_dict["rmse_Yint_Yint_flow"] = rmse
    deconf_dict["rmse_cfac_flow"] = rmse_cfac
    deconf_dict["rmse_cfac_naive"] = rmse_cfac_naive
    deconf_dict["rmse_Yint_Y"] = rmseYYint
    deconf_dict["sim_Yint_slope"] = slope_df_org_statsmodel.item() if sim_dict["xdim"] == 1 else slope_df_org_statsmodel
    deconf_dict["flow_Yint_slope"] = slope_xdf_statsmodel.item() if sim_dict["xdim"] == 1 else slope_xdf_statsmodel
    deconf_dict["sim_Y_naive"] = slope_df_org_statsmodel_naive.item() if sim_dict["xdim"] == 1 else slope_df_org_statsmodel_naive
    deconf_dict["sim_dict"] = sim_dict
    deconf_dict["df_deconf"] = xdf
    return deconf_dict


def fn_make_plot(flow, epoch, loss, train_data, test_data, datafile, time_str):
    with open(datafile + '.pkl', 'rb') as file:
        # Load the dictionary back
        sim_dict = pickle.load(file)
    print(f'plotting during training')
    with torch.no_grad(): 
        z, x = flow.sample(3000)
    print(f'plotting during training--after sample')
    z = z.cpu().detach().numpy().squeeze()
    x = x.cpu().detach().numpy().squeeze()
    # print(f'x is {x}')
    # print(f'x.shape is {x.shape}')

    x_df = pd.DataFrame(x[:,-2:])
    x_df = x_df.rename(columns = {0: 'Xd', 1: 'Y'})
    z_df = pd.DataFrame(z[:,-2:])
    z_df = z_df.rename(columns = {0: 'Zd-1', 1: 'Zd'})
    ### drop outliers
    # calculate some percentiles
    cutoff = .00005
    percentiles_lo = x_df.quantile(cutoff)
    percentiles_hi = x_df.quantile(1-cutoff)
    
    # Create a mask to identify rows to drop (where any value is outside the desired percentile range)
    mask = (x_df < percentiles_lo) | (x_df > percentiles_hi)
    
    # Drop rows based on the mask (any row with True in any column should be dropped)
    x_filtered = x_df[~mask.any(axis=1)].reset_index(drop = True)
    z_filtered = z_df[~mask.any(axis=1)].reset_index(drop = True)
        
    print(f'samples dropped: {x_df.shape[0]-x_filtered.shape[0]}')
    
    
    ## deconfound to add current state of deconfounding to plot
    deconf_dict = deconf_permutation(sim_dict, flow)

    ## check invertibility
    # go from df_obs to Z space
    with torch.no_grad():
        _, zposterior = flow.forward_transform(torch.from_numpy(train_data.values).float().to(device))
        _, zposterior_test = flow.forward_transform(torch.from_numpy(test_data.values).float().to(device))

        recovered = flow.inverse_transform(zposterior)
        recovered_test = flow.inverse_transform(zposterior_test)
        
    obs_rec = pd.concat([train_data, 
                         pd.DataFrame(recovered.cpu()[:,-2:], columns = ["Xdrec", "Yrec"]),
                         pd.DataFrame(zposterior.cpu()[:,-2:], columns = ["Zd-1", "Zd"])], axis = 1)
    obs_rec_test = pd.concat([test_data.iloc[:,-2:], pd.DataFrame(recovered_test.cpu()[:,-2:], columns = ["X1rec", "Yrec"])], axis = 1)
    
    # Create a figure
    fig = plt.figure(figsize=(18, 18))
    no_rows = 2
    no_cols = 2

    # Add first two subplots with shared axes
    axes1 = fig.add_subplot(no_rows, no_cols, 1)  # 1 row, 3 columns, position 1

    sns.scatterplot(ax=axes1, data=x_filtered, x="Xd", y="Y", alpha=0.3, edgecolor='black', color='blue', label = 'sampled from flow')
    sns.regplot(ax=axes1, data=x_filtered, x="Xd", y="Y", scatter=False, color="blue", label = 'flow_sampled')
    sns.regplot(ax=axes1, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Y", scatter=False, color="red", label = 'observed')
    sns.scatterplot(ax=axes1, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Y", alpha=0.4, edgecolor='red', color='red', label='observed data')
    axes1.set_title('sampled and observed data')

    # Add the second subplot
    axes2 = fig.add_subplot(no_rows, no_cols, 2)  # Position 3
    sns.scatterplot(ax=axes2, data=z_filtered, x="Zd-1", y="Zd", alpha=0.4, edgecolor='black', color='blue', label='sampled data latent space')
    sns.scatterplot(ax=axes2, data=obs_rec, x="Zd-1", y="Zd", alpha=0.4, edgecolor='black', color='green', label='training data latent space')

    mus = pd.DataFrame((flow.mu_prior.cpu()).detach().numpy()[:,-2:], columns=['z1', 'z2'])
    std = pd.DataFrame((flow.log_var_prior.cpu()).exp().sqrt().detach().numpy()[:,-2:], columns=['stdz1', 'stdz2'])
    
    if 'vade' in locals():
        musvade = pd.DataFrame((vade.gmm.means_), columns=['z1', 'z2'])
        stdvade = pd.DataFrame(np.sqrt(vade.gmm.covariances_), columns=['stdz1', 'stdz2'])

    sns.scatterplot(ax=axes2, data=mus, x="z1", y="z2", alpha=1, s = 70, edgecolor='black', color='red', label='mu', marker = 'X')
    if 'vade' in locals():
        sns.scatterplot(ax=axes2, data=musvade, x="z1", y="z2", alpha=1, s = 70, edgecolor='black', color='green', label='mu_vade', marker = 'X')

    axes2.set_title('latent space (sampled and training data)')
    
    
    
    ### add third axis
    axes3 = fig.add_subplot(no_rows, no_cols, 3)  # 1 row, 3 columns, position 1
    if "U" in sim_dict["df"].columns:
        sns.scatterplot(ax=axes3, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Y", hue = "U", label = "flow")
    if "Yint" in sim_dict["df"].columns:
        sns.scatterplot(ax=axes3, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Yint", alpha=0.4, edgecolor='black', color='blue', label = "true")
        sns.regplot(ax=axes3, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Yint", scatter=False, color="blue")
    sns.scatterplot(ax=axes3, data=deconf_dict["df_deconf"], x=f'X{sim_dict["xdim"]}', y="Yint_flow", color='green', label = "flow deconfounded")
    sns.regplot(ax=axes3, data=sim_dict["df"], x=f'X{sim_dict["xdim"]}', y="Y", scatter=False, color="red", label = 'naive regression')
    plt.grid(True)
    sns.regplot(ax=axes3, data=deconf_dict["df_deconf"], x=f'X{sim_dict["xdim"]}', y="Yint_flow", scatter=False, color="green")

    axes3.set_title('true and estimated deconfounded, red is naive')

    
    ### add fourth axis
    axes4 = fig.add_subplot(no_rows, no_cols, 4)  # 1 row, 3 columns, position 1
    if "Yint" in sim_dict["df"].columns:
        sns.scatterplot(ax=axes4, data=deconf_dict["df_deconf"], x=f'Z{sim_dict["xdim"]+1}', y=sim_dict["df"]["U"], alpha=0.9, edgecolor='black', hue = sim_dict["df"]["U"])
        axes4.set_title('true confounder against Z3 (=ZY)')


    # Set overall title
    corxxrec = obs_rec[f'X{sim_dict["xdim"]}'].corr(obs_rec['Xdrec'])
    coryyrec = obs_rec['Y'].corr(obs_rec['Yrec'])
    fig.suptitle(f'epoch = {epoch}, loss = {np.round(loss, 4)}, corxxrec = {np.round(corxxrec, 1)}, coryyrec = {np.round(coryyrec, 1)}')
    plt.tight_layout()
    # plt.savefig(plot_folder + "scatters_interventional_wlatent/model" + time_str + "_scatter_interventionals_wlatent.pdf", bbox_inches='tight', dpi=100)
    img_path = f'{plot_folder}train/{time_str}'
    os.makedirs(img_path, exist_ok=True)
    print(f'img path is {img_path}')
    plt.savefig(f'{img_path}/training_{time_str}_epoch{epoch}.png', bbox_inches='tight', dpi=75)
    # plt.show()


