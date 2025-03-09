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


'''
This code was largely developed starting from the simple flow model
implemented by Diedrik Nielsen, available here
https://github.com/probabilisticai/probai-2022/blob/main/day_3/3_didrik/realnvp_solution.ipynb

on forward and inverse naming convention see
https://www.youtube.com/watch?v=bu9WZ0RFG0U&t=1769s
https://youtu.be/bu9WZ0RFG0U?si=nL7XQNG-MgMg0j8F&t=2315
'''

class Flow(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.verbose = 0
        self.params = params
        self.device = params["device"]
        self.to(self.device)

        self.dim_xy = params["dim_XY"]


        bijections = []
        if self.params["flow_type"] == 'nonlinear':
            for i in range(params["no_layers"]+1):
                bijections.append(RestrictedCouplingBijection(make_net(input_dim = math.ceil((self.dim_xy-1)/2), output_dim = 1)))
                bijections.append(CausalTransformation(self.dim_xy, self.device))
                bijections.append(CircularReorderingExceptLastD(self.dim_xy-1))
        if self.params["flow_type"] == 'linear': 
            for i in range(params["no_layers"]+1):
                bijections.append(CausalTransformation(self.dim_xy, self.device))
                bijections.append(CircularReorderingExceptLastD(self.dim_xy-1))
        self.bijections = nn.ModuleList(bijections[:-1])  # Remove the last ReverseBijection 
        self.n_classes = params["n_classes"]   
        self.pi_prior = nn.Parameter(params["pi_prior"]).to(self.device)
        self.mu_prior = nn.Parameter(params["mu_prior"]).to(self.device)
        self.log_var_prior = nn.Parameter(params["log_var_prior"]).to(self.device)
        
    @property
    def base_dist(self):
        if self.n_classes == 1:
            base_dist = Normal( loc=self.mu_prior, scale=self.log_var_prior.exp().sqrt(), )
        else:    
            pi_prior_normalized = softplus(self.pi_prior)/softplus(self.pi_prior).sum()
            logits = torch.log(pi_prior_normalized / (1 - pi_prior_normalized))
            
            # set up GMM        
            mixture_distribution = D.Categorical(logits=logits)
            component_distribution = D.Independent(D.Normal(
                self.mu_prior, self.log_var_prior.exp().sqrt()), 1)
            base_dist = D.MixtureSameFamily(mixture_distribution, component_distribution)
        
        return base_dist
    
    def forward_transform(self, x):
        x = x.to(self.device)  # Move input tensor to the specified device
        log_prob = torch.zeros(x.shape[0], device=self.device)
        for bijection in self.bijections:
            x, ldj = bijection.forward(x)
            log_prob += ldj 
        if self.n_classes == 1:
            log_prob += self.base_dist.log_prob(x).sum(1).to(self.device) 
        else:
            log_prob += self.base_dist.log_prob(x).to(self.device)
        return log_prob, x # should be called z
    
    def inverse_transform(self, z):
        z = z.to(self.device)  # Move input tensor to the specified device
        for bijection in reversed(self.bijections):
            z = bijection.inverse(z)
        return z # should be called x
    
    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,)).float().to(self.device)
        x = self.inverse_transform(z).to(self.device)
        return z, x

class CircularReorderingExceptLastD(nn.Module): 
    def __init__(self, d):
        super().__init__()
        self.d = d  # X dimensions, ie. the number of dimensions that needs to be circled through
    
    def forward(self, x): 
        if x.size(-1) >= self.d:
            # Create a new order for indices
            new_indices = torch.cat((torch.tensor([self.d - 1]), torch.arange(0, self.d - 1), torch.arange(self.d, x.size(-1))))
            x_shuffled = x[..., new_indices]
        else:
            # If x has less than d dimensions, do not modify it
            x_shuffled = x
        return x_shuffled, x.new_zeros(x.shape[0]) 
    
    def inverse(self, z):
        if z.size(-1) >= self.d:
            # Inverse the shuffling by finding the original indices' positions
            original_indices = torch.cat((torch.arange(1, self.d), torch.tensor([0]), torch.arange(self.d, z.size(-1))))
            z_shuffled = z[..., original_indices]
        else:
            # If z has less than d dimensions, do not modify it
            z_shuffled = z
        return z_shuffled


class RestrictedCouplingBijection(nn.Module): 
    '''
    is tailored to clustered deconfounding model
    scale parameter = 0
    only shift parameter is learned
    '''
    def __init__(self, net): 
        super().__init__() 
        self.net = net.to(device)
    def forward(self, x): 
        x_processed, x_last = x[:, :-1], x[:, -1:]
        id, x2 = torch.chunk(x_processed, 2, dim=-1)
        b = self.net(id)
        z2 = x2 + b
        z = torch.cat([id, z2, x_last], dim=-1)
        ldj = torch.zeros_like(b).sum(-1) 
        return z, ldj 
    def inverse(self, z): 
        with torch.no_grad(): 
            z_processed, z_last = z[:, :-1], z[:, -1:]
            id, z2 = torch.chunk(z_processed, 2, dim=-1) 
            b = self.net(id) 
            x2 = (z2 - b)
            x = torch.cat([id, x2, z_last], dim=-1)
            return x



class CausalTransformation(nn.Module):
    def __init__(self, d, device): # d is dimXY
        super(CausalTransformation, self).__init__()
        self.device = device
        
        # Xavier/Glorot Initialization
        std_dev = torch.sqrt(torch.tensor(2. / (d + d)))
        A_init = torch.tril(torch.randn(d, d, device=self.device) * std_dev)
        self.A = nn.Parameter(A_init).to(self.device)


        # Create a mask to keep A lower triangular during updates
        self.mask = torch.tril(torch.ones(d, d, device=self.device))
        if d >= 3:
            self.mask = torch.tril(torch.ones(d, d, device=self.device))
            self.mask[:-1,:-1] = torch.diag(torch.ones(d - 1, device=self.device))

    def forward(self, z):
        self.A.data = self.A.data * self.mask
        z = torch.matmul(z, self.A.T) #+ self.b
        # Compute log determinant of the Jacobian
        ldj = torch.sum(torch.log(torch.abs(torch.diag(self.A))))
        return z, ldj

    def inverse(self, fz):
        # Apply the mask to ensure A is lower triangular
        self.A.data = self.A.data * self.mask
        # Compute the inverse of A
        A_inv = torch.inverse(self.A.T)
        return torch.matmul(fz, A_inv)
    

def make_net(input_dim = 1, output_dim = 2):
    net = nn.Sequential(
        nn.Linear(input_dim, 4), nn.ReLU(),
        # nn.Linear(32, 128), nn.ReLU(),
        # nn.Linear(128, 64), nn.ReLU(),
        # nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(4, output_dim)
    )
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    return net.to(device)



