import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import rc
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True, unbiased=False)

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet")
        return data * self.std + self.mean

class PyTorchMLPRegressorFF(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes=(10,), max_iter=500, lr=.005, random_state=None, B=None, idx=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        if B is not None:
            self.input_dim = 2 * self.input_dim  # Adjust input dimension because of sin and cos
            
        layers = [nn.Linear(self.input_dim, hidden_layer_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_layer_sizes)):
            layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], self.output_dim))
        
        self.model = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Make safer (Mariana)
        #self.B = torch.tensor(B, dtype=torch.float32).to(self.device)
        self.B = None
        if B is not None:
            self.B = torch.tensor(B, dtype=torch.float32, device=self.device)

        self.model = self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def input_mapping(self, x):
        if self.B is None:
            return x
        else:
            x_proj = (2. * torch.pi * x) @ self.B.T
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor): 
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):                           
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        X_mapped = self.input_mapping(X)
        # Patch for robustness (Mariana)
        #X_scaled = self.feature_scaler.fit_transform(X_mapped)
        #y_scaled = self.target_scaler.fit_transform(y)
        #X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        #y_scaled = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        X_scaled = self.feature_scaler.fit_transform(X_mapped)
        y_scaled = self.target_scaler.fit_transform(y)
        if not isinstance(X_scaled, torch.Tensor):
            X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        else:
            X_scaled = X_scaled.clone().detach().float()
        if not isinstance(y_scaled, torch.Tensor):
            y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
        else:
            y_scaled = y_scaled.clone().detach().float()
        X_scaled = X_scaled.to(self.device).float()
        y_scaled = y_scaled.to(self.device).float()

        losses = []
        # dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
        # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for i in range(self.max_iter):
            # for X_batch, y_batch in tqdm(data_loader):
            self.optimizer.zero_grad()
            outputs = self.model(X_scaled)
            loss = self.criterion(outputs, y_scaled)
            loss.backward()
            self.optimizer.step()
            if i % 100 == 0:
                print("Epoch", str(i), "Loss", loss.item())
            losses.append(loss.item())
        torch.save(self.model.state_dict(), 'reg_2.2_FF.torch')
        plt.plot(losses)
    
    def fit_load(self, X=None, y=None):
        
        # X_mapped = self.input_mapping(X)
        # X_scaled = self.feature_scaler.fit_transform(X_mapped)
        # y_scaled = self.target_scaler.fit_transform(y)
        X_mean = np.load(os.path.join(dir_path,'X_mean_reg_2.2.1_FF.npy'))
        X_std = np.load(os.path.join(dir_path,'X_std_reg_2.2.1_FF.npy'))
        Y_mean = np.load(os.path.join(dir_path,'Y_mean_reg_2.2.1_FF.npy'))
        Y_std = np.load(os.path.join(dir_path,'Y_std_reg_2.2.1_FF.npy'))

        # Make float (Mariana)
        #self.feature_scaler.mean = torch.tensor(X_mean).to(self.device)
        #self.feature_scaler.std = torch.tensor(X_std).to(self.device)
        #self.target_scaler.mean = torch.tensor(Y_mean).to(self.device)
        #self.target_scaler.std = torch.tensor(Y_std).to(self.device)
        self.feature_scaler.mean = torch.tensor(X_mean, dtype=torch.float32, device=self.device)
        self.feature_scaler.std  = torch.tensor(X_std,  dtype=torch.float32, device=self.device)
        self.target_scaler.mean  = torch.tensor(Y_mean, dtype=torch.float32, device=self.device)
        self.target_scaler.std   = torch.tensor(Y_std,  dtype=torch.float32, device=self.device)


        # X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        # y_scaled = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        # dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32))
        # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        
        # Make it robust to either CPU or CUDA runtimes)\
        #self.model.load_state_dict(torch.load(os.path.join(dir_path,'reg_2.2.1_FF.torch')))
        self.model.to(self.device)  # keep, harmless
        state_dict = torch.load(
            os.path.join(dir_path, 'reg_2.2.1_FF.torch'),
            map_location=self.device  # -> cpu if no CUDA, cuda if available
        )
        self.model.load_state_dict(state_dict)
        print("device:", self.device, "param_device:", next(self.model.parameters()).device)
        self.model.eval()           # optional but nice to have


    def predict(self, X):
        if not isinstance(X, torch.Tensor): X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_mapped = self.input_mapping(X)
        X_scaled = self.feature_scaler.transform(X_mapped)

        # Patch for robustness
        #X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(X_scaled, torch.Tensor):
            X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
        else:
            X_scaled = X_scaled.clone().detach().float()
        X_scaled = X_scaled.to(self.device)

        with torch.no_grad():
            predictions_scaled = self.model(X_scaled)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions.cpu().numpy().squeeze()

def find_closest_to_zero(arr):
    # Check if the array is empty
    if len(arr) == 0:
        return None

    # Check if all elements are positive
    if arr[0] >= 0:
        return arr[0]

    # Check if all elements are negative
    if arr[-1] < 0:
        return arr[-1]

    # Binary search to find the first non-negative number
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] >= 0:
            right = mid - 1
        else:
            left = mid + 1

    # `left` is now the index of the smallest non-negative number
    # Compare the largest negative and the smallest non-negative numbers
    if left > 0 and arr[left] >= 0 and abs(arr[left - 1]) < arr[left]:
        return arr[left - 1]
    else:
        return arr[left]


def load_reg():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 126
    output_dim = 48
    hidden_layer_sizes = (48, 48)

    B = np.load(os.path.join(dir_path,'B_reg_2.2.1_FF.npy'))
    reg = PyTorchMLPRegressorFF(input_dim, output_dim, hidden_layer_sizes, B=B, lr=.005, idx=0)
    # X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    reg.fit_load()
    return reg