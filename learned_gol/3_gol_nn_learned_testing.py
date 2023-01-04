# Take the Gol Conv architecture from the hardcoded 
# example and try getting it to learn N, M, mean, 
# and sd from example GOL steps
import torch
import torch.nn as nn 

from einops import repeat, rearrange
from scipy.signal import convolve2d
import numpy as np

import matplotlib.pylab as plt
from tqdm import tqdm



class TunableGaussian(nn.Module):
    def __init__(self):
        super(TunableGaussian, self).__init__()
        self.mean = torch.nn.Parameter(torch.rand(1))
        self.mean.requires_grad = True
        self.sd = torch.nn.Parameter(torch.rand(1))
        self.sd.requires_grad = True
    def forward(self, x):
        gauss_val = (1 / torch.sqrt(torch.Tensor([2]) * torch.pi * self.sd)) * torch.exp((-(x - self.mean)**2) / (2 * self.sd**2))
        return gauss_val
    

class BasicMLP(nn.Module):
    def __init__(self):
        super(BasicMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.model(x)
    

class OriginalCANet(nn.Module):
    
    def __init__(self, activation, M = 0.1, N = 1):
        super(OriginalCANet, self).__init__()
        # Calculate parameters from kernel definition
        self.gauss_mean = ((5/2) * torch.Tensor([N])) + torch.Tensor([M])
        self.gauss_const = (1 / torch.sqrt(torch.Tensor([2]) * torch.pi))
        self.grow_thresh = self.gauss_const * torch.exp(-(torch.Tensor([N])**2)/8)
        # Generate the key layers of the model
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same", stride=1)
        self.activation = activation
        
    def forward(self, x):
        # Get counts
        conv_sums = self.conv_layer(x)
        batch, c, h, w = conv_sums.shape
        # Flatten
        sum_flattened = rearrange(conv_sums, 'b c h w -> b (c h w)')
        # Run through activation function
        activated_flattened = torch.zeros(sum_flattened.shape)
        for ind in range(batch):
            activated_flattened[ind, :] = torch.Tensor([self.activation(x) for x in sum_flattened[ind, :].unsqueeze(1)])
        # Reshape back to images
        output = rearrange(activated_flattened, 'b (c h w) -> b c h w', c=c, h=h, w=w)
        return output 
    
    def hardcode_forward(self, x):
        conv_sums = self.conv_layer(x)
        return ((conv_sums >= 2.1) & (conv_sums <= 3.1)).int()


def train_mapping(model, data, optimiser = None, loss_function = None, epochs=50, learning_rate=3e-4):
    optim = torch.optim.Adam(model.parameters(), learning_rate) if optimiser is None else optimiser
    loss_fn = nn.MSELoss() if loss_function is None else loss_function
    epoch_pb = tqdm(range(epochs))
    for _ in epoch_pb:
        epoch_loss = 0.
        for (X, y) in data:
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss.item()
            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        epoch_pb.set_description(f"loss: {epoch_loss / len(dataset):.3f}")
    return model


def basic_gol(state):
    counts = convolve2d(state, [[1, 1, 1], [1, 7, 1], [1, 1, 1]], boundary="fill", mode="same")
    return np.isin(counts, [9, 10, 3]).astype(np.uint8)


def setup_model_custom_conv_layer(model_class, conv_layer_name, N, M, activation):
    manual_dict = model_class(activation).state_dict()
    manual_dict[f'{conv_layer_name}.weight'] = torch.Tensor([[[[N, N, N], [N, M, N], [N, N, N]]]])
    manual_dict[f'{conv_layer_name}.bias'] = torch.Tensor([0])
    # Create model from this
    model = model_class(activation)
    model.load_state_dict(manual_dict)
    return(model)


EPOCHS = 200
LEARNING_RATE = 3e-4

# Create the mapping samples
test_mean = 2.6
test_sd = 1
start_values = torch.arange(0, 10, 0.05)
end_values = torch.Tensor([1 if (2.1 <= x) & (x <= 3.1) else 0 for x in start_values]) #(1 / torch.sqrt(torch.Tensor([2]) * torch.pi * test_sd)) * torch.exp((-(start_values - test_mean)**2) / (2 * test_sd**2))
dataset = list(zip(start_values.unsqueeze(1), end_values.unsqueeze(1)))


# Create the GOL samples
samples = 1
width = 20
starts = np.random.randint(0, 2, size=(samples, width, width))
next_steps = np.array([basic_gol(start) for start in starts])


### 1 ###
# Test tunability of gaussian (even if we might not need it in the end)
# First, test fitting a gaussian to a dummy of the activation 
# function - want to see if the non-smoothness makes it possible
tg_fit = train_mapping(model = TunableGaussian(), data = dataset, epochs=150)
# Show real and predicted distribution
fig, axs = plt.subplots(1, 2)
axs[0].plot(start_values, end_values)
axs[1].plot(start_values, tg_fit(start_values).detach().numpy())
axs[1].axvline(x=2.1, color='r')
axs[1].axvline(x=3.1, color='r')
plt.show()
# YAY it works


### 2 ###
# Tune an MLP for this task instead - more extensible to more 
# complicated functions
mlp_fit = train_mapping(model = BasicMLP(), data=dataset, epochs=200)
# Confirm results
mlp_preds = mlp_fit(start_values.unsqueeze(1)).detach().numpy().flatten()

fig, axs = plt.subplots(1, 2)
axs[0].plot(start_values, end_values)
axs[1].plot(start_values, mlp_preds)
axs[1].axvline(x=2.1, color='r')
axs[1].axvline(x=3.1, color='r')
plt.show()


### 3 ###
# Insert this tuned model into the hardcoded conv architecture and
# check it still works. 
orig_with_activ = setup_model_custom_conv_layer(OriginalCANet, "conv_layer", N=1, M=0.1, activation=mlp_fit)
model_starts = torch.Tensor(repeat(starts, 'b w h -> b c w h', c = 1))
model_hard_preds = orig_with_activ.hardcode_forward(model_starts)
model_preds = orig_with_activ(model_starts)
# Show the effect of the filtering
fig, axs = plt.subplots(1, 4)
axs[0].imshow(model_starts[0, 0, :, :], cmap="gray")
axs[1].imshow(next_steps[0, :, :], cmap="gray")
axs[2].imshow(model_hard_preds[0, 0, :, :], cmap="gray")
axs[3].imshow(model_preds[0, 0, :, :] >= 0.3, cmap="gray")
plt.show()
# okay - damn, that works for both setups (MLP and TG)!