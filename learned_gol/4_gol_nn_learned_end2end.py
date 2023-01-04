# Now we've demonstrated it works in principle (3), learn GOL
# end-to-end with either activation
import torch
import torch.nn as nn 

from torch.utils.data import TensorDataset, DataLoader

from einops import repeat, rearrange
from scipy.signal import convolve2d
import numpy as np

import matplotlib.pylab as plt
from tqdm import tqdm

from sklearn.metrics import roc_curve


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
    
    def __init__(self, activation):
        super(OriginalCANet, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same", stride=1)
        self.activation = activation
        
    def forward(self, x):
        # Get counts
        conv_sums = self.conv_layer(x)
        _, c, h, w = conv_sums.shape
        # Flatten
        sum_flattened = rearrange(conv_sums, 'b c h w -> b (c h w)')
        # Run through activation function
        activated_flattened = torch.clamp(self.activation(sum_flattened), 0, 1)
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
        epoch_pb.set_description(f"loss: {epoch_loss / len(data):.3f}")
    return model


def basic_gol(state):
    counts = convolve2d(state, [[1, 1, 1], [1, 7, 1], [1, 1, 1]], boundary="fill", mode="same")
    return np.isin(counts, [9, 10, 3]).astype(np.uint8)


def generate_gol_dataset(samples=10000, edge_size=20, to_tensor_batch=True):
    starts = np.random.randint(0, 2, size=(samples, edge_size, edge_size))
    next_steps = np.array([basic_gol(start) for start in starts])
    if to_tensor_batch:
        starts = torch.Tensor(repeat(starts, 'b w h -> b c w h', c=1))
        next_steps = torch.Tensor(repeat(next_steps, 'b w h -> b c w h', c=1))
    return starts, next_steps


# Generate input/output pairs for GOL
train_starts, train_next_steps = generate_gol_dataset(samples=10000, edge_size=20)
gol_train_ds = TensorDataset(train_starts, train_next_steps)
gol_train_dl = DataLoader(gol_train_ds, batch_size=128, shuffle=True)

test_starts, test_next_steps = generate_gol_dataset(samples=10000, edge_size=20)
gol_test_ds = TensorDataset(test_starts, test_next_steps)
gol_test_dl = DataLoader(gol_test_ds, batch_size=128, shuffle=True)


# Create and train the model
model = OriginalCANet(activation = TunableGaussian())
loss = nn.BCELoss()
out = train_mapping(model, gol_train_ds, loss_function=loss, epochs=100)

# Test on the test dataset
test_preds = model(test_starts)
fpr, tpr, thresh = roc_curve(test_next_steps.detach().numpy().flatten(), test_preds.detach().numpy().flatten())
plt.scatter(fpr, tpr, s=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# Show an example
sample = 2
fix, axs = plt.subplots(1, 4)
axs[0].imshow(test_starts[sample, 0, :, :].detach().numpy(), cmap="gray")
axs[1].imshow(test_next_steps[sample, 0, :, :].detach().numpy(), cmap="gray")
axs[2].imshow(test_preds[sample, 0, :, :].detach().numpy(), cmap="gray")
axs[3].imshow(test_preds[sample, 0, :, :].detach().numpy() >= thresh[2], cmap="gray")
plt.show()


# Show a trajectory
steps = 3
start = np.random.randint(0, 2, size=(20, 20))
true_grid, model_grid = start.copy(), start.copy()

with torch.no_grad():
    true_traj, model_traj = [start], [start]
    for step in range(steps):
        true_grid = basic_gol(true_grid)
        model_grid_tensor = model(torch.Tensor(repeat(model_grid, 'w h -> b c w h', b=1, c=1)))
        model_grid = (model_grid_tensor.detach().numpy() >= thresh[2])[0,0,:,:]
        
        true_traj.append(true_grid.copy())
        model_traj.append(model_grid.copy())

true_traj = np.array(true_traj) 
model_traj = np.array(model_traj) 

fig, axs = plt.subplots(steps, 2)
for step in range(steps):
    axs[step, 0].imshow(true_traj[step, :, :], cmap="gray")
    axs[step, 1].imshow(model_traj[step, :, :], cmap="gray")
plt.show()

print((true_grid == model_grid).all())
