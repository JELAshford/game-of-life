# Final part of the theoretical work I dd on the train: using this view of 
# GOL with gaussian activations, can I hardcode a ConvNet (with custom activations)
# to perform GOL then see whether it can discover the necessary rule with
# training examples? 
import torch.nn as nn
import torch

from scipy.signal import convolve2d
import numpy as np 

from einops import repeat


class FilteredGaussian(nn.Module):
    def __init__(self, mean, sd, filter):
        super(FilteredGaussian, self).__init__()
        self.mean = torch.Tensor([mean])
        self.sd = torch.Tensor([sd])
        self.filter_val = filter
        
    def forward(self, x):
        activated = (1 / torch.sqrt(torch.Tensor([2]) * torch.pi * self.sd)) * torch.exp((-(x - self.mean)**2) / (2 * self.sd**2))
        filtered = (activated >= self.filter_val).int()
        return filtered
    


class ConvCA(nn.Module):
    
    def __init__(self, M = 0.1, N = 1):
        super(ConvCA, self).__init__()
        # Calculate parameters from kernel definition
        self.gauss_mean = ((5/2) * torch.Tensor([N])) + torch.Tensor([M])
        print(self.gauss_mean)
        self.gauss_const = (1 / torch.sqrt(torch.Tensor([2]) * torch.pi))
        self.grow_thresh = self.gauss_const * torch.exp(-(torch.Tensor([N])**2)/8)
        # Generate the key layers of the model
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same", stride=1)
        self.activation = FilteredGaussian(mean = self.gauss_mean, sd = 1, filter = self.grow_thresh)
        
    def forward(self, x):
        conv_sums = self.conv_layer(x)
        return self.activation(conv_sums)
    


def basic_gol(state):
    counts = convolve2d(state, [[1, 1, 1], [1, 7, 1], [1, 1, 1]], boundary="fill", mode="same")
    return np.isin(counts, [9, 10, 3]).astype(np.uint8)



# Manually setup weights of Conv layer as proof of principles
M = 0.1
N = 1
manual_dict = ConvCA().state_dict()
manual_dict['conv_layer.weight'] = torch.Tensor([[[[N, N, N], [N, M, N], [N, N, N]]]])
manual_dict['conv_layer.bias'] = torch.Tensor([0])
# Create model from this
model = ConvCA()
model.load_state_dict(manual_dict)


# Generate an example set
samples = 5
width = 20
starts = np.random.randint(0, 2, size=(samples, width, width))
next_steps = np.array([basic_gol(start) for start in starts])


# Run through model
model_starts = torch.Tensor(repeat(starts, 'b w h -> b c w h', c = 1))
model_preds = model(model_starts)

# Visualise sample
import matplotlib.pylab as plt

fig, axs = plt.subplots(samples, 3)
axs[0, 0].title.set_text("Original State")
axs[0, 1].title.set_text("True Next")
axs[0, 2].title.set_text("Model Prediction")
for val in range(samples):
    axs[val, 0].imshow(model_starts[val, 0, :, :], cmap="gray")
    axs[val, 1].imshow(next_steps[val, :, :], cmap="gray")
    axs[val, 2].imshow(model_preds[val, 0, :, :], cmap="gray")

plt.show()
