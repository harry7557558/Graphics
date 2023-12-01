import torch
import numpy as np
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    pass

model = torch.load('model1c.pth',
                   map_location=torch.device('cpu'))

state_dict = model.state_dict()

tensors = []
for key, tensor in state_dict.items():
    if 'bias' in key: continue
    tensor = tensor.detach().cpu().numpy()
    tensors.append((key, tensor.reshape(-1)))
concat = np.concatenate([t[1] for t in tensors])
xlim = [np.amin(concat), np.amax(concat)]
xlim = [-np.amax(np.abs(concat)), np.amax(np.abs(concat))]

grid_nx = int(1.5*np.ceil(len(tensors)**0.5))
grid_ny = int(np.ceil(len(tensors)/grid_nx))
fig, axs = plt.subplots(grid_ny, grid_nx)
for i in range(len(tensors)):
    key, weight = tensors[i]
    ax = axs[i//grid_nx, i%grid_nx]
    ax.set_title(key)
    ax.hist(weight, density=True)
    #ax.hist(weight, density=True, log=True)
    ax.set_xlim(xlim)


#plt.hist(np.concatenate(concat), bins=50, density=True)
plt.show()
