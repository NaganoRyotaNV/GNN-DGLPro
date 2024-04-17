# %%
import dgl
from dgl.data import DGLDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.nn import GraphConv,MaxPooling
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import time
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import yaml

#from torchviz import make_dot
from IPython.display import display
%matplotlib inline

# %%
root_path='save/ndata_8patch.dgl/config2.yaml/'
data_num=5
#Test acc plot
for i in range(data_num):
    data=np.load(f'{root_path}model{i+1}_linear_skip/test_acc_list.npy')
    x=[i for i in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Test acc')
    ax.set_xlabel('epochs')
    ax.set_ylabel('acc')
    ax.set_xlim(0,data.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{root_path}model{i+1}_linear_skip/test_acc.jpg',dpi=300)
    plt.close()


# %%
root_path='save/ndata_8patch.dgl/config2.yaml/'
data=np.load(f'{root_path}model2_linear_skip/train_loss_list.npy')
x=[i for i in range(data.shape[0])]
y=data
plt.plot(x,y)
plt.show()

# %%
for i in range(10):
    print(y[i])

# %%
def count_different_elements(lst, value):
    count = 0
    for element in lst:
        if element == value:
            count += 1
    return count
tv=y[0]
print(count_different_elements(y,tv))

# %%
comp=True
while comp:
    print('start while')
    for i in range(5000):
        a=torch.randn(1)
        if a<0:
            break
        comp=False


