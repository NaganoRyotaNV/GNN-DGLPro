# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random
import cv2
from tqdm import tqdm
import dgl
import networkx as nx
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision import transforms
import torchvision.transforms.functional as F

# %%
target_size=(224,224)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1)
])

STL10_train = STL10("STL10", split='train', download=True, transform=transform)
 
STL10_test = STL10("STL10", split='test', download=True, transform=transform)


# %%
traindataloader = DataLoader(dataset=STL10_train,batch_size=512,shuffle=True)

# %%
def img2graph(images,src,dst):
    batch_size=images.shape[0]
    feat=torch.randn(64*batch_size,20)
    batched_graph=dgl.batch([dgl.graph((src,dst)) for _ in range(batch_size)])
    batched_graph.ndata['f']=feat
    return batched_graph

# %%
num_nodes=64
edges=[]
for i in range(num_nodes):
    for j in range(i+1,num_nodes):
        edges.append((i,j))
src,dst=zip(*edges)
i=0
for image,label in tqdm(traindataloader):
    batched_graph=img2graph(image,src,dst)
    print(batched_graph)
    if i >10:
        break
    else:
        i+=1

# %%
print(batched_graph.ndata['f'])


