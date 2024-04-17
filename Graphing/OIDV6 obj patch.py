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
from torchvision.datasets import STL10,VOCSegmentation
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt

# %%
image=read_image('../data/OIV6/Car/data/000efa99e67d6f0c.jpg')
plt.imshow(image.permute(1,2,0))
plt.show()

# %%
img=F.resize(image,size=(256,256))
plt.imshow(img.permute(1,2,0))
plt.show()

# %%
segment_image=read_image('../data/OIV6/Car/labels/000efa99e67d6f0c.png')
plt.imshow(segment_image.permute(1,2,0))
plt.show()
segment_img=F.resize(segment_image,size=(256,256))
plt.imshow(segment_img.permute(1,2,0))
plt.show()

# %%
out,cnt=torch.unique(segment_img,return_counts=True)
print(out)
print(cnt)

# %%
binari_image=(segment_img > 128).to(torch.uint8)*255
plt.imshow(binari_image.permute(1,2,0))
plt.show()

# %%
out,cnt=torch.unique(binari_image,return_counts=True)
print(out)
print(cnt)
print(f'{(cnt[1]/(256*256))*100}%')

# %%
orig=img.detach().permute(1,2,0)
segorig=binari_image.detach().permute(1,2,0)
num_patch=8
size=orig.shape[0]
print(size)
patch_width=int(size/num_patch)
print(patch_width)
data=[]

for i in range(0,size,patch_width):
    #print(i)
    for j in range(0,size,patch_width):
        #print(j)
        data.append(orig[i: i + patch_width,j: j + patch_width, :])

#分割した各パッチを正方形に表示
# 1枚の図を作成
fig = plt.figure()

# 画像を追加
for i in range(num_patch**2):
    ax = fig.add_subplot(num_patch, num_patch, i+1)
    ax.imshow(data[i])
    ax.axis('off')

# 画像を表示
plt.tight_layout()
plt.show()
print(f'patch num: {num_patch**2}  patch pic size: {patch_width}')

# %%
data=[]
for i in range(0,size,patch_width):
    #print(i)
    for j in range(0,size,patch_width):
        #print(j)
        data.append(segorig[i: i + patch_width,j: j + patch_width, :])

#分割した各パッチを正方形に表示
# 1枚の図を作成
fig = plt.figure()

# 画像を追加
for i in range(num_patch**2):
    ax = fig.add_subplot(num_patch, num_patch, i+1)
    ax.imshow(data[i],vmin=0,vmax=255)
    ax.axis('off')

# 画像を表示
plt.tight_layout()
plt.show()
print(f'patch num: {num_patch}  patch pic size: {patch_width}')

# %%
out,cnt=torch.unique(data[0],return_counts=True)
print(out)
print(cnt)
if len(out)==1:
    if out[0] == 0:
        print('0.0%')
    else:
        print('100.0%')
else:
    print(f'{(cnt[1]/(32*32))*100}%')

# %%
start=time.time()
for d in data:
    out,cnt=torch.unique(d,return_counts=True)
    if len(out)==1:
        if out[0] == 0:
            print('0.0%')
        else:
            print('100.0%')
    else:
        print(f'{(cnt[1]/(32*32))*100}%')
print(f'time: {time.time() - start}')


