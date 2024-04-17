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
from torchvision.datasets import STL10
from torchvision import transforms
 
STL10_train = STL10("STL10", split='train', download=True)
 
STL10_test = STL10("STL10", split='test', download=True, transform=transforms.ToTensor())


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
#点数が最大のときと最小のときの画像・kp・desをそれぞれの手法ごとに保存
akaze=cv2.AKAZE_create()
sift=cv2.SIFT_create()
detect_dict={'akaze':{'time':0,'ave':0,'max':0,'min':0,'most':{'img':0,'kp':0,'des':0},'worst':{'img':0,'kp':0,'des':0}},
             'sift':{'time':0,'ave':0,'max':0,'min':0,'most':{'img':0,'kp':0,'des':0},'worst':{'img':0,'kp':0,'des':0}}}

# %%
#akaze
kps=[]
dess=[]
imgs=[]
for i,j in STL10_train:
    img = np.array(i)
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    imgs.append(img)


# %%
#kpの数が0のときはスキップする
MIN=10000
MAX=-1000
start=time.time()
for i in imgs:
    kp,des=akaze.detectAndCompute(i,None)

    if len(kp)==0:
        continue
    else:
        kps.append(len(kp))
        dess.append(len(des))

    if MIN>len(kp):
        detect_dict['akaze']['worst']['img'] = i
        detect_dict['akaze']['worst']['kp'] = kp
        detect_dict['akaze']['worst']['des'] = des

        MIN=len(kp)

    if MAX<len(kp):
        detect_dict['akaze']['most']['img'] = i
        detect_dict['akaze']['most']['kp'] = kp
        detect_dict['akaze']['most']['des'] = des
        
        MAX=len(kp)

detect_dict['akaze']['time']=time.time() - start
detect_dict['akaze']['ave']=sum(kps)/len(kps)
detect_dict['akaze']['max']=max(kps)
detect_dict['akaze']['min']=min(kps)

# %%
print(detect_dict['akaze'])
print(detect_dict['sift'])

# %%
kpimg=cv2.drawKeypoints(detect_dict['akaze']['most']['img'],detect_dict['akaze']['most']['kp'],None,4)
plt.imshow(kpimg)
plt.savefig('images/akaze_most.jpg',dpi=300)
plt.show()

# %%
kpimg=cv2.drawKeypoints(detect_dict['akaze']['worst']['img'],detect_dict['akaze']['worst']['kp'],None,4)
plt.imshow(kpimg)
plt.savefig('images/akaze_worst.jpg',dpi=300)
plt.show()

# %%
#sift
kps=[]
dess=[]
imgs=[]
for i,j in STL10_train:
    img = np.array(i)
    #img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    imgs.append(img)


# %%
#kpの数が0のときはスキップする
MIN=10000
MAX=-1000
start=time.time()
for i in imgs:
    kp,des=sift.detectAndCompute(i,None)

    if len(kp)==0:
        continue
    else:
        kps.append(len(kp))
        dess.append(len(des))

    if MIN>len(kp):
        detect_dict['sift']['worst']['img'] = i
        detect_dict['sift']['worst']['kp'] = kp
        detect_dict['sift']['worst']['des'] = des

        MIN=len(kp)

    if MAX<len(kp):
        detect_dict['sift']['most']['img'] = i
        detect_dict['sift']['most']['kp'] = kp
        detect_dict['sift']['most']['des'] = des
        
        MAX=len(kp)

detect_dict['sift']['time']=time.time() - start
detect_dict['sift']['ave']=sum(kps)/len(kps)
detect_dict['sift']['max']=max(kps)
detect_dict['sift']['min']=min(kps)

# %%
kpimg=cv2.drawKeypoints(detect_dict['sift']['most']['img'],detect_dict['sift']['most']['kp'],None,4)
plt.imshow(kpimg)
plt.savefig('images/sift_most.jpg',dpi=300)
plt.show()

# %%
kpimg=cv2.drawKeypoints(detect_dict['sift']['worst']['img'],detect_dict['sift']['worst']['kp'],None,4)
plt.imshow(kpimg)
plt.savefig('images/sift_worst.jpg',dpi=300)
plt.show()

# %%
techs=['akaze','sift']
for tech in techs:
    print(tech)
    print(f'time:{detect_dict[tech]["time"]}\nmax:{detect_dict[tech]["max"]}\nmin:{detect_dict[tech]["min"]}\nave:{detect_dict[tech]["ave"]}\n')


