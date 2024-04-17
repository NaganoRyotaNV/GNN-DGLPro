# %%
import dgl
from dgl.data import DGLDataset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv,MaxPooling
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import pandas
import networkx as nx
%matplotlib inline

# %%
class CIFAR10TrainDataset(DGLDataset):
    def __init__(self,data_path):
        self.data_path = data_path
        super().__init__(name='cifar10_train__gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class CIFAR10TestDataset(DGLDataset):
    def __init__(self,data_path):
        self.data_path = data_path
        super().__init__(name='cifar10_test_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

# %%
traindataset = CIFAR10TrainDataset("../data/NewMyData/train_dist_40_full.dgl")
testdataset = CIFAR10TestDataset("../data/NewMyData/test_dist_40_full.dgl")

# %%
def return_two_list(node_num):
    taikaku = torch.full((node_num,node_num),fill_value=1.)
    for i in range(node_num):
        taikaku[i][i] = 0.
    src_ids = []
    dst_ids = []
    for i in range(node_num):
        for j in range(i,node_num):
            if taikaku[i][j] != 0:
                src_ids.append(i)
                dst_ids.append(j)
                src_ids.append(j)
                dst_ids.append(i)
    tensor_src = torch.tensor(src_ids)
    tensor_dst = torch.tensor(dst_ids)
    return tensor_src,tensor_dst

# %%
num_node_list = [5,10,15,20,25,30,35]
#num_node_list = [5]
graphs = []
labels = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
for node_num in num_node_list:
    print(f'node : {node_num}')
    src,dst = return_two_list(node_num)
    graphs = []
    labels = []
    for graph,label in tqdm(traindataset):
        graph = graph.to(device)
        pool_graph = torch.zeros((node_num,node_num),device=device)
        for p in range(node_num):
            pool_graph[p] = graph.ndata['feat value'][-node_num + p][-node_num:]
        g = dgl.graph((src,dst),num_nodes=node_num,device=device)
        g.ndata['feat value'] = pool_graph
        graphs.append(g)
        labels.append(label)
    output_labels = {'label':torch.tensor(labels)}
    path = f'../data/somedata/train_dist_{node_num}_full.dgl'
    dgl.save_graphs(path,g_list=graphs,labels=output_labels)

        
for node_num in num_node_list:
    print(f'node : {node_num}')
    src,dst = return_two_list(node_num)
    graphs = []
    labels = []
    for graph,label in tqdm(testdataset):
        graph = graph.to(device)
        pool_graph = torch.zeros((node_num,node_num),device=device)
        for p in range(node_num):
            pool_graph[p] = graph.ndata['feat value'][-node_num + p][-node_num:]
        g = dgl.graph((src,dst),num_nodes=node_num,device=device)
        g.ndata['feat value'] = pool_graph
        graphs.append(g)
        labels.append(label)
    output_labels = {'label':torch.tensor(labels)}
    path = f'../data/somedata/test_dist_{node_num}_full.dgl'
    dgl.save_graphs(path,g_list=graphs,labels=output_labels)


# %% [markdown]
# ### 画像を複数枚に分割し各パッチをノードとし、最近傍ノードと接続するグラフを作成

# %%
#テスト。6x6のカラー画像を作成。ただし2x2ごとに単一のカラーに設定。ランダムで。
images = np.random.randint(0,255,(9,9,3))
for i in range(0,9,3):
    for j in range(0,9,3):
        images[i:i+3,j:j+3,:] = np.random.randint(0,255,3,np.uint8)
plt.imshow(images)
plt.show()

# %%
#画像をn分割。テストで3枚に分割。
num_patch=3
patch_width=int(9/3)
data=[]

for i in range(0,9,patch_width):
    for j in range(0,9,patch_width):
        data.append(images[i: i + num_patch,j: j + num_patch, :])

#分割した各パッチを正方形に表示
# 1枚の図を作成
fig = plt.figure()

# 画像を追加
for i in range(9):
    ax = fig.add_subplot(patch_width, patch_width, i+1)
    ax.imshow(data[i])
    ax.axis('off')

# 画像を表示
plt.tight_layout()
plt.show()

# %%
# 正方形の一辺の長さ
side_length = 3

# 0から8までの整数値を持つ正方形のリストを作成
square_list = np.arange(side_length**2).reshape((side_length, side_length))

print(square_list)

# %%
def get_nearest_neighbors(image, row, col):
    # 画像の形状を取得
    height, width = image.shape[:2]

    # 注目画素の周囲8画素の座標を計算
    neighbors_coords = [(row-1, col-1), (row-1, col), (row-1, col+1),
                        (row, col-1), (row, col+1),
                        (row+1, col-1), (row+1, col), (row+1, col+1)]

    # 注目画素の最近傍画素の値を抜き出す
    nearest_neighbors = []
    for r, c in neighbors_coords:
        # 座標が画像範囲内かチェック
        if 0 <= r < height and 0 <= c < width:
            pixel_value = image[r, c]
            nearest_neighbors.append(pixel_value)
        else:
            # 画像範囲外の場合は0を追加するなど適切な処理を行う
            #nearest_neighbors.append(0)
            pass

    return nearest_neighbors

# %%
graph=dgl.DGLGraph()
graph.add_nodes(4)
graph.add_edges([0, 1, 2], [1, 2, 3])

nx_graph=graph.to_networkx()
pos = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
#pos=nx.spring_layout(nx_graph)
nx.draw(nx_graph,pos,with_labels=True,node_color='blue')
plt.show()

# %%
arr=np.random.randint(0,9,(3,3))
indices=np.ndindex(arr.shape)
all_indices=[idx for idx in indices]
rot90_indices=np.array(all_indices).reshape((3,3,2))
print(arr)
print(all_indices)
print(rot90_indices)

# %%
def get_rot90_index(arr):
    row,col=arr.shape
    indices=np.ndindex(arr.shape)#インデックス計算
    all_indices=[idx for idx in indices]#リストに変換
    rot90=np.array(all_indices).reshape((row,col,2))#3,3,2に変形
    rot90=np.rot90(rot90,k=1)#90度回転
    rot90_flatt=rot90.reshape(row*col,2)#フラットに変換
    return rot90_flatt

side_length = 8
#ノード数９のグラフを作成
G=nx.complete_graph(side_length**2)
g=dgl.DGLGraph()
g.add_nodes(side_length**2)

#3x3のリストのインデックスを取得し90回転させたリストを取得
rand_array=np.zeros((side_length,side_length))
rot90=get_rot90_index(rand_array)
print(rot90)
pos={}
for i in range(side_length**2):
    pos[i]=rot90[i]
print(pos)

# 正方形の一辺の長さ
#side_length = 3
# 0から8までの整数値を持つ正方形のリストを作成
square_list = np.arange(side_length**2).reshape((side_length, side_length))

#ノード番号に対応したインデックスを取得
inds=np.ndindex(square_list.shape)
inds=[idx for idx in inds]
#各ノードと最近傍ノード間にエッジを張る
'''for i in range(side_length**2):
    x,y=inds[i]
    #print(x,y)
    flatt_nh=get_nearest_neighbors(square_list,x,y)
    #print(flatt_nh)
    for j in flatt_nh:
        if i == j:
            continue
        else:
            g.add_edges(j,i)'''
nx_g=g.to_networkx()
nx.draw(G,pos,with_labels=True,node_color='blue')
plt.show()

# %%



