# %%
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm
import dgl
import networkx as nx
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import glob
from PIL import Image

# %%
target_size=(256,256)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(target_size),
    transforms.Grayscale()
])

# %%
def get_nearest_neighbors(image, row, col): #最近傍ノード番号の取得
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
def make_graph(side_length): #パッチ数分のグラフを作成
    g=dgl.DGLGraph()
    g.add_nodes(side_length**2)
    square_list = np.arange(side_length**2).reshape((side_length, side_length))
    #ノード番号に対応したインデックスを取得
    inds=np.ndindex(square_list.shape)
    inds=[idx for idx in inds]
    #各ノードと最近傍ノード間にエッジを張る
    for i in range(side_length**2):
        x,y=inds[i]
        flatt_nh=get_nearest_neighbors(square_list,x,y)
        for j in flatt_nh:
            if i == j:
                continue
            else:
                g.add_edges(j,i)
    return g
def image_patch(image,num_patch):
    #画像サイズ
    size=image.shape[1]
    #1パッチ当たりの画素数
    patch_width=int(size/num_patch)
    #パッチ保存用配列
    data=[]

    for i in range(0,size,patch_width):
        for j in range(0,size,patch_width):
            data.append(image[:, i : i + patch_width, j : j + patch_width])
    
    return torch.stack(data,dim=0)

# %%
#物体別グラフデータセット作成セル

#パッチ数
num_patch=8

#トレーニングデータセット
graphs=[]
labels=[]
test_mode=False
complete_graph=False
test_mumber=100

object_name = 'airplane'
directions=['front','front side','side','back side','back']
_labels=[]
file_paths=[]
for i,dir in enumerate(directions):
    folder_path  = f'../data/sub/{object_name}/{dir}/*'
    file_names=glob.glob(folder_path)
    file_paths.extend(file_names)
    _labels.extend(i for _ in range(len(file_names)))

'''# フォルダのパスを指定
folder_path = f'../data/sub/{object_name}/**/*'  # フォルダのパスを適切に設定してください

# フォルダ内のファイル名を取得
file_names = glob.glob(folder_path)
print(len(file_names))
_labels=[]
# 取得したファイル名を表示
for file_name in file_names:
    _labels.append(int(file_name[-9]))
#print(_labels)'''


for image,label in tqdm(zip(file_paths,_labels)):
    if test_mode == True:
        if test_mumber < 0:
            break
        else:
            test_mumber -= 1
    image=transform(Image.open(image))
    #画像をパッチに分割
    n_feat=image_patch(image,num_patch)
    #グラフ作成
    if complete_graph:
        g=nx.complete_graph(num_patch**2)
        G=dgl.from_networkx(g)
        G.ndata['f']=n_feat
    else:
        G=make_graph(num_patch)
        #グラフにノード特徴 'f' としてパッチ画像を入力
        G.ndata['f']=n_feat

    #graphsにグラフ labelsにラベルを代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/ICPKGI/{num_patch}patch_gray_{object_name}.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)



# %%
a=[]
a.extend(0 for _ in range(5))
print(a)
a.extend(1 for _ in range(5))
print(a)

# %%
#全物体グラフデータセット作成セル

#パッチ数
num_patch=8

#トレーニングデータセット
graphs=[]
labels=[]
test_mode=False
complete_graph=False
test_mumber=100

object_names = ['airplane','bus','car']
directions=['front','front side','side','back side','back']
_labels=[]
file_paths=[]
for i,dir in enumerate(directions):
    folder_path  = f'../data/sub/{object_name}/{dir}/*'
    file_names=glob.glob(folder_path)
    file_paths.extend(file_names)
    _labels.extend(i for _ in range(len(file_names)))

for image,label in tqdm(zip(file_paths,_labels)):
    if test_mode == True:
        if test_mumber < 0:
            break
        else:
            test_mumber -= 1
    image=transform(Image.open(image))
    #画像をパッチに分割
    n_feat=image_patch(image,num_patch)
    #グラフ作成
    if complete_graph:
        g=nx.complete_graph(num_patch**2)
        G=dgl.from_networkx(g)
        G.ndata['f']=n_feat
    else:
        G=make_graph(num_patch)
        #グラフにノード特徴 'f' としてパッチ画像を入力
        G.ndata['f']=n_feat

    #graphsにグラフ labelsにラベルを代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/ICPKGI/{num_patch}patch_gray_{object_name}.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)

# %%
#パッチ数
num_patch=8

#トレーニングデータセット
graphs=[]
labels=[]
test_mode=False
complete_graph=False
test_mumber=100

object_names = ['airplane','bus','car']
directions=['front','front side','side','back side','back']
'''_labels=[]
file_paths=[]
for i,dir in enumerate(directions):
    folder_path  = f'../data/sub/{object_name}/{dir}/*'
    file_names=glob.glob(folder_path)
    file_paths.extend(file_names)
    _labels.extend(i for _ in range(len(file_names)))'''

for object_label,object_name in tqdm(enumerate(object_names)):
    for direction_label,direction in tqdm(enumerate(directions)):
        folder_path = f'../data/sub/{object_name}/{direction}/*'
        file_names=glob.glob(folder_path)
        for image in file_names:
            image=transform(Image.open(image))
            #画像をパッチに分割
            n_feat=image_patch(image,num_patch)
            #グラフ作成
            if complete_graph:
                g=nx.complete_graph(num_patch**2)
                G=dgl.from_networkx(g)
                G.ndata['f']=n_feat
                G.ndata['d']=torch.tensor([[direction_label]*(num_patch**2)]).reshape(num_patch**2,1)
            else:
                G=make_graph(num_patch)
                #グラフにノード特徴 'f' としてパッチ画像を入力
                G.ndata['f']=n_feat
                G.ndata['d']=torch.tensor([[direction_label]*(num_patch**2)]).reshape(num_patch**2,1)

            #graphsにグラフ labelsにラベルを代入
            graphs.append(G)
            labels.append(object_label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/ICPKGI/{num_patch}patch_gray_all.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)

# %%
i = 2
a=torch.tensor([[i]*(2**2)]).reshape(2**2,1)
print(a)



