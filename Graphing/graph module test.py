# %%
import dgl
import torch
import torch.nn as nn
from dgl.nn import GINEConv,GATConv,GraphConv
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
from torchvision.datasets import STL10
from torchvision import transforms

target_size=(224,224)
transform=transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])

STL10_train = STL10("STL10", split='train', download=True, transform=transform)
 
STL10_test = STL10("STL10", split='test', download=True, transform=transform)


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
g=dgl.graph(([0,1,2,1],
             [1,2,1,0]))
in_feat=10
out_feat=2
batch=10
graphs=[]
n_feat=torch.randn(g.num_nodes(),in_feat,in_feat)
e_feat=torch.randn(g.num_edges(),in_feat)
print(n_feat.shape)
print(e_feat.shape)

# %%
dence=nn.Linear(28,10)
input_feat=torch.randn(64,3,28)
conv=GraphConv(10,5)

# %%
pred=dence(input_feat)
pred=conv(pred)

# %%
print(pred.shape)
h=torch.mean(pred,1)
print(h.shape)

# %%
conv=GraphConv(10,10)
dence=nn.Linear(10,10)
pred=conv(g,n_feat)
dpred=dence(pred)

# %%
print(dpred.shape)
g.ndata['h']=dpred
h=dgl.mean_nodes(g,'h')
print(h)

# %%
print(n_feat[0])
print(pred[0])

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
def make_graph(side_length):
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

# %%
st=time.time()
for image,label in tqdm(STL10_train):
    n_feat=image_patch(image,8)
print(time.time()-st)

# %%
print(STL10_train[0][0].max())

# %%
fig = plt.figure()
num_patch=8
patch_width=int(224/num_patch)
# 画像を追加
for i in range(num_patch**2):
    ax = fig.add_subplot(num_patch, num_patch, i+1)
    ax.imshow(n_feat.permute(0,2,3,1)[i])
    ax.axis('off')

# 画像を表示
plt.tight_layout()
plt.show()
print(f'patch num: {num_patch}  patch pic size: {patch_width}  class: {label}')

# %%
print(n_feat)
print(e_feat)
conv=GINEConv(nn.Linear(in_feat,out_feat))
res=conv(g,n_feat,e_feat)
print(res)

# %%
a=torch.randn((10,10,3))
b=torch.randn((10,10,3))
c=a.detach()

# %%
cos=nn.CosineSimilarity(0)
output=torch.cosine_similarity(a.flatten(),b.flatten(),dim=0)
print(output.item())

# %%
g=dgl.graph(([0,1,2,1],
             [1,2,1,0]))
in_feat=2
out_feat=2
n_feat=torch.randn(g.num_nodes(),in_feat)
e_feat=torch.randn(g.num_edges(),in_feat)

# %%
gat1=GATConv(in_feat,10,2)
pred=gat1(g,n_feat,e_feat,get_attention=True)


