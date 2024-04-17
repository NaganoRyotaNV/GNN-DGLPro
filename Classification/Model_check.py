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
import time
import datetime
import pandas as pd
from torchinfo import summary

# %%
class STL10TrainDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='stl10_train_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入
        self.dim_nfeats=len(self.graphs[0].ndata['feat'][0])

    def __getitem__(self, idx):
        if self.transforms == None:
            return self.graphs[idx], self.labels[idx]
        else:
            data=self.transforms(self.graphs[idx])
            return data,self.labels[idx]
    def __len__(self):
        return len(self.graphs)


class STL10TestDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='stl10_test_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入
        self.dim_nfeats=len(self.graphs[0].ndata['feat'][0])

    def __getitem__(self, idx):
        if self.transforms == None:
            return self.graphs[idx], self.labels[idx]
        else:
            data=self.transforms(self.graphs[idx])
            return data,self.labels[idx]
        
    def __len__(self):
        return len(self.graphs)

# %%
'''class DynamicGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(DynamicGCN,self).__init__()
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat).clamp(0)
        for layer in self.middle_layers:
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h).clamp(0)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')'''
class DynamicGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(DynamicGCN,self).__init__()
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat).clamp(0)
        for layer in self.middle_layers:
            h=layer(g,h).clamp(0)
        h=self.output_layer(g,h).clamp(0)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')
    

class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,linear=False):
        super(PatchGCN,self).__init__()
        self.linear_on = True
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if self.linear_on:
            self.linear_layers=nn.ModuleList([nn.Linear(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat).clamp(0)
        for i,layer in enumerate(self.middle_layers):
            if self.linear_on:
                skip=h
                h=self.linear_layers[i](h)
                h=h+skip
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h).clamp(0)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')

# %%
batch_size=512
summary(model=PatchGCN(2352,[2000,2000,1000],10),input_size=(batch_size,64,3,28,28))

# %%
#transform = transforms.Compose([transforms.Normalize(0,1)])
traindataset=STL10TrainDataset('../data/STL10 Datasets/train/nnum20_ndatapic9_enone_akaze.dgl')
testdataset=STL10TestDataset('../data/STL10 Datasets/test/nnum20_ndatapic9_enone_akaze.dgl')

# %%
if os.name =='posix':
    num_workers = 2
else:
    num_workers = 0
num_workers = 0
traindataloader = GraphDataLoader(traindataset,batch_size = 512,shuffle = True,num_workers = num_workers,pin_memory = True)
testdataloader = GraphDataLoader(testdataset,batch_size = 1000,shuffle = True,num_workers = num_workers,pin_memory = True)
print(f'num_wokers = {num_workers}')
print(os.name)

# %%
#テストラスト
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path='save/nnum20_ndatapic9_enone_akaze.dgl/config1-2.yaml/model1/model_weight.pth'
print(os.path.isfile(model_path))
model=torch.load(model_path,map_location=torch.device('cpu'))
model.to(device)

test_num_correct = 0
test_num_tests = 0
save_test_acc=0
with torch.no_grad():
    #全テストデータでの正答率
    model.eval()
    for batched_graph, labels in testdataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'])
        test_num_correct += (pred.argmax(1) == labels).sum().item()
        test_num_tests += len(labels)
    print('Test accuracy:', test_num_correct / test_num_tests)
    save_test_acc=(test_num_correct / test_num_tests)


# %%
#テストベスト
model_path='save/nnum20_ndatapic9_enone_akaze.dgl/config1-2.yaml/model1/best_model_weight.pth'
print(os.path.isfile(model_path))
model=torch.load(model_path,map_location=torch.device('cpu'))
model.to(device)

test_num_correct = 0
test_num_tests = 0
save_test_acc=0
with torch.no_grad():
    #全テストデータでの正答率
    model.eval()
    for batched_graph, labels in testdataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat'])
        test_num_correct += (pred.argmax(1) == labels).sum().item()
        test_num_tests += len(labels)
    print('Test accuracy:', test_num_correct / test_num_tests)
    save_test_acc=(test_num_correct / test_num_tests)


# %%
#config -1系のデータ処理
#読み込みデータのパス指定
data_name=['nnum20_ndatapic9_enone_akaze.dgl',
           'nnum20_ndatapic21_enone_akaze.dgl',
           'nnum50_ndatapic9_enone_akaze.dgl',
           'nnum50_ndatapic21_enone_akaze.dgl']
data=np.zeros((8,10))
for i,dname in enumerate(data_name): #データセット選択
    print(dname)
    testdataset=STL10TestDataset(f'../data/STL10 Datasets/test/{dname}')
    testdataloader = GraphDataLoader(testdataset,batch_size = 1000,shuffle = True,num_workers = num_workers,pin_memory = True)
    for j in range(10): #model指定
        data_path=f'save/{dname}/model{j+1}/'
        model_path=f'{data_path}/model_weight.pth'
        yaml_path=f'{data_path}/acc_result.yaml'
        with open(yaml_path,'r') as f:
            config=yaml.safe_load(f)
        
        #yamlからtrainの値を読み込み代入
        data[i*2][j]=float(config['train acc'])

        #モデルを読み込みテストデータで推論し値を代入
        model=torch.load(model_path)
        model.to(device)
        
        test_num_correct = 0
        test_num_tests = 0
        save_test_acc=0
        with torch.no_grad():
            #全テストデータでの正答率
            model.eval()
            for batched_graph, labels in testdataloader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                pred = model(batched_graph, batched_graph.ndata['feat'])
                test_num_correct += (pred.argmax(1) == labels).sum().item()
                test_num_tests += len(labels)
            save_test_acc=(test_num_correct / test_num_tests)
        data[i*2+1][j]=save_test_acc
#データフレームを作成しdataを代入してcsvで出力
index=['20-9 train','20-9 test','20-21 train','20-21 test','50-9 train','50-9 test','50-21 train','50-21 test']
columns=['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']
df=pd.DataFrame(data=data,index=index,columns=columns)
df.to_csv('check-1.csv')

# %%
#config -2系のデータ処理
#読み込みデータのパス指定
data_name=['nnum20_ndatapic9_enone_akaze.dgl',
           'nnum20_ndatapic21_enone_akaze.dgl',
           'nnum50_ndatapic9_enone_akaze.dgl',
           'nnum50_ndatapic21_enone_akaze.dgl']
data=np.zeros((8,10))
for i,dname in enumerate(data_name): #データセット選択
    print(dname)
    testdataset=STL10TestDataset(f'../data/STL10 Datasets/test/{dname}')
    testdataloader = GraphDataLoader(testdataset,batch_size = 1000,shuffle = True,num_workers = num_workers,pin_memory = True)
    for j in range(10): #model指定
        data_path=f'save/{dname}/config{(i%2)+1}-2.yaml/model{j+1}/'
        model_path=f'{data_path}/model_weight.pth'
        yaml_path=f'{data_path}/acc_result.yaml'
        with open(yaml_path,'r') as f:
            config=yaml.safe_load(f)
        
        #yamlからtrainの値を読み込み代入
        data[i*2][j]=float(config['train acc'])

        #モデルを読み込みテストデータで推論し値を代入
        model=torch.load(model_path)
        model.to(device)
        
        test_num_correct = 0
        test_num_tests = 0
        save_test_acc=0
        with torch.no_grad():
            #全テストデータでの正答率
            model.eval()
            for batched_graph, labels in testdataloader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                pred = model(batched_graph, batched_graph.ndata['feat'])
                test_num_correct += (pred.argmax(1) == labels).sum().item()
                test_num_tests += len(labels)
            save_test_acc=(test_num_correct / test_num_tests)
        data[i*2+1][j]=save_test_acc
#データフレームを作成しdataを代入してcsvで出力
index=['20-9 train','20-9 test','20-21 train','20-21 test','50-9 train','50-9 test','50-21 train','50-21 test']
columns=['model1','model2','model3','model4','model5','model6','model7','model8','model9','model10']
df=pd.DataFrame(data=data,index=index,columns=columns)
df.to_csv('check-2.csv')

# %%
for i in range(4):
    print((i%2)+1)


