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
        self.dim_nfeats=len(self.graphs[0].ndata['f'][0])

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
        self.dim_nfeats=len(self.graphs[0].ndata['f'][0])

    def __getitem__(self, idx):
        if self.transforms == None:
            return self.graphs[idx], self.labels[idx]
        else:
            data=self.transforms(self.graphs[idx])
            return data,self.labels[idx]
        
    def __len__(self):
        return len(self.graphs)

# %%
class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,linear=False):
        super(PatchGCN,self).__init__()
        self.linear_on = linear
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if self.linear_on:
            self.linear_layers=nn.ModuleList([nn.Linear(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            if self.linear_on:
                h=self.linear_layers[i](h)
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')

# %%
data_path=['ndata_8patch.dgl']
config_files=['config2.yaml']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
for data_number in range(len(data_path)):
    print(f'data_number: {data_number}')
    #データ読み込み
    traindataset=STL10TrainDataset(f'../../data/STL10 Datasets/train/{data_path[data_number]}')
    testdataset=STL10TestDataset(f'../../data/STL10 Datasets/test/{data_path[data_number]}')

    #データローダー作成
    num_workers=2
    traindataloader = GraphDataLoader(testdataset,batch_size = 256,shuffle = True,num_workers = num_workers,pin_memory = True)
    testdataloader = GraphDataLoader(traindataset,batch_size = 1000,shuffle = True,num_workers = num_workers,pin_memory = True)

    #設定ファイル読み込み
    with open(f'config/config2.yaml','r') as f:
        config=yaml.safe_load(f)

    #パラメータ設定
    lr=0.0001
    epochs=1500

    #モデルの学習
    for model_name, model_config in config.items():
        #時間計測
        start=time.time()
        #結果を保存するディレクトリを作成
        #Classification/save
        #save_dir=f'../Classification/save/{data_path[data_number]}/config1.yaml/{model_name}'
        save_dir=f'../../Classification/save/{data_path[data_number]}/config2.yaml/{model_name}_linear'
        os.makedirs(save_dir,exist_ok=True)

        loop=True
        loop_num=1

        while loop:
            #モデルの初期化
            linear_on=True
            model=PatchGCN(model_config['input_size'],model_config['hidden_size'],model_config['output_size'],linear_on)
            model.to(device)
            lossF=nn.CrossEntropyLoss()
            optimizer=optim.AdamW(model.parameters(),lr=lr)

            #情報保存用の変数の初期化
            #トレーニング用
            num_correct=0
            num_tests=0
            train_loss_list = []
            train_acc_list = []
            #テスト用
            test_num_correct = 0
            test_num_tests = 0
            best_acc=0
            test_acc_list = []

            loss_index=0
            losss_num=0

            for epoch in tqdm(range(epochs)):
                #トレーニング
                model.train()
                for batched_grapg, labels in traindataloader:
                    batched_grapg = batched_grapg.to(device)
                    labels = labels.to(device)
                    pred = model(batched_grapg,batched_grapg.ndata['f'])
                    loss=lossF(pred,labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    num_correct += (pred.argmax(1) == labels).sum().item()
                    num_tests += len(labels)
                train_loss_list.append(loss.item())
                train_acc_list.append(num_correct / num_tests)
                #カウントリセット
                num_correct=num_tests=0

                #初回のlossを記録
                if epoch == 0:
                    loss_index=loss.item()
                    triger=True
                    losss_num+=1
                #初回以降で最初のlossの値と同値ならカウントプラス1
                if epoch>0 and loss_index==loss.item():
                    if triger==True:
                        losss_num+=1
                #初回以降で最初のlossの値と異なるなら連続していないとする
                elif epoch>0 and loss_index!=loss.item():
                    triger=False
                #初回と同じlossが10回記録されてかつそれが連続している場合学習をストップ
                if triger==True and losss_num==10:
                    loop_num+=1
                    torch.cuda.empty_cache()
                    break

                #テスト
                model.eval()
                for tbatched_graph, tlabels in testdataloader:
                    tbatched_graph = tbatched_graph.to(device)
                    tlabels = tlabels.to(device)
                    tpred = model(tbatched_graph, tbatched_graph.ndata['f'])
                    tpred=F.softmax(tpred)
                    test_num_correct += (tpred.argmax(1) == tlabels).sum().item()
                    test_num_tests += len(tlabels)

                test_acc_list.append(test_num_correct/test_num_tests)
                if best_acc < test_num_correct/test_num_tests:
                    best_acc = test_num_correct/test_num_tests
                    best_weight = model
                #カウントリセット
                test_num_correct=test_num_tests=0

            #完全学習後の正答率の計算(推論)
            with torch.no_grad():
                #情報保存用の変数の初期化
                #トレーニング用
                num_correct=0
                num_tests=0
                save_train_acc=0
                #テスト用
                test_num_correct = 0
                test_num_tests = 0
                save_test_acc=0

                #全トレーニングデータでの正答率計算
                model.train()
                for batched_graph, labels in traindataloader:
                    batched_graph = batched_graph.to(device)
                    labels = labels.to(device)
                    pred = model(batched_graph, batched_graph.ndata['f'])
                    num_correct += (pred.argmax(1) == labels).sum().item()
                    num_tests += len(labels)
                print('Training accuracy:', num_correct / num_tests)
                save_train_acc=(num_correct / num_tests)

                #全テストデータでの正答率
                model.eval()
                for batched_graph, labels in testdataloader:
                    batched_graph = batched_graph.to(device)
                    labels = labels.to(device)
                    pred = model(batched_graph, batched_graph.ndata['f'])
                    test_num_correct += (pred.argmax(1) == labels).sum().item()
                    test_num_tests += len(labels)
                print('Test accuracy:', test_num_correct / test_num_tests)
                save_test_acc=(test_num_correct / test_num_tests)

            #while loop 終了
            loop=False

        #各エポックごとの損失・正答率の記録をモデルごとに.npy形式で保存
        np.save(f'{save_dir}/train_loss_list',train_loss_list)
        np.save(f'{save_dir}/train_acc_list',train_acc_list)
        np.save(f'{save_dir}/test_acc_list',test_acc_list)
        torch.save(model.state_dict(),f'{save_dir}/model_weight.pth')
        torch.save(best_weight.state_dict(),f'{save_dir}/best_model_weight.pth')
        #完全学習後のトレーニング・テストデータそれぞれの正答率を.yaml形式で保存
        log={'train acc':save_train_acc,
            'test acc':save_test_acc,
            'epochs':epochs,
            'config':model_config,
            'best test acc':best_acc,
            'linear_on':str(linear_on),
            'date time':datetime.datetime.now(),
            'run time':time.time() - start}
            
        with open(f'{save_dir}/acc_result.yaml',"w") as f:
            yaml.dump(log,f)

        torch.cuda.empty_cache()


