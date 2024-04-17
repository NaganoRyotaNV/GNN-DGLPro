# %%
#普通に逆伝搬で学習を行うネットワークで学習を行う
#出力層に近い中間層の出力を埋め込み表現としてクラス分類を行う
#そもそもの学習による分類精度も一応示す

# %%
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dgl.nn import GraphConv,SAGEConv
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import seaborn as sns
import random
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import os
import yaml
import time
import datetime

# %%
class ICPKGIDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='ICPKGI_gprah')
    
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
    

class EmbeddingNetwork(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.conv3 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        '''self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)'''
        self.flatt=nn.Flatten()

    def forward(self, g, features):
        '''x=self.flatt(features)
        x = torch.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        #x = self.conv3(g,x)'''
        
        x = self.flatt(features)
        x = torch.relu(self.conv1(g,x))
        x = torch.relu(self.conv2(g,x))
        x = self.conv3(g,x)

        g.ndata['h'] = x
        return g
    

class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,liner=False,embedding = False):
        super(PatchGCN,self).__init__()
        self.embedding=embedding

        self.input_layer=SAGEConv(input_size,hidden_size[0], aggregator_type='mean')
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if liner==True:
            self.liner_on = True
            self.liner_layers=nn.ModuleList([nn.Liner(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        else:
            self.liner_on =False
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            if self.liner_on==True:
                h=self.liner_layers[i](h)
            h=layer(g,h)
            h=self.m(h)
        g.ndata['emb'] = h
        h=self.output_layer(g,h)
        g.ndata['h'] = h

        
        if self.embedding:
            return dgl.mean_nodes(g,'h'),dgl.mean_nodes(g,'emb')
        else:
            return dgl.mean_nodes(g,'h')
    
    
def _train_test_split(data,data_num):
    shuffle_data=random.sample(data,len(data))
    return shuffle_data[:-data_num], shuffle_data[-data_num:]


def TestAccPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Test accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,data.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/test_acc.jpg',dpi=300)
    plt.close()


def TrainAccPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Train accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,data.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/train_acc.jpg',dpi=300)
    plt.close()


def TrainLossPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Train loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_xlim(0,data.shape[0])
    fig.savefig(f'{dir}/train_loss.jpg',dpi=300)
    plt.close()


def TrainTestAccPlot(traindata,testdata,dir):
    traindata=np.array(traindata)
    testdata=np.array(testdata)
    x=[j for j in range(traindata.shape[0])]

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,traindata,label='Train accuracy')
    ax.plot(x,testdata,label='Test accuracy')
    ax.legend()

    ax.set_title('Train & Test accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,traindata.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/train_test_acc.jpg',dpi=300)
    plt.close()

# %%
graphs=[[] for _ in range(5)]
dataset=ICPKGIDataset('../data/ICPKGI/8patch_gray_car.dgl')
labels=[i.item() for _,i in dataset]
print(len(labels))

# %%
traindataset, testdataset, trainlabels, testlabels=train_test_split(dataset,labels,shuffle=True,test_size=50,stratify=labels)

# %%
cn=[0]*5
for i,l in traindataset:
    cn[l.item()]+=1
print(cn)

# %%
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
object_name = 'airplane'  #car bus airplane
setting_file = "config2.yaml"

#データ読み込み
dataset=ICPKGIDataset(f'../data/ICPKGI/8patch_gray_{object_name}.dgl')

#各クラスから均等に10個ずつテスト用として抜き出しtrainデータセットとtestデータセットを作成
labels=[i.item() for _,i in dataset]
traindataset, testdataset, trainlabels, testlabels=train_test_split(dataset,labels,test_size=0.2,shuffle=True,stratify=labels)

#データローダー作成
traindataloader=GraphDataLoader(traindataset,batch_size=512,shuffle=True,num_workers = 0,pin_memory = True)
testdataloader=GraphDataLoader(testdataset,batch_size=10,shuffle=True,num_workers = 0,pin_memory = True)

#設定ファイル読み込み
with open(f'./configs/{setting_file}','r') as f:
    config = yaml.safe_load(f)

#パラメータ設定
lr = 0.0001
epochs = 1000

print(f'object name: {object_name}')
for model_name, model_config in config.items():
    #時間計測
    start=time.time()
    #結果を保存するディレクトリを作成
    #Classification/save
    #save_dir=f'../Classification/save/{data_path[data_number]}/config1.yaml/{model_name}'
    #save_dir=f'../../Classification/save/embedding/single class/{object_name}/{model_name}'
    save_dir=f'./save/embedding/single class/{object_name}/{model_name}'
    os.makedirs(save_dir,exist_ok=True)

    #モデルの初期化
    model=PatchGCN(model_config['input_size'],model_config['hidden_size'],model_config['output_size'])
    model.to(device)
    lossF=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(),lr=lr)

    #情報保存用の変数の初期化
    #トレーニング用
    num_correct=0
    num_tests=0
    loss_correct=0
    train_loss_list = []
    train_acc_list = []
    #テスト用
    test_num_correct = 0
    test_num_tests = 0
    best_acc=0
    test_acc_list = []

    for epoch in tqdm(range(epochs)):
        #トレーニング
        model.train()
        for i,(batched_graph, labels) in enumerate(traindataloader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            pred = model(batched_graph, batched_graph.ndata['f'])
            loss = lossF(pred,labels)
            loss_correct += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        train_loss_list.append(loss_correct / (i+1))
        train_acc_list.append(num_correct / num_tests)
        #カウントリセット
        num_correct=num_tests=loss_correct=0

        #テスト
        model.eval()
        for tbatched_graph, tlabels in testdataloader:
            tbatched_graph = tbatched_graph.to(device)
            tlabels = tlabels.to(device)
            tpred = model(tbatched_graph, tbatched_graph.ndata['f'])
            tpred = F.softmax(tpred,dim=1)
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
        correct_by_class = [0]*5
        total_by_class = [0]*5
        for batched_graph, labels in testdataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            pred=model(batched_graph, batched_graph.ndata['f'])
            pred = F.softmax(pred,dim=1)
            predicted_labels = pred.argmax(1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted_labels[i].item()
                total_by_class[true_label] += 1
                if true_label == predicted_label:
                    correct_by_class[true_label]+=1
            test_num_correct += (pred.argmax(1) == labels).sum().item()
            test_num_tests += len(labels)
        print('Test accuracy:', test_num_correct / test_num_tests)
        class_accuracy = [correct_by_class[i] / total_by_class[i] if total_by_class[i] > 0 else 0 for i in range(5)]
        for i in range(5):
            print(f'Class {i}: Accuracy {class_accuracy[i]:.2%}')
        save_test_acc=(test_num_correct / test_num_tests)

    #各エポックごとの損失・正答率の記録をモデルごとに.npy形式で保存
    np.save(f'{save_dir}/train_loss_list',train_loss_list)
    np.save(f'{save_dir}/train_acc_list',train_acc_list)
    np.save(f'{save_dir}/test_acc_list',test_acc_list)
    torch.save(model,f'{save_dir}/model_weight.pth')
    torch.save(best_weight,f'{save_dir}/best_model_weight.pth')
    #保存したnpyを画像にプロット＆保存
    TrainAccPlot(train_acc_list,save_dir)
    TrainLossPlot(train_loss_list,save_dir)
    TestAccPlot(test_acc_list,save_dir)
    TrainTestAccPlot(train_acc_list,test_acc_list,save_dir)
    #完全学習後のトレーニング・テストデータそれぞれの正答率を.yaml形式で保存
    log={'train acc':save_train_acc,
        'test acc':save_test_acc,
        'epochs':epochs,
        'config':model_config,
        'best test acc':best_acc,
        'date time':datetime.datetime.now(),
        'run time':time.time() - start}
        
    with open(f'{save_dir}/acc_result.yaml',"w") as f:
        yaml.dump(log,f)

    torch.cuda.empty_cache()

# %%
print(pred)

# %%
print(pred.argmax(1))
print(labels)


