import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import yaml
import datetime

from models import PatchGCN,PatchGAT,PatchGATv2,PatchGCN2
import modules


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
    

def main():
    #初期化
    data_paths=['ndata_8patch_gray.dgl']
    config_paths=['gray test config.yaml']
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    #データセット別ループ
    for data_path in data_paths:
        #データセット読み込み
        if not os.path.exists(f'GNN-DGLPro/data/STL10 Datasets/train/{data_path}'):
            print(f'{data_path} is nothing.')
            sys.exit()
        traindataset = STL10TrainDataset(f'GNN-DGLPro/data/STL10 Datasets/test/{data_path}')
        testdataset = STL10TestDataset(f'GNN-DGLPro/data/STL10 Datasets/train/{data_path}')

        print(traindataset[0])
        #データローダー作成
        num_workers=0
        traindataloader = GraphDataLoader(traindataset,batch_size = 1024,shuffle = True,num_workers = num_workers,pin_memory = True)
        testdataloader = GraphDataLoader(testdataset,batch_size = 64,shuffle = True,num_workers = num_workers,pin_memory = True)

        #設定ファイル読み込み
        config_path='gray test config.yaml'
        with open(f'GNN-DGLPro/Classification/Patch train/config/{config_path}','r') as f:
            config = yaml.safe_load(f)

        #ハイパラ
        lr=0.0001
        epochs=50

        #学習推論開始
        for model_name, model_config in config.items():
            #初期設定
            loop=True
            loop_num=1
            print(model_name)

            linear_on=True
            
            if linear_on:
                save_dir=rf'GNN-DGLPro/Classification/save/{data_path}/GATConv/{config_path}/{model_name}_linear'
            else:
                save_dir=rf'GNN-DGLPro/Classification/save/{data_path}/GATConv/{model_name}'
            os.makedirs(save_dir,exist_ok=True)

            #lossが10回以内に変化するまでモデルの初期化を繰り返す
            while loop:
                print(f'loop: {loop_num}')
                start = time.time()
                model=PatchGAT(model_config['input_size'], model_config['hidden_size'], model_config['output_size'], model_config['num_heads'], linear_on)
                model.to(device)
                lossF=nn.CrossEntropyLoss()
                optimizer=optim.AdamW(model.parameters(), lr=lr)

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

                previous_value = None
                consecutive_count = 0

                for epoch in tqdm(range(epochs)):
                    model.train()
                    for batched_graph, labels in traindataloader:
                        batched_graph =batched_graph.to(device)
                        labels = labels.to(device)
                        pred = model(batched_graph,batched_graph.ndata['f'])
                        #print(f'pred shape: {pred.shape}')
                        loss=lossF(pred,labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        num_correct += (pred.argmax(1) == labels).sum().item()
                        num_tests += len(labels)
                    train_loss_list.append(loss.item())
                    train_acc_list.append(num_correct / num_tests)
                    num_correct=num_tests=0
                    
                    #10回連続でlossの値が変わらない場合ループを抜けて初めから学習しなおす
                    if loss.item() == previous_value:
                        consecutive_count += 1
                        if consecutive_count == 10 and epoch < 12:
                            print("10 consecutive values are the same. Exiting the loop.")
                            break
                    else:
                        consecutive_count = 1

                    previous_value = loss.item()


                    #学習途中でのテストデータの正答率計算
                    model.eval()
                    with torch.no_grad():
                        for tbatched_graph, tlabels in testdataloader:
                            tbatched_graph = tbatched_graph.to(device)
                            tlabels = tlabels.to(device)
                            tpred = model(tbatched_graph, tbatched_graph.ndata['f'])
                            tpred= F.softmax(tpred,dim=0)
                            test_num_correct += (tpred.argmax(1) == tlabels).sum().item()
                            test_num_tests += len(tlabels)

                    test_acc_list.append(test_num_correct/test_num_tests)
                    if best_acc < (test_num_correct/test_num_tests):
                        best_acc = (test_num_correct/test_num_tests)
                        best_weight = model
                    #カウントリセット
                    test_num_correct=test_num_tests=0

                #学習完了後の正答率の計算
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
                        pred = F.softmax(pred,dim=0)
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
                        pred = F.softmax(pred,dim=0)
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
                'linear_on':str(linear_on),
                'date time':datetime.datetime.now(),
                'run time':time.time() - start}
                
            with open(f'{save_dir}/acc_result.yaml',"w") as f:
                yaml.dump(log,f)
            
            print('\n')
            torch.cuda.empty_cache()

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


if __name__ == '__main__':
    '''
    想定データ
    データセット：複数
    設定ファイル：複数
    '''
    main()