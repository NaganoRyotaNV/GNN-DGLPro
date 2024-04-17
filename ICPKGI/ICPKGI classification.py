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
from modules import ICPKGIDataset,TrainAccPlot,TrainLossPlot,TestAccPlot,TrainTestAccPlot,TestEmbAccPlot,ClassAcc
from models import PatchGCN



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
object_names = ['airplane','bus','car']  #car bus airplane
setting_file = "config2.yaml"

#設定ファイル読み込み
with open(f'GNN-DGLPro/ICPKGI/configs/{setting_file}','r') as f:
    config = yaml.safe_load(f)


#パラメータ設定
lr = 0.0001
epochs = 1000
get_embedding=True
cos=nn.CosineSimilarity(-1)
for object_name in object_names:
    #データ読み込み
    dataset=ICPKGIDataset(f'GNN-DGLPro/data/ICPKGI/8patch_gray_{object_name}.dgl')

    #各クラスから均等に10個ずつテスト用として抜き出しtrainデータセットとtestデータセットを作成
    labels=[i.item() for _,i in dataset]
    traindataset, testdataset, trainlabels, testlabels=train_test_split(dataset,labels,test_size=0.2,shuffle=True,stratify=labels)

    #データローダー作成
    traindataloader=GraphDataLoader(traindataset,batch_size=1024,shuffle=True,num_workers = 0,pin_memory = True)
    testdataloader=GraphDataLoader(testdataset,batch_size=512,shuffle=True,num_workers = 0,pin_memory = True)

    #全学習を通して変わらない事前情報
    #トレーニングデータのそれぞれのクラス数
    train_label_num=[0]*5
    for i in trainlabels:
        train_label_num[i]+=1

    print(f'object name: {object_name}')
    for model_name, model_config in config.items():
        #時間計測
        start=time.time()
        #結果を保存するディレクトリを作成
        #Classification/save
        #save_dir=f'../Classification/save/{data_path[data_number]}/config1.yaml/{model_name}'
        #save_dir=f'../../Classification/save/embedding/single class/{object_name}/{model_name}'
        save_dir=f'GNN-DGLPro/ICPKGI/save/embedding/single class/{object_name}/{model_name}'
        os.makedirs(save_dir,exist_ok=True)

        #モデルの初期化
        #model=PatchGCN(model_config['input_size'],model_config['hidden_size'],model_config['output_size'])
        model=PatchGCN(model_config['input_size'], model_config['hidden_size'], model_config['output_size'], model_config['num_heads'],embedding=get_embedding)
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
        emb_best_acc=0
        test_acc_list = []
        test_emb_acc_list=[]
        print('epochスタート')
        for epoch in tqdm(range(epochs)):
            train_emb_graphs=[]
            train_emb_labels=[]
            model.train()
            for i,(batched_graph,labels) in enumerate(traindataloader):
                train_emb_labels.extend(labels.tolist())
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                
                pred,emb = model(batched_graph,batched_graph.ndata['f'])
                train_emb_graphs.extend(dgl.unbatch(emb))
                loss=lossF(pred,labels)
                loss_correct+=loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_correct += (pred.argmax(1) == labels).sum().item()
                num_tests += len(labels)
            train_loss_list.append(loss_correct / (i+1))
            train_acc_list.append(num_correct / num_tests)
            #カウントリセット
            num_correct=num_tests=loss_correct=0

            #類似度分類
            direction_graphs=torch.zeros(5,64,model_config['hidden_size'][-1]) #中間層の出力のクラス特徴を保存するリスト
            for emb_graph,emb_label in zip(train_emb_graphs,train_emb_labels):
                #中間層の出力をラベル数で割ってクラスインデックスに加算する
                direction_graphs[emb_label]+=((emb_graph.ndata['emb'])/train_label_num[emb_label]).to('cpu')
            #print(f'direction_graphs shape:{direction_graphs.shape}')
            #print(f'direction_labels:{train_label_num}')
            
            #テスト
            test_emb_graphs=[]
            test_emb_labels=[]
            model.eval()
            for tbatched_graph, tlabels in testdataloader:
                test_emb_labels.extend(tlabels.tolist())
                tbatched_graph = tbatched_graph.to(device)
                tlabels = tlabels.to(device)
                tpred,temb = model(tbatched_graph, tbatched_graph.ndata['f'])
                test_emb_graphs.extend(dgl.unbatch(temb))

                tpred = F.softmax(tpred,dim=1)
                test_num_correct += (tpred.argmax(1) == tlabels).sum().item()
                test_num_tests += len(tlabels)

            test_acc_list.append(test_num_correct/test_num_tests)
            if best_acc < test_num_correct/test_num_tests:
                best_acc = test_num_correct/test_num_tests
                best_weight = model
            #カウントリセット
            test_num_correct=test_num_tests=0
            #類似度計算
            stack_test_emb = torch.stack([g.ndata['emb'].to('cpu') for g in test_emb_graphs],dim=0).unsqueeze(1)
            #print(f'stack_test_emb shape:{stack_test_emb.shape}')
            emb_pred = torch.sum(cos(direction_graphs,stack_test_emb),dim=-1)
            test_emb_correct=(emb_pred.argmax(1)==torch.tensor(test_emb_labels)).sum().item()
            test_emb_acc = test_emb_correct/len(test_emb_labels)
            if emb_best_acc<test_emb_acc: #学習中の一番正答率が高かった時の正答率を保存する
                emb_best_acc=test_emb_acc
            if epoch==(epochs-1):
                class_correct=[0]*5
                class_total=[0]*5
                for i,j in zip(emb_pred.argmax(1),test_emb_labels):
                    class_total[j]+=1
                    if i.item() == j:
                        class_correct[j]+=1
                
            #print(f'test embedding acc:{test_emb_acc*100}%')
            test_emb_acc_list.append(test_emb_acc)

            
        print(f'{epochs} acc : {test_emb_acc*100}')
        #各エポックごとの損失・正答率の記録をモデルごとに.npy形式で保存
        np.save(f'{save_dir}/train_loss_list',train_loss_list)
        np.save(f'{save_dir}/train_acc_list',train_acc_list)
        np.save(f'{save_dir}/test_acc_list',test_acc_list)
        np.save(f'{save_dir}/test_emb_acc_list',test_emb_acc_list)
        torch.save(model,f'{save_dir}/model_weight.pth')
        torch.save(best_weight,f'{save_dir}/best_model_weight.pth')
        #保存したnpyを画像にプロット＆保存
        TrainAccPlot(train_acc_list,save_dir)
        TrainLossPlot(train_loss_list,save_dir)
        TestAccPlot(test_acc_list,save_dir)
        TrainTestAccPlot(train_acc_list,test_acc_list,save_dir)
        TestEmbAccPlot(test_emb_acc_list,save_dir)
        ClassAcc(class_correct,class_total,save_dir)
        #完全学習後のトレーニング・テストデータそれぞれの正答率を.yaml形式で保存
        log={
            'epochs':epochs,
            'config':model_config,
            'best test acc':best_acc,
            'best emb test acc':emb_best_acc,
            'date time':datetime.datetime.now(),
            'run time':time.time() - start}
            
        with open(f'{save_dir}/acc_result.yaml',"w") as f:
            yaml.dump(log,f,sort_keys=False)
        print('\n')
        torch.cuda.empty_cache()
