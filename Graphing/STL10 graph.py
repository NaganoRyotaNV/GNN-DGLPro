w# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random
import cv2
from tqdm import tqdm
import dgl
import networkx as nx

# %%
from torchvision.datasets import STL10
from torchvision import transforms
 
STL10_train = STL10("STL10", split='train', download=True, transform=transforms.ToTensor())
 
STL10_test = STL10("STL10", split='test', download=True, transform=transforms.ToTensor())


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
def getpic(img,pos,k_size):#特徴点を中心に画像を抜き出す
    getimgs=[]
    pad_size=k_size//2
    #元画像をpad_size分パディング
    padimg=np.pad(img,pad_size)
    for i in range(len(pos)):
        #パディングした分座標をずらす
        x=int(pos[i].pt[0])+pad_size
        y=int(pos[i].pt[1])+pad_size
        #座標を中心にk_size分抜き出す
        getimg=padimg[x-k_size//2:x+k_size//2+1,y-k_size//2:y+k_size//2+1]
        getimgs.append(getimg)
    getimgs=torch.tensor(getimgs,dtype=torch.float32)

    return getimgs

# %%
#ノードが特徴記述子のトレーニングデータ作成


akaze=cv2.ORB_create()
graphs=[]
labels=[]
size=(512,512)
node_num=100
test_number=100444

for image,label in tqdm(STL10_train):
    #動作テスト
    #if test_number<0:
    #    break
    #else:
    #    test_number-=1

    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp) == 0:
        continue

    #上位抜出し
    dec=np.empty([len(kp)])
    for i in range(len(kp)):
        dec[i]=kp[i].response
    dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


    if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
        top_des=torch.tensor(des,dtype=torch.float32)
        g=nx.complete_graph(len(des))
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des
    else:#kpの数がnode_num以上であればこちら
        #強度が大きい順にノード数分のdesを格納
        top_des=torch.empty(node_num,des[0].shape[0],dtype=torch.float32)
        for i in range(1,node_num+1):
            top_des[i-1]=torch.from_numpy(des[dec_sort_index[-i]]).clone()

        #グラフ作成 networkx -- dgl経由
        g=nx.complete_graph(node_num)
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des

    #graphs代入　labels代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/STL10 Datasets/train/nnum{node_num}_ndatades_enone_orb.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)
    

# %%
#ノードが特徴記述子のテストデータ作成
akaze=cv2.ORB_create()
graphs=[]
labels=[]
size=(512,512)
node_num=100
test_number=10044444
none_num=0
for image,label in tqdm(STL10_test):
    #動作テスト
    #if test_number<0:
    #    break
    #else:
    #    test_number-=1

    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp) == 0:
        none_num+=1
        continue


    #上位抜出し
    dec=np.empty([len(kp)])
    for i in range(len(kp)):
        dec[i]=kp[i].response
    dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


    if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
        top_des=torch.tensor(des,dtype=torch.float32)
        g=nx.complete_graph(len(des))
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des
    else:#kpの数がnode_num以上であればこちら
        #強度が大きい順にノード数分のdesを格納
        top_des=torch.empty(node_num,des[0].shape[0],dtype=torch.float32)
        for i in range(1,node_num+1):
            top_des[i-1]=torch.from_numpy(des[dec_sort_index[-i]]).clone()

        #グラフ作成 networkx -- dgl経由
        g=nx.complete_graph(node_num)
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des

    #graphs代入　labels代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/STL10 Datasets/test/nnum{node_num}_ndatades_enone_orb.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)
    

# %%
#ノードが特徴点を中心とする画像のトレーニングデータ作成
akaze=cv2.AKAZE_create()
graphs=[]
labels=[]
size=(512,512)
node_num=50
k_size=9
test_number=100
test_mode=False

for image,label in tqdm(STL10_train):
    #動作テスト
    if test_mode==True:
        if test_number<0:
            break
        else:
            test_number-=1
    

    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    #img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp) == 0:
        continue

    #上位抜出し
    dec=np.empty([len(kp)])
    for i in range(len(kp)):
        dec[i]=kp[i].response
    dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


    if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
        top_des=getpic(img,kp,k_size)
        g=nx.complete_graph(len(des))
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des
    else:#kpの数がnode_num以上であればこちら
        #強度が大きい順にノード数分のdesを格納
        top_kp=[]
        for i in range(1,node_num+1):
            top_kp.append(kp[dec_sort_index[-i]])
        top_des=getpic(img,top_kp[:node_num],k_size)

        #グラフ作成 networkx -- dgl経由
        g=nx.complete_graph(node_num)
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des

    #graphs代入　labels代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/STL10 Datasets/train/nnum{node_num}_ndatapic{k_size}_enone_akaze.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)
    

# %%
#ノードが特徴点を中心とする画像のテストデータ作成
akaze=cv2.AKAZE_create()
graphs=[]
labels=[]
size=(512,512)
node_num=50
k_size=9
test_number=100
test_mode=False

for image,label in tqdm(STL10_test):
    #動作テスト
    if test_mode==True:
        if test_number<0:
            break
        else:
            test_number-=1
    

    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    #img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp) == 0:
        continue

    #上位抜出し
    dec=np.empty([len(kp)])
    for i in range(len(kp)):
        dec[i]=kp[i].response
    dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


    if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
        top_des=getpic(img,kp,k_size)
        g=nx.complete_graph(len(des))
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des
    else:#kpの数がnode_num以上であればこちら
        #強度が大きい順にノード数分のdesを格納
        top_kp=[]
        for i in range(1,node_num+1):
            top_kp.append(kp[dec_sort_index[-i]])
        top_des=getpic(img,top_kp[:node_num],k_size)

        #グラフ作成 networkx -- dgl経由
        g=nx.complete_graph(node_num)
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des

    #graphs代入　labels代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/STL10 Datasets/test/nnum{node_num}_ndatapic{k_size}_enone_akaze.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)
    

# %%
print(none_num)

# %% [markdown]
# # 以下雑多

# %%
def getpic(img,pos,k_size):#特徴点を中心に画像を抜き出す
    getimgs=[]
    pad_size=k_size//2
    #元画像をpad_size分パディング
    padimg=np.pad(img,pad_size)
    for i in range(len(pos)):
        #パディングした分座標をずらす
        x=int(pos[i].pt[0])+pad_size
        y=int(pos[i].pt[1])+pad_size
        #座標を中心にk_size分抜き出す
        getimg=padimg[x-k_size//2:x+k_size//2+1,y-k_size//2:y+k_size//2+1]
        getimgs.append(getimg)
    getimgs=torch.tensor(getimgs,dtype=torch.float32)

    return getimgs

# %%
akaze=cv2.AKAZE_create()
sift=cv2.SIFT_create()
orb=cv2.ORB_create()
times=[]
kps=np.zeros((3,len(STL10_train)))
dess=np.zeros((3,len(STL10_train)))

# %%
img=cv2.imread('../images/OECU_LOGO.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
kp,des=akaze.detectAndCompute(img,None)
kpimg=cv2.drawKeypoints(img,kp,None,4)
plt.imshow(kpimg)
plt.show()

# %%
print(img.shape)

# %%
c=0
skip=0
s=time.time()
for i,j in tqdm((STL10_train)):
    img=i.numpy().transpose(1,2,0)
    #img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp)==0:
        skip+=1
        continue
    else:    
        kps[0][c]=len(kp)
        dess[0][c]=len(des)
    c+=1
times.append(time.time() - s)


# %%
print(times,len(kps[0]),skip,np.average(kps[0][:139]),np.std(kps[0][:139]),np.max(kps[0][:139]),np.min(kps[0][:139]),np.median(kps[0][:139]))
print(kps[0][4])

# %%
oo=[i for i in range(10)]

# %%
print(oo)
print(oo[:5])

# %%
c=0
s=time.time()
for i,j in tqdm((STL10_train)):
    img=i.numpy().transpose(1,2,0)
    img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=sift.detectAndCompute(img,None)
    kps[1][c]=len(kp)
    dess[1][c]=len(des)
    c+=1
times.append(time.time() - s)


# %%
c=0
s=time.time()
for i,j in tqdm((STL10_train)):
    img=i.numpy().transpose(1,2,0)
    img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=orb.detectAndCompute(img,None)
    kps[2][c]=len(kp)
    dess[2][c]=len(des)
    c+=1
times.append(time.time() - s)

print(times)

# %%
for i in range(3):
    print(np.average(kps[i]),np.std(kps[i]),np.max(kps[i]),np.min(kps[i]),np.median(kps[i]))

# %%
np.save('dess',dess)

# %%
for img,label in tqdm(STL10_train):
    img=img.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=orb.detectAndCompute(img,None)
    break

# %%
print(des[0].shape[0])

# %%
img=STL10_train[0][0]
img = img.permute(1,2,0).numpy()
plt.imshow(img)
plt.show()

# %%
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.imshow(gray,cmap='gray')
plt.show()

# %%
up = cv2.resize(gray,(512,512),interpolation=cv2.INTER_LANCZOS4)
plt.imshow(up,cmap='gray')
plt.show()

# %%
up=cv2.normalize(up,None,0,255,cv2.NORM_MINMAX).astype('uint8')
kp,des=akaze.detectAndCompute(up,None)
kpimg=cv2.drawKeypoints(up,kp,None,4)
plt.imshow(kpimg,cmap='gray')
plt.show()

# %%
node_num=20
dec=np.empty([len(kp)])
for i in range(len(kp)):
    dec[i]=kp[i].response
dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
    top_des=torch.tensor(des,dtype=torch.float32)
    g=nx.complete_graph(len(des))
    G=dgl.from_networkx(g)
    #ノード特徴代入
    G.ndata['feat']=top_des
else:#kpの数がnode_num以上であればこちら
    #強度が大きい順にノード数分のdesを格納
    top_des=torch.empty(node_num,61,dtype=torch.float32)
    top_kp=[]
    for i in range(1,node_num+1):
        top_des[i-1]=torch.from_numpy(des[dec_sort_index[-i]]).clone()
        top_kp.append(kp[dec_sort_index[-i]])

# %%
top_kpimg=cv2.drawKeypoints(up,top_kp,None,4)
plt.imshow(top_kpimg,cmap='gray')
plt.show()

# %%
minipic=getpic(up,top_kp[13].pt,300)
plt.imshow(minipic,cmap='gray')
plt.show()

# %%
print(type(minipic))

# %%
#node featureが特徴点を中心とする画像
akaze=cv2.AKAZE_create()
graphs=[]
labels=[]
size=(512,512)
node_num=20
k_size=10
test_number=10
test_mode=True

for image,label in tqdm(STL10_train):
    #動作テスト
    if test_mode==True:
        if test_number<0:
            break
        else:
            test_number-=1
    

    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    #img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp) == 0:
        continue

    #上位抜出し
    dec=np.empty([len(kp)])
    for i in range(len(kp)):
        dec[i]=kp[i].response
    dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


    if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
        top_des=getpic(img,top_kp,k_size)
        g=nx.complete_graph(len(des))
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des
    else:#kpの数がnode_num以上であればこちら
        #強度が大きい順にノード数分のdesを格納
        top_des=getpic(img,top_kp[:node_num],k_size)

        #グラフ作成 networkx -- dgl経由
        g=nx.complete_graph(node_num)
        G=dgl.from_networkx(g)
        #ノード特徴代入
        G.ndata['feat']=top_des

    #graphs代入　labels代入
    graphs.append(G)
    labels.append(label)

#グラフの保存
output_labels={'label':torch.tensor(labels)}
path=f'../data/STL10 Datasets/train/nnum{node_num}_ndatapic_enone_akaze.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)
    

# %%
graphs[1].ndata['feat'].shape

# %%
ok=[0,0,0]
bad=[0,0,0]
for image,label in tqdm(STL10_test):
    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    #img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=akaze.detectAndCompute(img,None)
    if len(kp)==0:
        bad[0]+=1
    else:
        ok[0]+=1

for image,label in tqdm(STL10_test):
    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=orb.detectAndCompute(img,None)
    if len(kp)==0:
        bad[1]+=1
    else:
        ok[1]+=1
for image,label in tqdm(STL10_test):
    #画像拡大　特徴抽出
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #アルゴリズムがORBかSIFTの場合下一行を実行
    img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
    kp,des=sift.detectAndCompute(img,None)
    if len(kp)==0:
        bad[2]+=1
    else:
        ok[2]+=1

print(f'akaze:total(5000),ok({ok[0]}),bad({bad[0]})')
print(f'orb:total(5000),ok({ok[1]}),bad({bad[1]})')
print(f'sift:total(5000),ok({ok[2]}),bad({bad[2]})')

# %%
start=time.time()
for i in range(20):
    getpic(up,top_kp[i].pt,120)
print(time.time()-start)

# %%
G=nx.complete_graph(node_num)
pos={}
for n,i in enumerate(top_kp):
    pos[n]=i.pt

# %%
nx.draw_networkx(G,pos=pos,with_labels=False)
plt.show()

# %%
print(des[0])

# %%
cars=[]
for i,j in STL10_train:
    if j==2:
        cars.append(i)

# %%
plt.figure(figsize=(12*2,8*2))
for i,img in enumerate(cars[30:]):
    plt.subplot(2,5,i+1)
    img=img.permute(1,2,0)
    plt.imshow(img)
    if i==6:
        true_img=img.numpy()
    if i == 9:
        break
plt.show()

# %%
plt.imshow(true_img)
plt.show()

# %%

true_img=cv2.normalize(true_img,None,0,255,cv2.NORM_MINMAX).astype('uint8')
print(true_img)

# %%
grayimg=cv2.cvtColor(true_img,cv2.COLOR_BGR2GRAY)
true_img=cv2.resize(grayimg,size,interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('car.jpg',true_img)

# %%
img=cv2.imread('car.jpg')
size=(512,512)
akaze=cv2.AKAZE_create()
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
upimg=cv2.resize(grayimg,size,interpolation=cv2.INTER_LANCZOS4)
kp,des=akaze.detectAndCompute(upimg,None)


# %%

node_num=10
dec=np.empty([len(kp)])
for i in range(len(kp)):
    dec[i]=kp[i].response
dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
    top_des=torch.tensor(des,dtype=torch.float32)
    g=nx.complete_graph(len(des))
    G=dgl.from_networkx(g)
    #ノード特徴代入
    G.ndata['feat']=top_des
else:#kpの数がnode_num以上であればこちら
    #強度が大きい順にノード数分のdesを格納
    top_des=torch.empty(node_num,61,dtype=torch.float32)
    top_kp=[]
    for i in range(1,node_num+1):
        top_des[i-1]=torch.from_numpy(des[dec_sort_index[-i]]).clone()
        top_kp.append(kp[dec_sort_index[-i]])

# %%
print(img.dtype)

# %%
kpimg=cv2.drawKeypoints(upimg,top_kp,None,4)
plt.figure(figsize=(12*2,8*2))
plt.imshow(kpimg)
plt.show()

# %%
img=cv2.imread('../images/r_002.png',-1)
index=np.where(img[:,:,3]==0)
img[index]=[255,255,255,255]

size=(512,512)
akaze=cv2.AKAZE_create()
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
upimg=cv2.resize(grayimg,size,interpolation=cv2.INTER_LANCZOS4)
kp,des=akaze.detectAndCompute(upimg,None)

node_num=10
dec=np.empty([len(kp)])
for i in range(len(kp)):
    dec[i]=kp[i].response
dec_sort_index=np.argsort(dec) #大きい順のインデックスを格納した配列


if len(kp)<node_num: #もしkpの数がnode_num以下だった場合ノード数len(kp)個のグラフを作成
    top_des=torch.tensor(des,dtype=torch.float32)
    g=nx.complete_graph(len(des))
    G=dgl.from_networkx(g)
    #ノード特徴代入
    G.ndata['feat']=top_des
else:#kpの数がnode_num以上であればこちら
    #強度が大きい順にノード数分のdesを格納
    top_des=torch.empty(node_num,61,dtype=torch.float32)
    top_kp=[]
    for i in range(1,node_num+1):
        top_des[i-1]=torch.from_numpy(des[dec_sort_index[-i]]).clone()
        top_kp.append(kp[dec_sort_index[-i]])

kpimg=cv2.drawKeypoints(upimg,top_kp,None,4)
plt.figure(figsize=(12*2,8*2))
plt.imshow(kpimg)
plt.show()

# %%
print(type(img))
plt.figure(figsize=(12*2,8*2))
akaze=cv2.AKAZE_create()
size=(512,512)

grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
upimg=cv2.resize(grayimg,size,interpolation=cv2.INTER_LANCZOS4)
kp,des=akaze.detectAndCompute(grayimg,None)
upkp,updes=akaze.detectAndCompute(upimg,None)
print(len(kp),len(upkp))

plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(grayimg,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(upimg,cmap='gray')

plt.show()

# %%
clss=[[],[],[],[],[],[],[],[],[],[]]
for image,label in tqdm(STL10_train):
    img=image.numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp,des=akaze.detectAndCompute(img,None)
    clss[label].append(len(kp))

    

# %%
kpnum=np.zeros(max(clss[0])+1)

# %%
kpnum.shape[0]

# %%
for i in clss[0]:
    kpnum[i] +=1

# %%
x=[i for i in range(kpnum.shape[0])]
plt.bar(x,kpnum)
plt.show()

# %%
plt.figure(figsize=(12*2,8*2))

upimgs=[]
linear=cv2.resize(grayimg,size,interpolation=cv2.INTER_LINEAR)
upimgs.append(linear)
cubic=cv2.resize(grayimg,size,interpolation=cv2.INTER_CUBIC)
upimgs.append(cubic)
nearest=cv2.resize(grayimg,size,interpolation=cv2.INTER_NEAREST)
upimgs.append(nearest)
lanczos4=cv2.resize(grayimg,size,interpolation=cv2.INTER_LANCZOS4)
upimgs.append(lanczos4)


for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(upimgs[i],cmap='gray')
plt.show()


# %%
akaze=cv2.AKAZE_create()
kps=[]
start=time.time()
for i in range(3):
    kp=akaze.detect(STL10_train[i][0].numpy().transpose(1,2,0))
print(time.time()-start)

# %%
kps=[]
size=(512,512)
plt.figure(figsize=(12*2,8*2))
for i in range(3):
    img=STL10_train[i][0].numpy().transpose(1,2,0)
    img=cv2.resize(img,size,interpolation=cv2.INTER_LANCZOS4)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    plt.subplot(1,3,i+1)
    plt.imshow(img,cmap='gray')
    kps.append(akaze.detectAndCompute(img,None)[0])
plt.show()

# %%
print(len(kps[2]))

# %%
g5=nx.complete_graph(5)
print(g5)
g5.add_nodes_from([(1,{'feat':[1,2,32,3,4,5,3452345,43,5,45,23]})])
g5.add_nodes_from([(2,{'feat':200})])
print(nx.get_node_attributes(g5,'feat'))

# %%
nx.draw(g5)
plt.show()

# %%
dglg5=dgl.from_networkx(g5)

# %%
print(dglg5)


