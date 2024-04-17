# %%
from ultralytics import YOLO
import os
from torchvision.io.image import read_image

# %%
#class ids
'''
names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

'''

# %%
model=YOLO('yolov8x.pt')

# %%
results=model("./VOC/VOCdevkit/VOC2012/JPEGImages/",classes=[5,7],save_crop=True)

# %%
results

# %%
number_list=[f"number: {t}" for t in range(10)]
print(number_list)

# %%
image=read_image('./runs/detect/predict/crops/car/2007_000364.jpg')
print(image.shape)
print(image.shape[1]+image.shape[2])

# %%
num=0
folder_path='./runs/detect/predict2/crops/truck/'
for filename in os.listdir(folder_path):
    img=read_image(folder_path+filename)
    if img.shape[1] + img.shape[2]<200:
        num+=1
        #os.remove(folder_path+filename)
print(num)

# %%
len(os.listdir('./runs/detect/predict/crops/car/'))


