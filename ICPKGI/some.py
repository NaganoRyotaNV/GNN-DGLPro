# %%
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss
import numpy as np

# %%
pred = torch.randint(0,3,(4,5))
label = torch.randint(0,3,(4,5))
print(pred)
print(label)

# %%
print(torch.flatten(label))
print(torch.flatten(pred))
print(accuracy_score(torch.flatten(label),torch.flatten(pred)))

# %%
def return_two(a):
    return a,a+1

# %%
a,b = return_two(3)
print(a)


