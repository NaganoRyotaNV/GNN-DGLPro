# %%
import pandas as pd

# %%
classes_path='../data/classes.csv'
seg_classes_path='../data/segmentation_classes.csv'
output_path='../data/seg_classes_name.csv'

# %%
df1=pd.read_csv(classes_path,header=None,names=['id','name'])
df2=pd.read_csv(seg_classes_path,header=None,names=['id'])

merged_df=df2.merge(df1,on='id',how='left')

merged_df.to_csv(output_path,index=False)


