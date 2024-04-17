# %%
#Fiftyoneによるデータセットのダウンロード＆検証用ファイル

# %%
import fiftyone.zoo as foz
import fiftyone as fo
import matplotlib.pyplot as plt

# %%
classes=['Cat','Dog','Car','Cattle','Airplane','Bus','Monkey','Train','Bird','Frog']
datasets=[]
for cls in classes:
    dataset=foz.load_zoo_dataset(
        "open-images-v6",
        split='train',
        label_types=['segmentations'],
        classes=[cls],
        max_samples=500,
        only_matching=True,
        dataset_name=f'OIV6{cls}',
        drop_existing_dataset=True
    )
    dataset.persistent=True
    datasets.append(dataset)

# %%
print(datasets[9])

# %%
dataset=foz.load_zoo_dataset(
    "open-images-v6",
    split='train',
    label_types=['segmentations'],
    classes=['Cat'],
    max_samples=5,
    only_matching=True,
    dataset_name='OIV6-cat-dog'
)

sub_dataset=foz.load_zoo_dataset(
    'open-images-v6',
    split='train',
    label_types=['segmentations'],
    classes=['Dog'],
    max_samples=5,
    only_matching=True,
    dataset_name='dog-subset'
)

# %%
dataset.merge_samples(sub_dataset)

# %%
#dataset.name='OIDV6-CatDogs'
dataset.persistent=True

# %%
for i,dataset in enumerate(datasets):
    dataset.persistent=True
    dataset.export(export_dir=f'../data/OIV6/{classes[i]}',dataset_type=fo.types.ImageSegmentationDirectory,label_field='ground_truth')

# %%
session=fo.launch_app(dataset.view())


