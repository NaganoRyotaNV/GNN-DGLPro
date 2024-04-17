# %%
import os

# %%

folder_path = '../data/airplane/back/'  # 対象のフォルダパスを指定してください
num = 1
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):  # 拡張子に合わせて変更してください
        original_path = os.path.join(folder_path, filename)
        new_filename = f'4_{str(num).zfill(3)}.jpg'   # 先頭一文字を 'X' に変更する例
        new_path = os.path.join(folder_path, new_filename)

        os.rename(original_path, new_path)
        print(f"{filename} のファイル名を {new_filename} に変更しました")
        num+=1


# %%
a = 1
print(str(a).zfill(3))


