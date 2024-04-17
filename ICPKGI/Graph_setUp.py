import subprocess

def setup_environment(env_name):
    """Conda環境をセットアップし、必要なパッケージをインストールする"""
    
    # 環境を作成
    print(f"Creating a new conda environment named '{env_name}'...")
    subprocess.run(f"conda create -n {env_name} python=3.8 -y", shell=True)

    # 必要なパッケージのリスト
    packages = [
        "numpy",
        "matplotlib",
        "torch",      # PyTorch
        "torchvision", # 画像の前処理やデータセットに必要
        "dgl",         # DGL for graph neural networks
        "networkx",    # Graph visualization
        "tqdm",        # Progress bar
        "pillow",      # PIL for image handling
        "glob2"        # For filename pattern matching
    ]
    
    # パッケージをインストール
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(f"conda install -n {env_name} {package} -c conda-forge -c pytorch -c dglteam -y", shell=True)

    # 環境のアクティベーションとデアクティベーションについての情報を出力
    print("\nTo activate this environment, use:")
    print(f"conda activate {env_name}")
    print("\nTo deactivate an active environment, use:")
    print("conda deactivate")

# 環境名を定義
environment_name = "gnn_dgl_env"

# 環境セットアップ関数を実行
setup_environment(environment_name)
