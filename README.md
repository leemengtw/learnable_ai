# Practical AI
> 以 PyTorch、TensorFlow、scikit-learn 以及 pandas 等函式庫實作的實務機器學習、深度學習應用。


## 本地開發與環境設置

載下 repo：

```bash
export REPO=practical_ai
git clone https://github.com/leemengtaiwan/$REPO.git
cd $REPO
```

建立新的 [Anaconda](https://www.anaconda.com/) 環境並在該環境內安裝函式庫：

```bash
conda create -n $REPO python=3.6 -y
conda activate $REPO  # 或 source activate $REPO
pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge opencv
pip install easydict
pip install more-itertools
```

安裝 Jupyter Lab：

```bash
conda install -c conda-forge jupyterlab -y
pip install ipywidgets
```

安裝 [JupyterLab GPU Dashboards](https://github.com/rapidsai/jupyterlab-nvdashboard)：

```bash
pip install jupyterlab-nvdashboard
jupyter labextension install jupyterlab-nvdashboard
```

設置 [nbdev](https://github.com/fastai/nbdev) 環境並以 editable 的方式安裝此 package：

```bash
nbdev_install_git_hooks
nbdev_build_lib
pip install -e ".[dev]"
```

更新文件：

```bash
nbdev_build_docs
```
