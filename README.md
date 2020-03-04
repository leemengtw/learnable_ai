# PyTorch Lightning Applications
> 以 PyTorch Lightning 實作的機器學習與深度學習應用。


## 本地開發

載下 repo：

```bash
export REPO=pytorch_lightning_applications
git clone https://github.com/leemengtaiwan/$REPO.git
cd $REPO
```

建立新的 [Anaconda](https://www.anaconda.com/) 環境並安裝函式庫：

```bash
conda create -n $REPO python=3.6 -y
conda activate $REPO
pip install -r requirements.txt
```

設置 [nbdev](https://github.com/fastai/nbdev) 環境並以可修改的方式安裝此 package：

```bash
nbdev_install_git_hooks
nbdev_build_lib
pip install -e .
nbdev_build_docs
```
