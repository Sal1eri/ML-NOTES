# AI-LAB FOR PERSONAL USE
Learning and experimenting with AI technologies in a personal lab environment.

## 下载数据集

```bash
# 下载全部
python download/download_benchmarks.py --all

# 下载指定数据集（支持短名或全名）
python download/download_benchmarks.py -b cifar10
python download/download_benchmarks.py -b uoft-cs/cifar10

# 指定下载目录
python download/download_benchmarks.py -b cifar10 -d ./data
```

