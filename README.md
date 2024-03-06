## 配置安全帽识别环境

首先确保环境与代码一致：

```bash
Python>=3.7
Pytorch==1.5.x
```

创建虚拟环境

```bash
conda create --name helmet-cpu python=3.7
```

创建成功后我们激活它：

```bash
conda activate helmet-cpu
```

接下来在 helmet-cpu 这个虚拟环境安装 PyTorch == 1.5.1 版本：

```bash
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.mirrors.ustc.edu.cn/simple
```

接下来在 helmet-cpu 这个虚拟环境安装 requirements.txt 里面的依赖：

```bash
cd <代码目录>

pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
```

接下来在 在 helmet-cpu 这个虚拟环境执行执行对象检测任务：

```bash
python3 detect.py 
```

