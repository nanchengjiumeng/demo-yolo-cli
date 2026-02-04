# 激活环境
```bash
# 安装依赖
uv sync

# 激活环境
source ./venv/bin/activate
```

# xanylabeling工具

```bash
# 打开工具
uv run xanylabeling

# 打开项目，开始拉框
uv run xanylabeling --filename ./coco8/images/train --output ./coco8/xlabels/train

# 将YOLO的text转为xanlabeling的json格式
uv run utils/covert_hbb_txt_2_json.py \
  --images coco8/images/train \
  --labels coco8/labels/train \
  --classes coco8/classes.txt 

#  将xanylabeling的json格式转为YOLO的text格式
uv run xanylabeling convert --task xlabel2yolo --mode detect --images ./coco8/images/train --labels ./coco8/xlabels/train \
    --output ./coco8/labels/train --classes coco8/classes.txt 
```

# 预测

```bash
uv run yolo predict model=./runs/detect/yolo26n_coco8_experiment1/weights/best.pt source=./coco8/images/val/000000000036.jpg save=True
```

# 训练

```bash
# 生成yolo的yaml文件
uv run utils/gen_yolo_yaml.py \
  --images coco8/images \
  --labels coco8/labels \
  --classes coco8/classes.txt \
  --output coco8.yaml


# 开始训练
uv run yolo train data='./coco8.yaml' model=yolo26n.pt epochs=10 lr0=0.01

# 开始训练，指定名称
uv run yolo train data='./coco8.yaml' model=yolo26n.pt epochs=10 lr0=0.01 name='yolo26n_coco8_experiment1'

# 中断后继续训练
uv run yolo train model='./runs/detect/yolo26n_coco8_experiment1/weights/last.pt' resume=True
```


# Winodws GPU

1. 核心依赖：安装支持 CUDA 的 PyTorch
```bash
# 先删除现有的 torch（如果是 CPU 版）
uv pip uninstall torch torchvision torchaudio

# 安装支持 CUDA 12.1 的版本 (目前主流)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. 硬件驱动：安装 NVIDIA 驱动

打开 CMD，输入 nvidia-smi。如果能看到显卡型号和驱动版本号，说明驱动没问题。

3. 环境检查：验证是否成功开启 GPU
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

4. 训练时如何指定 GPU
```bash
# 列出可用显卡
uv run python -c "import torch; print(torch.cuda.device_count())"

# 使用 0 号显卡
uv run yolo train model=yolo26n.pt data=coco8.yaml device=0

# 如果有多个显卡
uv run yolo train model=yolo26n.pt data=coco8.yaml device=0,1
```