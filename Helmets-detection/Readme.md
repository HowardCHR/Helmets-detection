# YOLO 头盔检测系统

基于 YOLOv8 的头盔检测系统，支持模型训练和图片检测，提供 Streamlit 可视化界面。

## 功能特点

- 🎯 **高效检测**：使用 YOLOv8 算法进行实时头盔检测
- 🖼️ **可视化界面**：基于 Streamlit 的用户友好界面
- 🔄 **模型训练**：支持自定义数据训练头盔检测模型
- 🚀 **硬件加速**：支持 Apple MPS 加速（适用于 Apple Silicon 设备）
- 📂 **批量处理**：支持多张图片同时检测与结果展示

## 项目结构

```
├── app.py               # Streamlit 应用界面
├── detect_helmets.py    # 核心检测功能模块
├── train_custom.py      # 模型训练脚本
├── clean.py             # 清理工具脚本
├── data/                # 数据集目录
│   ├── train/           # 训练集（图片和标签）
│   ├── valid/           # 验证集（图片和标签）
│   ├── test/            # 测试集（图片和标签）
│   └── data.yaml        # 数据集配置文件
├── models/              # 模型目录
│   └── yolov8n.pt       # 预训练模型
├── runs/                # 运行结果目录
│   ├── detect/          # 训练结果
│   └── inference/       # 推理结果
└── requirement.txt      # 项目依赖
```

## 安装说明

### 1. 克隆项目（如果适用）

```bash
git clone <repository_url>
cd YOLO
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# MacOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirement.txt
```

## 使用方法

### 1. 使用 Web 界面

启动 Streamlit 应用：

```bash
streamlit run app.py
```

在浏览器中打开界面后：
1. 上传一张或多张图片
2. 调整置信度阈值（默认 0.5）
3. 选择模型路径（默认使用训练后的模型）
4. 点击「开始检测」按钮
5. 查看检测结果并可下载带标注的图片

### 2. 命令行检测

直接使用 `detect_helmets.py` 进行批量检测：

```bash
python detect_helmets.py
```

可以修改脚本中的参数来自定义检测行为：
- `model_path`：模型权重文件路径
- `source`：测试图片/视频/文件夹路径
- `conf`：置信度阈值

## 自定义训练

### 1. 准备数据集

确保数据集按照以下结构组织：
```
data/
├── train/
│   ├── images/  # 训练图片
│   └── labels/  # 标签文件（YOLO 格式）
├── valid/
│   ├── images/  # 验证图片
│   └── labels/  # 标签文件
└── data.yaml    # 数据集配置文件
```

### 2. 配置数据集

修改 `data/data.yaml` 文件：
```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 2  # 类别数量
names: ['hat', 'person']  # 类别名称
```

### 3. 开始训练

运行训练脚本：

```bash
python train_custom.py
```

可以修改脚本中的训练参数：
- `epochs`：训练轮数
- `imgsz`：图片大小
- `batch`：批次大小

## 模型信息

- **预训练模型**：`models/yolov8n.pt` - YOLOv8 nano 版本，体积小、速度快
- **训练后模型**：`runs/detect/train1/weights/best.pt` - 自定义训练后的最优模型

## 常见问题

### 1. 检测结果不准确

- 增加训练轮数（epochs）
- 检查数据集质量和标注准确性
- 尝试调整置信度阈值

### 2. 运行速度慢

- 对于 Apple Silicon 设备，确保 MPS 加速已启用（代码中已包含自动检测）
- 使用较小的模型或减小输入图像尺寸

### 3. 检测结果保存位置

- Web 界面检测结果：保存在 `runs/inference/streamlit_时间戳_索引/` 目录下
- 命令行检测结果：保存在 `runs/inference/helmet_detect/` 目录下

## 注意事项

- 模型较大时推理会比较慢
- 使用 Apple Silicon 时会自动启用 MPS 加速
- 训练数据质量对检测效果至关重要
- 多图片处理时，每个图片会单独显示结果和下载按钮

## 许可证

[MIT License](https://opensource.org/licenses/MIT)