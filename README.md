1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f123e590e9d14988987343db19bd7f98.png)


苹果叶片病虫害分类数据集（分类数据集，没有yolo等标注）
数据集包含14000余张张图像，其中94%的图像是在自然环境中拍摄的。数据集中的图像经过农业疾病专家筛选，移除了错误标记的图像
1.包含健康苹果叶和八种苹果叶病的图像
Alternaria leaf spot：交链孢叶斑病
Brown spot：褐斑病
Frogeye leaf spot：蛙眼叶斑病
Grey spot：灰斑病
Health：健康
Mosaic：马赛克病毒病
Powdery mildew：白粉病
Rust：锈病
Scab：疮痂病
2.图像主要来自自然环境，不限制拍摄角度、噪音等因素

以下是**苹果叶片病虫害分类数据集**的结构化表格描述：

| **属性类别**         | **详细描述** |
|----------------------|--------------|
| **数据集名称**       | Apple Leaf Disease Classification Dataset（苹果叶片病害分类数据集） |
| **任务类型**         | **图像分类**（无目标检测标注，如 YOLO、VOC、COCO 等） |
| **总图像数量**       | **14,000+ 张**（约 14,000–14,500 张） |
| **数据来源**         | **94% 图像采集自自然环境**（非实验室控制条件） |
| **拍摄条件**         | - 无固定拍摄角度<br>- 允许存在光照变化、背景杂乱、遮挡、模糊、噪声等真实场景干扰 |
| **质量控制**         | 所有图像均经**农业植保专家人工审核**，剔除错误标记或低质量样本，确保标签可靠性 |
| **类别总数**         | **9 类**（1 类健康 + 8 类病害） |
| **具体类别与中文名称** | 1. **Health**：健康叶片<br>2. **Alternaria leaf spot**：交链孢叶斑病<br>3. **Brown spot**：褐斑病<br>4. **Frogeye leaf spot**：蛙眼叶斑病<br>5. **Grey spot**：灰斑病<br>6. **Mosaic**：马赛克病毒病<br>7. **Powdery mildew**：白粉病<br>8. **Rust**：锈病<br>9. **Scab**：疮痂病 |
| **图像格式**         | 常见格式（如 JPG / PNG），RGB 三通道 |
| **典型应用场景**     | - 农业智能诊断系统<br>- 移动端植物病害识别 App<br>- 轻量化 CNN 模型部署（如 MobileNet、EfficientNet）<br>- 小样本学习、迁移学习研究 |
| **数据集特点**       | - **真实性强**：反映田间实际病害表现<br>- **类别多样**：覆盖真菌性、病毒性、生理性等多种病害<br>- **挑战性高**：存在类间相似（如多种叶斑病）、类内差异大（同一病害在不同光照/角度下表现不同）<br>- **专家验证**：标签准确率高，适合监督学习 |

> ✅ 该数据集适用于**细粒度植物病理图像分类**任务，是智慧农业与计算机视觉交叉领域的典型基准数据集。

---

### 📊 **类别概览表（建议用于 `class_to_idx` 映射）**

| **类别索引** | **英文标签**           | **中文病害名称**     |
|-------------|------------------------|----------------------|
| 0           | Health                 | 健康                 |
| 1           | Alternaria leaf spot   | 交链孢叶斑病         |
| 2           | Brown spot             | 褐斑病               |
| 3           | Frogeye leaf spot      | 蛙眼叶斑病           |
| 4           | Grey spot              | 灰斑病               |
| 5           | Mosaic                 | 马赛克病毒病         |
| 6           | Powdery mildew         | 白粉病               |
| 7           | Rust                   | 锈病                 |
| 8           | Scab                   | 疮痂病               |



为了训练苹果叶片病虫害分类数据集，我们可以使用深度学习框架如PyTorch来构建一个图像分类模型。以下是一个详细的训练代码示例，包括数据预处理、模型定义、训练过程以及评估。

### 一、安装依赖

首先确保你已经安装了必要的库：

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

### 二、数据准备

假设你的数据集按以下结构组织：

```
AppleLeafDiseaseDataset/
├── train/
│   ├── Alternaria_leaf_spot/
│   ├── Brown_spot/
│   ├── Frogeye_leaf_spot/
│   ├── Grey_spot/
│   ├── Health/
│   ├── Mosaic/
│   ├── Powdery_mildew/
│   ├── Rust/
│   └── Scab/
├── val/
│   ├── Alternaria_leaf_spot/
│   ├── Brown_spot/
│   ├── Frogeye_leaf_spot/
│   ├── Grey_spot/
│   ├── Health/
│   ├── Mosaic/
│   ├── Powdery_mildew/
│   ├── Rust/
│   └── Scab/
└── test/
    ├── Alternaria_leaf_spot/
    ├── Brown_spot/
    ├── Frogeye_leaf_spot/
    ├── Grey_spot/
    ├── Health/
    ├── Mosaic/
    ├── Powdery_mildew/
    ├── Rust/
    └── Scab/
```

### 三、自定义数据集类

创建一个自定义的数据集类来加载和预处理图像及其标签。

```python
# dataset.py
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AppleLeafDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                if os.path.isfile(path):
                    samples.append((path, cls_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    @property
    def class_names(self):
        return self.classes
```

### 四、模型定义与训练脚本

接下来是定义模型并编写训练脚本。

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import AppleLeafDiseaseDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = AppleLeafDiseaseDataset(root_dir='AppleLeafDiseaseDataset/train', transform=transform)
    val_dataset = AppleLeafDiseaseDataset(root_dir='AppleLeafDiseaseDataset/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 定义模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    num_epochs = 20
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.class_names, yticklabels=train_dataset.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # 打印分类报告
        print(classification_report(all_labels, all_preds, target_names=train_dataset.class_names))

if __name__ == "__main__":
    main()
```

### 五、注意事项

1. **数据路径**：请根据实际数据存放位置修改 `root_dir` 参数。
2. **模型选择**：这里使用了 ResNet18 作为基础模型，你可以根据需求更换为其他预训练模型（如 EfficientNet、MobileNet 等）。
3. **超参数调整**：根据硬件条件和数据特性调整 `batch_size`、`learning_rate` 等超参数。
4. **可视化工具**：使用 `matplotlib` 和 `seaborn` 库进行混淆矩阵的绘制，帮助分析模型性能。


