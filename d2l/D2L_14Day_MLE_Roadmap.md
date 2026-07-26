# 14天D2L冲刺路线图：DS → MLE New Grad

> **背景**：DS master new grad｜**目标**：MLE new grad面试就绪  
> **每日投入**：6小时｜**学习方式**：例子驱动  
> **核心教材**：[Dive into Deep Learning (d2l.ai)](https://d2l.ai)

---

## 全局概览

| 阶段 | 天数 | 主题 | 里程碑 |
|------|------|------|--------|
| 第一周 | Day 1–3 | PyTorch + DL基础 | 能手写MLP并调通训练循环 |
| 第一周 | Day 4–6 | CNN全家桶 | 复现ResNet跑CIFAR-10 |
| 第一周 | Day 7 | 周项目1 + 复盘 | 提交Week 1 Project |
| 第二周 | Day 8–10 | 序列模型 + Transformer | 写出完整Transformer |
| 第二周 | Day 11–12 | 高阶技巧 + 系统设计 | 能描述完整ML pipeline |
| 第二周 | Day 13 | 模拟面试 + 查漏 | 通过自测题 |
| 第二周 | Day 14 | 最终项目 | 上传GitHub展示作品 |

---

## 第一周：深度学习核心武器库

### Day 1｜PyTorch基础 + 线性模型（6h）

**今日目标**：把PyTorch当做"可微分NumPy"来理解

#### 学习内容（4h）
- D2L Chapter 2: 预备知识（张量、自动微分、线性代数回顾）
- D2L Chapter 3: 线性神经网络（线性回归、Softmax分类）
- 重点读：`autograd`机制——**理解计算图是一切的基础**

#### 关键例子理解顺序
```
手写线性回归（纯numpy）→ 改成PyTorch手动版 → 用nn.Module封装版
每步都print中间变量，感受梯度流动
```

#### 实践任务（2h）
```python
# 任务1：手写线性回归，3步走
# Step 1: 生成数据
import torch
X = torch.randn(100, 2)
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
y = X @ true_w + true_b + torch.randn(100) * 0.01

# Step 2: 手动实现（不用nn）
w = torch.randn(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(100):
    y_hat = X @ w + b
    loss = ((y_hat - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.03 * w.grad
        b -= 0.03 * b.grad
        w.grad.zero_()
        b.grad.zero_()

# Step 3: 用nn.Linear + SGD重写，对比两版本结果是否一致
```
- 任务2：在Fashion-MNIST上训练Softmax分类器，把测试集精度>85%

#### 今日检查清单
- [ ] 能解释 `loss.backward()` 到底做了什么
- [ ] 理解为什么要 `grad.zero_()`
- [ ] Softmax分类器跑通，loss曲线下降
- [ ] 完成d2l notebook：3.1, 3.2, 3.3, 3.6, 3.7

---

### Day 2｜MLP + 正则化 + 数值稳定性（6h）

**今日目标**：理解深层网络为什么难训练，以及如何驯服它

#### 学习内容（3.5h）
- D2L Chapter 4: 多层感知机全章
  - 4.1 MLP原理 → 4.4 模型选择/过拟合/欠拟合
  - **重点**：4.5 权重衰减、4.6 Dropout
  - **重点**：4.8 数值稳定性（梯度爆炸/消失）

#### 关键例子：亲手看梯度消失
```python
# 感受梯度消失
net = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(20)])
x = torch.randn(1, 4)
x.requires_grad_(True)
y = net(x).sum()
y.backward()
# 打印每层的梯度范数
for i, layer in enumerate(net):
    print(f"Layer {i} grad norm: {layer.weight.grad.norm():.6f}")
# 你会看到前几层梯度接近0 → 这就是问题所在
```

#### 实践任务（2.5h）
- 在MNIST上对比：无正则 vs Dropout(0.5) vs L2正则，画出train/val loss曲线
- 实现一个自定义Dropout层（不用nn.Dropout），验证和官方版结果接近
- 实验：调整hidden层数量（2层 vs 5层 vs 10层），观察梯度范数变化

#### 今日检查清单
- [ ] 能用一句话解释过拟合 vs 欠拟合
- [ ] 理解Dropout在train/eval模式的区别（`model.train()` vs `model.eval()`）
- [ ] 梯度消失实验跑通，有截图或记录

---

### Day 3｜CNN基础 + LeNet + AlexNet（6h）

**今日目标**：彻底理解卷积的"参数共享"和"局部感受野"为什么有效

#### 学习内容（3.5h）
- D2L Chapter 6: 卷积神经网络
  - 6.1–6.3 卷积操作（从互相关到卷积层）
  - 6.4–6.6 池化层、多通道
- D2L Chapter 7.1: LeNet（第一个现代CNN）
- D2L Chapter 8.1: AlexNet（ImageNet时代开始）

#### 关键例子：手写2D卷积
```python
# 不用F.conv2d，手动实现，理解参数
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 用这个检测图像边缘
X = torch.ones(6, 8)
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
print(corr2d(X, X))  # 边缘检测结果
```

#### 实践任务（2.5h）
- 复现LeNet在Fashion-MNIST上，精度>88%
- 修改LeNet：把sigmoid改成ReLU，加BatchNorm，看精度变化
- 用 `torchinfo.summary(model, input_size)` 打印模型参数量，理解每层shape变化

#### 今日检查清单
- [ ] 能手算：输入(N,C,H,W)经过Conv(out_channels, kernel, stride, padding)后的输出shape
- [ ] 理解为什么CNN比全连接参数少100倍
- [ ] LeNet跑通，能解释每层在做什么

---

### Day 4｜现代CNN：VGG / GoogLeNet / ResNet / DenseNet（6h）

**今日目标**：掌握ResNet——面试最高频考点之一

#### 学习内容（3h）
- D2L Chapter 8.2–8.7:
  - VGG（堆叠3×3的力量）
  - NiN（引入1×1卷积）
  - GoogLeNet / Inception（并行结构）
  - **ResNet** ⭐⭐⭐（重中之重）
  - DenseNet（ResNet的延伸）

#### 关键例子：ResBlock从零实现
```python
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, use_1x1=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # 当维度不匹配时用1x1卷积做shortcut
        self.shortcut = nn.Conv2d(channels, channels, 1, stride=stride) if use_1x1 else nn.Identity()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))  # ← 关键：残差连接

# 关键问题：为什么残差连接能解决梯度消失？
# 答：梯度可以直接通过shortcut回流，不需要经过多层非线性
```

#### 实践任务（3h）
- 用ResBlock搭一个Mini-ResNet（约4个Block），在CIFAR-10训练到>85%
- 实验：去掉shortcut（变成普通CNN），对比20层时的训练曲线差异
- 用 `torchvision.models.resnet18(pretrained=False)` 看官方实现，和自己的对比

#### 面试重点记录
> **高频问题**：ResNet为什么work？BatchNorm在哪里放？  
> **答**：残差连接让梯度可以绕过非线性层直接反传；BN通常放在Conv后、激活前（Pre-activation变体是激活前）

#### 今日检查清单
- [ ] 能徒手画出ResBlock结构图（含shortcut）
- [ ] Mini-ResNet跑通CIFAR-10，精度>85%
- [ ] 理解1×1卷积的作用（通道数变换 + 跨通道信息融合）

---

### Day 5｜序列模型：RNN / LSTM / GRU（6h）

**今日目标**：理解RNN的本质是"带记忆的函数"，LSTM是如何解决其遗忘问题的

#### 学习内容（3.5h）
- D2L Chapter 9: 现代循环神经网络
  - 9.1–9.3 序列数据 + 语言模型 + RNN基础
  - 9.4–9.5 从零实现RNN
- D2L Chapter 10.1–10.3: LSTM + GRU（重点）

#### 关键例子：LSTM门控机制可视化
```python
# 用一个极简例子理解门控
# 假设句子："The cats that the dog chased were scared"
# 主语"cats"和谓语"were"中间隔了很多词
# LSTM的forget gate学会保留"cats是复数"这个信息

# 直观感受：打印LSTM hidden state的范数
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(1, 50, 10)  # batch=1, seq_len=50, features=10
output, (h_n, c_n) = lstm(x)
print(f"output shape: {output.shape}")   # (1, 50, 20)
print(f"h_n shape: {h_n.shape}")         # (2, 1, 20) ← num_layers
print(f"c_n shape: {c_n.shape}")         # (2, 1, 20) ← cell state才是长期记忆

# LSTM vs GRU的关键区别：
# LSTM: 分离细胞状态(c)和隐藏状态(h)
# GRU: 合并成一个隐藏状态，参数更少，速度更快
```

#### 实践任务（2.5h）
- 用RNN做字符级语言模型：输入"hell"，预测"ello"（d2l经典例子）
- 对比RNN vs LSTM在长序列上的梯度范数（序列长度=100 vs 500）
- 用LSTM做情感分类（IMDB数据集，Accuracy>85%）

#### 今日检查清单
- [ ] 能解释LSTM的4个门（forget/input/output/cell update）各自的作用
- [ ] GRU vs LSTM：各自优缺点，何时用哪个
- [ ] 理解 `batch_first=True` 对shape的影响

---

### Day 6｜注意力机制 + Transformer（6h）

**今日目标**：这是最重要的一天——Transformer是现代DL的基石

#### 学习内容（3.5h）
- D2L Chapter 11: 注意力机制（全章精读）
  - 11.1 注意力汇聚（直觉：你阅读时眼睛聚焦在哪）
  - 11.2 注意力评分函数
  - 11.3 多头注意力
  - 11.5 Self-Attention
  - 11.7 **Transformer架构** ⭐⭐⭐

#### 关键例子：Scaled Dot-Product Attention手写
```python
import math

def attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_q, d_k)
    K: (batch, heads, seq_k, d_k)  
    V: (batch, heads, seq_v, d_v)
    """
    d_k = Q.size(-1)
    # Step 1: 计算相似度分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Step 2: 可选的mask（decoder用，防止看到未来）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Step 3: Softmax得到注意力权重
    attn_weights = torch.softmax(scores, dim=-1)
    # Step 4: 加权求和Value
    return torch.matmul(attn_weights, V), attn_weights

# 关键直觉：Q问"我要找什么"，K说"我是什么"，V说"我的内容是什么"
# 高相似度的(Q,K)对，对应的V权重更大
```

#### 实践任务（2.5h）
- 从零搭一个Mini-Transformer（2层、4头、d_model=128），做玩具级机器翻译
- 实验：可视化注意力权重矩阵（热力图），观察模型"在看哪里"
- 阅读并注释d2l的完整Transformer实现，在每个操作后加上shape注释

#### 面试重点记录
> **高频问题**：为什么要除以√d_k？为什么用Multi-Head？Position Encoding的作用？  
> **答**：防止d_k大时点积过大导致softmax梯度消失；多头学习不同类型的关系；Transformer无序列归纳偏置，需要显式注入位置信息

#### 今日检查清单
- [ ] 能默写Scaled Dot-Product Attention公式
- [ ] 能解释Encoder-Decoder架构中三种Attention（self/cross/masked self）的区别
- [ ] 注意力权重热力图可视化完成

---

### Day 7｜第一周项目 + 复盘（6h）

**Week 1 项目：图像分类全流程（4h）**

在CIFAR-10上做一个"工业级小流程"：

```
目标：从数据到部署，走一遍完整工程流程
```

```python
# 项目结构
cifar_project/
├── data.py        # 数据加载 + augmentation
├── model.py       # 自定义ResNet（用Day 4的代码）
├── train.py       # 训练循环 + 验证 + 早停
├── evaluate.py    # 混淆矩阵 + 每类精度
└── inference.py   # 单张图片预测 + 可视化

# 必须包含的技术点
# 1. 数据增强：RandomCrop, RandomHorizontalFlip, Normalize
# 2. 学习率调度：CosineAnnealingLR
# 3. 混合精度训练：torch.cuda.amp (如有GPU)
# 4. 模型保存/加载：torch.save / torch.load
# 5. 训练曲线：matplotlib画loss + accuracy

# 目标精度：>88%（不用pretrained）
```

**复盘任务（2h）**
- 整理Week 1笔记，每个模型写一张"模型卡片"（架构/创新点/适用场景）
- 做自测：用白板默写ResBlock、LSTM cell、Attention公式
- 记录3个"原来如此"的顿悟时刻

---

## 第二周：高阶技巧 + 面试就绪

### Day 8｜优化算法 + 训练技巧（6h）

**今日目标**：理解为什么Adam是"开箱即用的首选"，但SGD+Momentum有时更好

#### 学习内容（3h）
- D2L Chapter 12: 优化算法
  - 12.3 随机梯度下降 vs Mini-batch
  - 12.5 Momentum
  - 12.6 AdaGrad
  - 12.7 RMSProp
  - 12.8 **Adam** ⭐
  - 12.11 学习率调度

#### 关键例子：可视化优化器行为
```python
# 在Rosenbrock函数上对比优化器轨迹
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 各优化器的直觉：
# SGD: 直接走梯度方向，容易抖动
# Momentum: 加了惯性，过山谷时更稳
# Adam: 自适应学习率 + momentum，对稀疏梯度友好
# 实际建议：Adam默认lr=1e-3开始；fine-tuning用更小lr

# 学习率调度对比
schedulers = {
    'StepLR': torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1),
    'Cosine': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
    'OneCycle': torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=100)
}
# 画出lr变化曲线，直观感受
```

#### 实践任务（3h）
- 在同一个模型上对比5种优化器（SGD/Momentum/AdaGrad/RMSProp/Adam），画收敛曲线
- 实现学习率warmup：前5个epoch线性升到目标lr，之后cosine decay
- **Trick实验**：Gradient Clipping对LSTM的影响（梯度爆炸场景）

#### 今日检查清单
- [ ] 能推导Adam更新公式（一阶矩 + 二阶矩 + bias correction）
- [ ] 理解为什么RNN需要gradient clipping而CNN通常不需要
- [ ] 学习率warmup实现完成

---

### Day 9｜现代训练技巧 + 工程实践（6h）

**今日目标**：掌握让模型从"跑通"到"跑好"的工程技巧

#### 学习内容（2.5h）
- D2L Chapter 13: 计算性能（并行、内存优化）
- BatchNorm深入：原理 + 实现 + 注意事项
- 混合精度训练（FP16/BF16）
- 数据加载优化（DataLoader的num_workers, pin_memory）

#### 关键例子：BatchNorm从零实现
```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # 更新running统计量（推理时用）
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# 关键：train/eval模式的区别！这是面试常考点
```

#### 实践任务（3.5h）
- 用 `torch.profiler` 找出训练瓶颈（数据IO vs 前向 vs 反向）
- 实现混合精度训练，对比纯FP32的速度和显存
- 搭建一个完整的训练框架（含：logging/checkpoint/early_stopping/tensorboard）

#### 今日检查清单
- [ ] 理解BN在train/eval模式的区别（running_mean vs batch_mean）
- [ ] 能解释LayerNorm vs BatchNorm的适用场景（BN→CV，LN→NLP）
- [ ] 训练框架模板写好，后续项目直接复用

---

### Day 10｜计算机视觉：目标检测 + 迁移学习（6h）

**今日目标**：掌握迁移学习（MLE实际工作中用得最多的CV技能）

#### 学习内容（3h）
- D2L Chapter 14: 计算机视觉
  - 14.1 图像增广
  - 14.2 **微调（Fine-tuning）** ⭐⭐⭐
  - 14.3–14.5 目标检测基础（anchor boxes, IoU, NMS）
  - 14.8 区域卷积神经网络（R-CNN系列概览）

#### 关键例子：Fine-tuning全流程
```python
import torchvision.models as models

# 场景：用预训练ResNet-50识别你自己的数据集（假设10类）

# Step 1: 加载预训练模型
model = models.resnet50(weights='IMAGENET1K_V1')

# Step 2: 冻结骨干网络
for param in model.parameters():
    param.requires_grad = False

# Step 3: 替换最后一层（只有这层会训练）
model.fc = nn.Linear(model.fc.in_features, 10)

# Step 4: 先用大lr训练新层（5 epochs）
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
# ... train 5 epochs

# Step 5: 解冻后几层，用小lr fine-tune
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
# ... train another 10 epochs

# 这种"逐渐解冻"策略叫 Progressive Unfreezing（ULMFiT思路）
```

#### 实践任务（3h）
- 用预训练ResNet-50在一个小数据集（如Oxford Pets，37类）fine-tune，精度>90%
- 实验：全微调 vs 只训练最后层 vs 逐渐解冻，对比3种策略
- 理解NMS算法，手写一个简化版NMS

#### 今日检查清单
- [ ] 能解释为什么迁移学习有效（特征层次性）
- [ ] 理解什么时候冻结什么时候解冻
- [ ] IoU的计算公式能手写

---

### Day 11｜NLP：Embedding + BERT + 现代LLM基础（6h）

**今日目标**：理解从Word2Vec到BERT的演进，以及为什么GPT系列统治了世界

#### 学习内容（3h）
- D2L Chapter 15: 自然语言处理：预训练
  - 15.1 Word2Vec（skip-gram / CBOW）
  - 15.4 GloVe
- D2L Chapter 16: 自然语言处理：应用
  - **16.1–16.4 BERT** ⭐⭐⭐（精读）
  - 16.5–16.8 BERT下游任务微调

#### 关键例子：BERT的两个预训练任务
```python
# BERT的创新在于：双向 + 两个预训练任务

# Task 1: Masked Language Model (MLM)
# 输入: "The [MASK] sat on the mat"
# 目标: 预测[MASK] = "cat"
# 优势: 双向上下文！（比GPT的单向预测更适合理解任务）

# Task 2: Next Sentence Prediction (NSP)
# 输入: [CLS] Sentence A [SEP] Sentence B [SEP]
# 目标: IsNext / NotNext

# Fine-tuning BERT做分类（最简单的用法）
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
# outputs.logits → 分类结果

# 关键：[CLS] token的输出用于分类
```

#### 实践任务（3h）
- 用HuggingFace的BERT在SST-2情感分类上fine-tune（Accuracy>93%）
- 对比：LSTM方案 vs BERT方案在同一数据集上的精度和训练时间
- 阅读：BERT论文Abstract + Introduction（15分钟），理解核心贡献

#### BERT vs GPT vs T5 对比卡片
| 模型 | 架构 | 预训练任务 | 适合 |
|------|------|-----------|------|
| BERT | Encoder | MLM + NSP | 理解（分类/NER/QA） |
| GPT | Decoder | 语言模型（LM） | 生成 |
| T5 | Encoder-Decoder | 文本转文本 | 通用（seq2seq） |

#### 今日检查清单
- [ ] 理解为什么BERT是双向的，GPT是单向的
- [ ] BERT fine-tune完成，知道如何处理不同下游任务
- [ ] 能解释Tokenization中的WordPiece/BPE

---

### Day 12｜生成模型 + 系统设计基础（6h）

**今日目标**：了解VAE/GAN的原理，重点掌握ML系统设计框架

#### 上午（3h）：生成模型
- D2L Chapter 20.2: VAE（Variational Autoencoder）
- D2L Chapter 20.3: GAN基础
- Diffusion Model概念理解（不需要实现）

```python
# VAE的直觉：编码成分布，而不是点
# 普通AE: x → z（一个点）→ x'
# VAE:    x → μ,σ（一个分布）→ 采样z → x'
#              ↑
#         这让latent space连续，可以插值生成新样本

# VAE损失 = 重构损失 + KL散度
# KL散度强迫latent分布接近标准正态，防止塌缩到离散点
loss = reconstruction_loss + beta * kl_divergence

# GAN的直觉：生成器 vs 判别器的博弈
# G想生成以假乱真的样本
# D想区分真假
# 纳什均衡：G生成的样本无法被D区分 → 训练完成
```

#### 下午（3h）：ML系统设计框架
这是MLE面试的核心考点！

```
ML系统设计回答框架（CIRCLES法改版）：

1. 理解需求（2分钟）
   - 业务目标是什么？（CTR? 留存率? 安全性?）
   - 规模？（DAU, QPS, 数据量）
   - 延迟要求？（实时 < 100ms，还是批处理）

2. 定义ML问题（3分钟）
   - 监督/无监督/强化学习？
   - 分类/回归/排序/生成？
   - 目标指标（online metrics: CTR/DAU vs offline metrics: AUC/NDCG）

3. 数据（3分钟）
   - 数据来源，标注策略
   - 特征工程（user特征/item特征/context特征）
   - 数据不平衡怎么处理

4. 模型选择（5分钟）
   - Baseline → 复杂模型（先简后繁）
   - Trade-off: 精度 vs 速度 vs 可解释性

5. 训练（3分钟）
   - 离线训练 vs 在线学习
   - 负采样策略（负样本怎么选）

6. 线上服务（3分钟）
   - 特征存储（Feature Store）
   - 模型版本管理
   - A/B测试

7. 监控（2分钟）
   - 数据分布漂移检测
   - 模型性能监控，回滚机制
```

#### 实践任务
- 用上面的框架，模拟设计"YouTube推荐系统"（30分钟口述 + 写要点）
- 实现一个简单VAE在MNIST上生成手写数字

#### 今日检查清单
- [ ] 能用7步框架回答任何ML系统设计题
- [ ] VAE实现完成，能生成看起来像数字的样本
- [ ] 理解GAN训练不稳定的根本原因

---

### Day 13｜模拟面试 + 查漏补缺（6h）

#### 上午（3h）：编码面试刷题

**高频ML编码题清单**：
```python
# 题1：手写Softmax（含数值稳定版）
def softmax(x):
    x = x - x.max(dim=-1, keepdim=True).values  # 稳定性trick
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# 题2：手写Cross-Entropy Loss
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y]).mean()

# 题3：手写K-Means（DS背景应该会）
# 题4：手写Precision/Recall/F1
# 题5：实现BatchNorm（见Day 9）
# 题6：手写注意力机制（见Day 6）
# 题7：解释并实现Dropout
# 题8：实现一个简单的DataLoader
```

#### 下午（3h）：理论面试模拟

**按主题的高频问题**：

**基础DL理论**
- 为什么用ReLU而不是sigmoid？（梯度消失 + 计算效率）
- Dropout的本质是什么？（隐式ensemble + 防止协同适应）
- BatchNorm为什么有效？（减少internal covariate shift，允许大lr）
- L1和L2正则化的区别？（L1产生稀疏解，L2平滑解）

**CNN相关**
- 卷积的参数量怎么算？
- Global Average Pooling vs Flatten？
- 深层网络退化问题如何解决？

**Transformer相关**
- Self-Attention的时间复杂度？（O(n²d)，n是序列长度）
- 为什么需要Position Encoding？
- Multi-Head Attention的好处？

**MLE实践**
- 如何处理类别不平衡？（oversampling/undersampling/focal loss/class weights）
- 模型部署的常见优化？（量化/剪枝/蒸馏/ONNX）
- 如何debug模型不收敛？（检查清单见下方）

**模型不收敛Debug清单**
```
□ 检查数据：样本标签是否对齐？数据是否归一化？
□ 检查梯度：是否消失/爆炸？用gradient clipping
□ 检查学习率：太大（loss震荡）or太小（loss不动）？
□ 检查损失函数：是否适合任务？
□ 检查模型：从最小模型开始，能过拟合一个batch就说明模型ok
□ 检查数据增强：是否过于aggressive导致信息丢失？
```

#### 今日检查清单
- [ ] 完成所有8道编码题，计时（每题<20分钟）
- [ ] 录音自己回答5道理论题，回放检查是否清晰
- [ ] 整理出自己的"知识盲区"，明天项目中针对性实践

---

### Day 14｜最终项目（6h）

**项目：端到端图文多模态分类器**

从数据到GitHub，展示你已经是MLE水平。

#### 项目选择（三选一，根据兴趣）

**Option A（CV向）**：实现CLIP-lite（图文对比学习）
```python
# 核心思想：让图像和文本描述的embedding距离最近
# 用InfoNCE Loss训练

class CLIPLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = models.resnet50(weights='IMAGENET1K_V1')
        self.text_encoder = ...  # 简化版Transformer
        self.projection = nn.Linear(512, 128)
    
    def forward(self, images, texts):
        img_feat = self.projection(self.image_encoder(images))
        txt_feat = self.projection(self.text_encoder(texts))
        # 余弦相似度矩阵
        logits = img_feat @ txt_feat.T / temperature
        return logits
```

**Option B（NLP向）**：情感分析系统（含serving）
- BERT fine-tune + FastAPI serving + 简单前端
- 要求：能接受任意文本输入，返回情感分类 + 置信度

**Option C（推荐系统向）**：简单的双塔推荐模型
- User Tower + Item Tower
- 负采样策略实现
- 在MovieLens数据集上评估（HR@10, NDCG@10）

#### 必须包含的GitHub仓库结构
```
final_project/
├── README.md          # 清晰的项目介绍（含结果图表）
├── requirements.txt   # 环境依赖
├── data/
│   └── download.sh   # 数据下载脚本
├── src/
│   ├── model.py      # 模型定义
│   ├── dataset.py    # 数据处理
│   ├── train.py      # 训练入口
│   └── evaluate.py   # 评估
├── notebooks/
│   └── EDA.ipynb     # 探索性分析（含可视化）
├── experiments/
│   └── results.json  # 实验结果对比
└── checkpoints/      # 保存最好的模型
```

#### README模板（面试官看第一眼就要能懂）
```markdown
# 项目名称

## 问题描述
一句话说清楚解决什么问题

## 方法
- 模型架构：xxx
- 训练策略：xxx
- 关键创新：xxx

## 结果
| 方法 | Accuracy | F1 | 训练时间 |
|------|----------|----|---------|
| Baseline (LR) | 72% | 0.71 | 5s |
| 本方法 | 91% | 0.90 | 2h |

## 快速开始
pip install -r requirements.txt
python src/train.py --config configs/default.yaml

## 结果复现
...
```

---

## 里程碑 & 进度衡量

### 每日自测标准
| 天数 | 测试方式 | 通过标准 |
|------|---------|---------|
| Day 1 | 默写梯度下降代码 | <10分钟写出不看文档 |
| Day 3 | 手算CNN输出shape | 连续3题不出错 |
| Day 4 | 画出ResBlock | 含shortcut，30秒内完成 |
| Day 6 | 解释Attention机制 | 用自己的话，无需查资料 |
| Day 9 | BN的train/eval区别 | 能说出4个关键差异 |
| Day 13 | 编码题 | 8道题全部在20分钟内完成 |

### 周里程碑

**Week 1 结束时（Day 7）**
- [ ] 能手写：线性回归、MLP、ResBlock、LSTM cell
- [ ] CIFAR-10项目精度 > 88%
- [ ] 能向别人解释反向传播的计算图机制

**Week 2 结束时（Day 14）**
- [ ] 能手写Attention（含multi-head）
- [ ] 能用7步框架设计任何ML系统
- [ ] GitHub有3个以上有star价值的项目
- [ ] 模拟面试能连续回答30分钟不露怯

---

## 初学者常见错误（必读！）

### 错误1：跳过基础直接看高阶模型
```
❌ 第一天就想跑GPT
✅ 先把线性回归的每一行代码理解透彻
原因：基础不牢，遇到bug不知道从哪里查
```

### 错误2：只看代码，不理解为什么
```
❌ 复制d2l代码跑通就过
✅ 删掉某一行，预测会发生什么，再实际验证
技巧：在每个操作后打印shape，理解数据流动
```

### 错误3：不区分train/eval模式
```python
# 经典错误：eval时忘记切换模式
model.eval()  # 必须！否则Dropout/BN行为不对
with torch.no_grad():  # 节省内存，不计算梯度
    output = model(x)
```

### 错误4：GPU相关的新手陷阱
```python
# ❌ 常见错误
loss = criterion(output, labels)  # labels还在CPU，output在GPU → 报错

# ✅ 正确做法
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X, y = X.to(device), y.to(device)  # 数据也要移到同一设备
```

### 错误5：过拟合了还在调模型
```
遇到train loss很低但val loss很高时：
❌ 立刻换更复杂的模型
✅ 先加正则化（Dropout/L2/数据增强）
   验证：能在10个样本上过拟合吗？能 → 说明模型容量够，问题在数据/正则
```

### 错误6：学习率选择错误
```python
# 经验范围（Adam）：
# 从头训练：1e-3 到 1e-4
# Fine-tuning：1e-5 到 1e-4（预训练层更小）
# 学习率过大：loss震荡或nan
# 学习率过小：训练极慢，可能卡在局部最优

# 快速找学习率的方法（LR Finder）
from torch_lr_finder import LRFinder
finder = LRFinder(model, optimizer, criterion)
finder.range_test(train_loader, end_lr=1, num_iter=100)
finder.plot()  # 找loss下降最陡的位置的前一个数量级
```

### 错误7：面试时过于依赖记忆公式
```
面试官不是考你背公式，而是考你理解
✅ 正确姿势：先说直觉/动机，再推公式
例：问"为什么BatchNorm有效"
❌ "公式是 (x-μ)/σ * γ + β"
✅ "训练过程中每层的输入分布在不断变化（internal covariate shift），
    BN通过归一化稳定了这个分布，让后面的层更容易学习。
    γ和β是可学习的，让模型能恢复最优的分布尺度。"
```

---

## 免费工具 & 资源

### 核心教材
- **D2L**: https://d2l.ai （中文版：https://zh.d2l.ai）
- **配套代码**: https://github.com/d2l-ai/d2l-zh

### 算力资源（免费）
| 平台 | 免费GPU额度 | 备注 |
|------|-----------|------|
| Google Colab | T4/L4，约12h/天 | 需要科学上网 |
| Kaggle Notebooks | P100，30h/周 | 无需科学上网 |
| Hugging Face Spaces | CPU免费，GPU有限 | 适合部署demo |

### 数据集（无需下载，直接用）
```python
# torchvision自带
torchvision.datasets.MNIST / CIFAR10 / CIFAR100 / ImageNet / FashionMNIST

# HuggingFace datasets
from datasets import load_dataset
ds = load_dataset("imdb")  # 情感分类
ds = load_dataset("squad")  # 问答

# Kaggle数据集（在Kaggle notebook中直接用）
```

### 调试工具
```python
# 1. 打印模型结构和参数量
pip install torchinfo
from torchinfo import summary
summary(model, input_size=(batch, channels, H, W))

# 2. 可视化训练过程
pip install tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp1')
writer.add_scalar('Loss/train', loss, epoch)

# 3. 实验管理（进阶，可选）
# Weights & Biases: wandb.ai（免费个人账户）
pip install wandb
import wandb
wandb.init(project="my-project")
wandb.log({"loss": loss, "accuracy": acc})
```

### 面试资源
- **ML面试圣经**: https://huyenchip.com/ml-interviews-book/
- **论文精读（李沐）**: https://space.bilibili.com/1567748478
- **系统设计**: Machine Learning System Design Interview（书）
- **LeetCode**: 每天1题（Medium，重点：数组/树/动态规划）

---

## 每日检查清单（打印贴桌面）

```
📅 每日开始（15分钟）
□ 回顾昨天笔记，复述3个关键点
□ 今天要完成的具体任务（写下来）

📚 学习阶段（4小时）
□ 读D2L章节，边读边跑代码
□ 遇到不懂的地方：先想5分钟，再查
□ 关键公式/架构图：手写在纸上

💻 实践阶段（2小时）
□ 完成今日实践任务
□ 代码跑通，结果符合预期
□ 把代码commit到GitHub（保持记录）

🌙 每日结束（15分钟）
□ 写"今日顿悟"（1-3条）
□ 写"明天要复习"（1-2条疑问）
□ 更新进度表
```

---

## 14天后的下一步

完成这14天后，你应该具备：

1. **能手写**：从线性回归到Transformer的核心组件
2. **能应用**：用预训练模型快速解决新任务
3. **能设计**：用框架回答ML系统设计题
4. **有作品**：GitHub上3-5个高质量项目

**接下来的方向**（根据目标岗位选择）：

- **CV方向**：深入目标检测（YOLO系列）、分割（SAM）、扩散模型
- **NLP/LLM方向**：LLM fine-tuning（LoRA/QLoRA）、RAG系统、RLHF
- **推荐系统**：双塔模型、序列推荐（SASRec）、特征交叉（DeepFM）
- **MLOps**：Docker/K8s部署、特征工程系统、模型监控

---

*Made for: DS → MLE New Grad in 14 days | 每天6小时 | 专注实践*
