# 20天D2L冲刺路线图：DS → MLE New Grad

> **背景**：DS master new grad｜**目标**：MLE new grad面试就绪  
> **阶段一（Day 1–12）**：每天4小时｜**阶段二（Day 13–20）**：每天5小时  
> **总时长**：48h + 40h = **88小时**｜**学习方式**：例子驱动  
> **核心教材**：[Dive into Deep Learning (d2l.ai)](https://d2l.ai)

---

## 全局概览

| 阶段 | 天数 | 每日时长 | 主题 | 里程碑 |
|------|------|---------|------|--------|
| 基础期 | Day 1–4 | 4h | PyTorch + 线性模型 + MLP | 能手写并调通训练循环 |
| 基础期 | Day 5–8 | 4h | CNN全家桶 | 复现ResNet跑CIFAR-10 |
| 基础期 | Day 9–12 | 4h | 序列模型 + Transformer | 写出完整Attention |
| 进阶期 | Day 13–16 | 5h | 优化 + 工程技巧 + 迁移学习 | 掌握训练工程全套 |
| 进阶期 | Day 17–20 | 5h | NLP + 系统设计 + 面试冲刺 | 通过模拟面试 + 完成作品 |

---

## 阶段一：基础期（Day 1–12，每天4小时）

### Day 1｜PyTorch核心机制（4h）

**今日目标**：把PyTorch当"可微分NumPy"来用，彻底理解计算图

#### 学习内容（2.5h）
- D2L Chapter 2: 预备知识（张量操作、自动微分）
- 重点：`requires_grad`、`backward()`、计算图的构建与销毁

#### 关键例子
```python
# 感受autograd：让PyTorch帮你算导数
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x        # y = x³ + 2x
y.backward()
print(x.grad)              # dy/dx = 3x² + 2 = 14.0  ← PyTorch自动算出来的

# 计算图的直觉：每次forward都建了一张有向图
# backward()沿图反向传播，链式法则自动应用
```

#### 实践任务（1.5h）
- 手动验证：对 `f(x,y) = x²y + y³`，用PyTorch算偏导，和手算对比
- 实验：`detach()`、`with torch.no_grad()` 各自的效果，理解何时用哪个
- 实验：连续两次 `backward()` 会发生什么？为什么？

#### 今日检查清单
- [ ] 理解为什么每次反向传播后要 `grad.zero_()`
- [ ] 能解释 `retain_graph=True` 的使用场景
- [ ] 至少跑通3个autograd小实验

---

### Day 2｜线性模型 + Softmax分类（4h）

**今日目标**：用"三版实现"理解从手动到框架的抽象过程

#### 学习内容（2h）
- D2L Chapter 3.1–3.3: 线性回归（手动实现 → 简洁实现）
- D2L Chapter 3.4–3.7: Softmax回归（多分类基础）

#### 关键例子：三版线性回归
```python
# ── 版本1：纯手动 ──────────────────────────
w = torch.randn(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for _ in range(200):
    y_hat = X @ w + b
    loss = ((y_hat - y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.03 * w.grad; w.grad.zero_()
        b -= 0.03 * b.grad; b.grad.zero_()

# ── 版本2：用nn.Linear + 手动循环 ──────────
net = torch.nn.Linear(2, 1)
for _ in range(200):
    loss = torch.nn.MSELoss()(net(X), y)
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p -= 0.03 * p.grad; p.grad.zero_()

# ── 版本3：nn + optimizer（最终形态）────────
net = torch.nn.Linear(2, 1)
opt = torch.optim.SGD(net.parameters(), lr=0.03)
for _ in range(200):
    opt.zero_grad()
    torch.nn.MSELoss()(net(X), y).backward()
    opt.step()
# 三版结果应该一致，逐步理解"框架在帮你做什么"
```

#### 实践任务（2h）
- 在Fashion-MNIST上训练Softmax分类器，测试集精度 > 85%
- 手动实现数值稳定的Softmax（减去最大值），对比直接exp是否会溢出

#### 今日检查清单
- [ ] 三版线性回归都跑通，结果一致
- [ ] 理解 `opt.zero_grad()` 为什么要放在最前面
- [ ] Fashion-MNIST精度 > 85%

---

### Day 3｜MLP + 过拟合与正则化（4h）

**今日目标**：理解深层网络难训练的根因，学会用正则化驯服它

#### 学习内容（2h）
- D2L Chapter 4.1–4.3: MLP原理 + 从零实现 + 简洁实现
- D2L Chapter 4.4–4.6: 过拟合/欠拟合、权重衰减、Dropout

#### 关键例子：亲手看梯度消失
```python
# 20层网络，看梯度在前几层有多小
net = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(20)])
x = torch.randn(1, 4, requires_grad=True)
net(x).sum().backward()
for i, layer in enumerate(net):
    print(f"Layer {i:2d} grad norm: {layer.weight.grad.norm():.8f}")
# 输出你会看到：前面的层梯度趋近于0
# 这就是为什么我们需要ResNet、BatchNorm等技术
```

#### 实践任务（2h）
- 在MNIST上对比三种配置：无正则 / Dropout(p=0.5) / L2正则（weight_decay=1e-3），画收敛曲线
- 自己实现Dropout层（用随机mask），验证和 `nn.Dropout` 结果接近

#### 今日检查清单
- [ ] 理解Dropout在 `model.train()` 和 `model.eval()` 下行为的区别
- [ ] 三种正则化对比实验完成，有曲线图
- [ ] 能用一句话解释L1和L2正则的根本区别

---

### Day 4｜数值稳定性 + 训练循环模板（4h）

**今日目标**：搭好训练框架模板，后续所有实验都复用它

#### 学习内容（1.5h）
- D2L Chapter 4.8: 数值稳定性（梯度爆炸/消失）
- D2L Chapter 4.9: 参数初始化（Xavier / He init）
- 训练循环最佳实践

#### 关键例子：Xavier vs He初始化
```python
# 为什么初始化很重要？
# 错误初始化 → 激活值饱和 or 爆炸 → 梯度无法传播

# Xavier（适合sigmoid/tanh）
nn.init.xavier_uniform_(layer.weight)   # 方差 ∝ 1/n

# He（适合ReLU）
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # 方差 ∝ 2/n
# ReLU砍掉了一半神经元，所以方差要×2补偿
```

#### 实践任务（2.5h）
搭建可复用的训练框架模板（**后续所有实验都用这个**）：
```python
# trainer.py 模板
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    # ... 同上，去掉backward
    
def fit(model, train_loader, val_loader, epochs, optimizer, criterion, device, scheduler=None):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val = float('inf')
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(...)
        val_loss, val_acc = evaluate(...)
        if scheduler: scheduler.step()
        # 保存最优模型
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        # 记录history
    return history
```

#### 今日检查清单
- [ ] 训练框架模板写完并测试通过
- [ ] 理解为什么ReLU用He init，sigmoid用Xavier init
- [ ] `torch.save` / `torch.load` 模型保存加载流程跑通

---

### Day 5｜CNN基础 + LeNet（4h）

**今日目标**：彻底理解卷积的"参数共享"和"局部感受野"

#### 学习内容（2h）
- D2L Chapter 6.1–6.6: 卷积操作、池化层、多通道
- D2L Chapter 7.1: LeNet

#### 关键例子：手写2D卷积，理解参数量
```python
# 手动实现互相关运算
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 卷积 vs 全连接的参数量对比
# 输入：3×32×32图像
# 全连接层到256个神经元：3*32*32*256 = 786,432 个参数
# 卷积层(256, 3, 3×3)：256 * 3 * 3*3 = 6,912 个参数  ← 少100倍！
# 原因：参数共享（同一个filter扫描整张图）

# 输出shape速算公式（常考！）
# H_out = (H_in + 2*padding - kernel_size) / stride + 1
```

#### 实践任务（2h）
- 复现LeNet在Fashion-MNIST上，精度 > 88%
- 用 `torchinfo.summary(model, (1,1,28,28))` 打印每层的shape和参数量，理解数据流

#### 今日检查清单
- [ ] 能手算任意Conv层的输出shape
- [ ] LeNet精度 > 88%
- [ ] 理解MaxPool的作用（下采样 + 局部不变性）

---

### Day 6｜AlexNet + VGG + 1×1卷积（4h）

**今日目标**：理解"更深更宽"的演进逻辑，掌握1×1卷积这个万能工具

#### 学习内容（2h）
- D2L Chapter 8.1: AlexNet（ReLU + Dropout + 数据增强的组合拳）
- D2L Chapter 8.2: VGG（堆叠3×3的力量）
- D2L Chapter 8.3: NiN（引入1×1卷积）

#### 关键例子：1×1卷积的三个用途
```python
# 用途1：改变通道数（不改变H,W）
conv1x1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
# 256通道压缩到64通道，参数量：256*64*1*1 = 16,384

# 用途2：跨通道信息融合（不同通道之间交流）
# 用途3：增加非线性（后接ReLU）而不改变空间分辨率

# VGG的核心思想：用两个3×3代替一个5×5
# 感受野相同（5×5），但参数少（2*3*3=18 vs 5*5=25），且多了一层非线性
```

#### 实践任务（2h）
- 搭一个Mini-VGG（2个VGG Block），在CIFAR-10训练，精度 > 80%
- 实验：把VGG中的5×5卷积替换为两个3×3，对比参数量和精度

#### 今日检查清单
- [ ] 理解为什么VGG用两个3×3代替5×5
- [ ] 1×1卷积的3个用途能脱口而出
- [ ] Mini-VGG跑通

---

### Day 7｜ResNet + DenseNet（4h）⭐最重要的一天

**今日目标**：深度理解ResNet——面试最高频考点

#### 学习内容（2h）
- D2L Chapter 8.6: ResNet（精读）
- D2L Chapter 8.7: DenseNet（理解与ResNet的关系）

#### 关键例子：ResBlock从零实现
```python
class ResBlock(nn.Module):
    def __init__(self, channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        out_ch = channels * 2 if downsample else channels
        
        self.conv1 = nn.Conv2d(channels, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        
        # shortcut：维度不匹配时用1×1对齐
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if downsample else nn.Identity()
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))  # ← 残差连接核心

# 为什么work？梯度可以通过shortcut直接回流：
# ∂L/∂x = ∂L/∂(F+x) * (∂F/∂x + 1)
# 最后的"+1"保证了梯度至少不会消失
```

#### 实践任务（2h）
- 用ResBlock搭Mini-ResNet（4个Block），CIFAR-10精度 > 85%
- **对照实验**：去掉shortcut变成普通CNN，训练20层时对比loss曲线
- 用 `torchvision.models.resnet18` 看官方实现，找出和你的实现有何不同

#### 面试要背的答案
> **Q：ResNet为什么能训练很深的网络？**  
> A：残差连接让梯度可以绕过非线性层直接反传（梯度高速公路），解决了深层网络的梯度消失问题。同时，学习残差 F(x) 比学习完整映射 H(x) 更容易——在极端情况下，F(x)=0 就退化为恒等映射。

#### 今日检查清单
- [ ] 能徒手画出ResBlock（含shortcut和BN的位置）
- [ ] 对照实验完成，有曲线图证明shortcut的作用
- [ ] 理解Pre-activation ResNet（BN-ReLU-Conv）和原版的区别

---

### Day 8｜第一阶段项目：CIFAR-10全流程（4h）

**项目目标**：把前7天融合成一个"工业级小流程"

```
评分标准：
✅ 精度 > 88%（不用预训练）
✅ 代码结构清晰（可复用）
✅ 有完整的训练曲线和混淆矩阵
✅ 模型可保存/加载
```

#### 必须包含的技术点
```python
# 1. 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
])

# 2. CosineAnnealingLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. 混淆矩阵可视化（用seaborn heatmap）

# 4. 模型保存最优checkpoint
```

#### 项目结构
```
cifar_project/
├── data.py       # 数据加载 + augmentation
├── model.py      # ResNet定义
├── trainer.py    # 复用Day 4的模板
├── train.py      # 主入口
└── evaluate.py   # 混淆矩阵 + 每类精度
```

#### 今日检查清单
- [ ] 精度 > 88%，有截图
- [ ] 代码commit到GitHub
- [ ] 写下3条"本周学到的最重要的事"

---

### Day 9｜RNN + LSTM + GRU（4h）

**今日目标**：理解RNN是"带记忆的函数"，LSTM如何用门控解决遗忘

#### 学习内容（2.5h）
- D2L Chapter 9.1–9.5: 序列数据 + 语言模型 + RNN从零实现
- D2L Chapter 10.1–10.2: LSTM + GRU

#### 关键例子：LSTM的4个门
```python
# LSTM单步手动实现，帮助理解门控机制
def lstm_step(x, h_prev, c_prev, Wx, Wh, b):
    # 拼接输入和上一步hidden state
    combined = torch.cat([x, h_prev], dim=1)
    
    # 4个门（全部通过同一个大矩阵一次算完）
    gates = combined @ Wx + b   # shape: (batch, 4*hidden_size)
    i, f, g, o = gates.chunk(4, dim=1)  # 切分成4份
    
    i = torch.sigmoid(i)        # input gate：决定写入多少新信息
    f = torch.sigmoid(f)        # forget gate：决定遗忘多少旧信息  ← 关键！
    g = torch.tanh(g)           # cell gate：新信息的候选值
    o = torch.sigmoid(o)        # output gate：决定输出多少cell state
    
    c = f * c_prev + i * g      # 更新cell state（长期记忆）
    h = o * torch.tanh(c)       # 更新hidden state（短期记忆）
    return h, c

# GRU只有2个门（reset + update），参数更少，速度更快
# 适合数据量较小或实时性要求高的场景
```

#### 实践任务（1.5h）
- 字符级语言模型：输入"hell"预测"ello"（D2L经典例子，跑通即可）
- 对比RNN vs LSTM在序列长度=200时的梯度范数，验证LSTM对长依赖的优势

#### 今日检查清单
- [ ] 能解释LSTM的4个门各自的作用（面试常考）
- [ ] 理解 `h_n`（hidden state）和 `c_n`（cell state）的区别
- [ ] 理解 `batch_first=True` 对shape的影响

---

### Day 10｜注意力机制（4h）

**今日目标**：理解Attention的本质——"加权平均，权重由内容决定"

#### 学习内容（2.5h）
- D2L Chapter 11.1–11.3: 注意力汇聚 + 评分函数 + Bahdanau Attention
- D2L Chapter 11.5–11.6: Self-Attention + 位置编码

#### 关键例子：Scaled Dot-Product Attention手写
```python
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (batch, heads, seq, d_k)
    V:    (batch, heads, seq, d_v)
    
    直觉：Q问"我要找什么"，K说"我是什么"，V说"我的内容是什么"
    相似度高的(Q,K)对，对应的V权重更大
    """
    d_k = Q.size(-1)
    
    # 计算相似度，除以√d_k防止点积过大导致softmax梯度消失
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Decoder中需要mask，防止当前位置看到未来的信息
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn = torch.softmax(scores, dim=-1)          # 注意力权重
    return torch.matmul(attn, V), attn            # 输出 + 权重（可视化用）
```

#### 实践任务（1.5h）
- 实现Multi-Head Attention（把Q/K/V拆成多个头分别计算，再concat）
- 可视化单头注意力的权重矩阵（热力图），感受模型"在看哪里"

#### 今日检查清单
- [ ] 能默写Scaled Dot-Product Attention公式
- [ ] 理解为什么Multi-Head比Single-Head好（不同头学习不同类型的关系）
- [ ] 注意力热力图可视化完成

---

### Day 11｜Transformer（4h）⭐

**今日目标**：搭出完整Transformer，理解每个组件的作用

#### 学习内容（2h）
- D2L Chapter 11.7: Transformer架构（精读）
- 重点：Encoder/Decoder结构、三种Attention的区别

#### 三种Attention对比（必须搞清楚）
```
Encoder Self-Attention：
  每个token关注整个输入序列（双向）
  
Decoder Masked Self-Attention：
  每个token只能关注它之前的token（单向，防止作弊）
  
Decoder Cross-Attention：
  Q来自Decoder，K/V来自Encoder输出
  作用：让Decoder"查阅"原始输入信息
```

#### 关键例子：带shape注释的Transformer
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        attn_out, _ = self.attn(x, x, x)         # Self-Attention
        x = self.norm1(x + self.drop(attn_out))  # Add & Norm
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))    # Add & Norm
        return x  # (batch, seq, d_model)
```

#### 实践任务（2h）
- 用6个TransformerBlock搭Encoder，做文本分类（SST-2），精度 > 82%
- 在每个操作后打印shape，确保理解整个数据流

#### 面试要背的答案
> **Q：为什么Transformer比RNN好？**  
> A：①并行计算（RNN必须串行）②Self-Attention能直接建模任意距离的依赖（RNN随距离衰减）③更容易扩展到大规模数据

#### 今日检查清单
- [ ] 三种Attention区别能清楚解释
- [ ] 理解LayerNorm（而不是BatchNorm）用在Transformer的原因
- [ ] Position Encoding：为什么需要？sin/cos的直觉是什么？

---

### Day 12｜第二阶段项目：序列模型复盘（4h）

**项目：用Transformer做机器翻译（Mini版）**

```python
# 目标：英文数字 → 中文数字（玩具任务，但结构完整）
# "one two three" → "一 二 三"

# 这个任务够小（数分钟训练完），但包含完整的Seq2Seq结构：
# Encoder → Decoder → Cross-Attention → 自回归生成
```

**复盘清单（留1h）**
- 整理Day 9–11笔记：RNN/LSTM/Transformer各写一张"模型卡片"
  - 架构图（手画）
  - 解决了什么问题
  - 局限性是什么
  - 适用场景
- 白板测试：默写Attention公式 + ResBlock结构（不看资料，限时5分钟）

#### 今日检查清单
- [ ] Mini翻译模型能正确翻译数字
- [ ] 3张模型卡片写完
- [ ] 已掌握内容 vs 还需巩固的内容列清楚

---

## 阶段二：进阶期（Day 13–20，每天5小时）

### Day 13｜优化算法深入（5h）

**今日目标**：从"知道Adam"升级到"理解为什么用Adam，以及什么时候不用"

#### 学习内容（2.5h）
- D2L Chapter 12: 优化算法（12.3–12.11全部）
- 重点：Adam、学习率调度、学习率warmup

#### 关键例子：Adam推导与实现
```python
# Adam = Momentum（一阶矩）+ RMSProp（二阶矩）+ bias correction
class AdamOptimizer:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩（梯度的EMA）
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩（梯度²的EMA）
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * g**2
            # Bias correction（早期t小时，m和v被低估了）
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

# 直觉：m_hat是"方向"（梯度的移动平均）
#       v_hat是"自适应步长"（梯度大的方向走慢点，稀疏梯度走快点）
```

#### 学习率调度对比
```python
# 三种常用策略（画出lr曲线感受差异）
schedulers = {
    'StepLR (每30epoch×0.1)': torch.optim.lr_scheduler.StepLR(opt, 30, 0.1),
    'CosineAnnealing':         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
    'OneCycleLR':              torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=100)
}
# 实际建议：默认用Cosine；fine-tuning用带warmup的Cosine
```

#### 实践任务（2.5h）
- 在同一ResNet上对比5种优化器（SGD/Momentum/AdaGrad/RMSProp/Adam）
- 实现带warmup的Cosine调度（前5个epoch线性升，之后cosine降）
- 实验：gradient clipping对LSTM梯度爆炸的效果

#### 今日检查清单
- [ ] 能推导Adam的更新公式（含bias correction）
- [ ] 理解SGD+Momentum在某些场景优于Adam（如CV fine-tuning）
- [ ] warmup + cosine调度实现并可视化

---

### Day 14｜BatchNorm + 训练工程技巧（5h）

**今日目标**：掌握让模型从"跑通"到"跑好"的工程技能

#### 学习内容（2h）
- BatchNorm深入：原理、从零实现、train/eval区别
- LayerNorm vs BatchNorm vs GroupNorm 对比
- 混合精度训练（AMP）

#### 关键例子：BatchNorm手写实现
```python
class MyBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))
        self.eps, self.momentum = eps, momentum
    
    def forward(self, x):
        if self.training:
            # 训练时：用当前batch统计量
            mean = x.mean(0)
            var  = x.var(0, unbiased=False)
            # 更新running统计（推理时用）
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean.detach()
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var.detach()
        else:
            # 推理时：用全局running统计量
            mean, var = self.running_mean, self.running_var
        
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta

# 面试常考：train和eval模式下BN行为有何不同？
# → 训练用batch统计，推理用running统计（更稳定）
```

#### Norm对比卡片
| Norm | 统计维度 | 适用场景 |
|------|---------|---------|
| BatchNorm | 跨样本（N维） | CV（batch较大时稳定） |
| LayerNorm | 跨特征（C维） | NLP/Transformer（序列长度不固定） |
| GroupNorm | 跨部分通道 | 小batch CV（目标检测） |

#### 实践任务（3h）
- 用 `torch.profiler` 找出训练瓶颈（数据IO vs 前向 vs 反向）
- 实现AMP（混合精度）训练，对比纯FP32的速度和显存
- 完善Day 4的训练框架，加入：TensorBoard日志 + 早停 + AMP

#### 今日检查清单
- [ ] BN的train/eval区别能清楚解释（这是面试陷阱题！）
- [ ] LayerNorm vs BatchNorm的适用场景
- [ ] AMP训练跑通，能量化速度提升

---

### Day 15｜迁移学习 + 计算机视觉应用（5h）

**今日目标**：掌握fine-tuning——MLE日常工作中用得最多的CV技能

#### 学习内容（2h）
- D2L Chapter 14.1–14.2: 图像增广 + 微调（精读）
- Fine-tuning最佳实践：逐层解冻（Progressive Unfreezing）

#### 关键例子：Fine-tuning全流程
```python
# 场景：用预训练ResNet-50识别小数据集（假设10类，每类200张）
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')

# 第一阶段：冻结骨干，只训新头（5 epochs，lr=1e-3）
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# 第二阶段：解冻layer4，用小lr fine-tune（10 epochs）
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # 预训练层：小lr
    {'params': model.fc.parameters(),     'lr': 1e-3},  # 新层：大lr
])

# 第三阶段（可选）：解冻整个网络，lr再小一个数量级
# 关键原则：越靠近输入的层，特征越通用，越不需要改动
```

#### 目标检测基础（面试了解）
```python
# IoU（Intersection over Union）手写
def compute_iou(box1, box2):
    # box格式：(x1, y1, x2, y2)
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return inter_area / (area1 + area2 - inter_area)
```

#### 实践任务（3h）
- 在Oxford-IIIT Pets（37类）上fine-tune ResNet-50，精度 > 90%
- 对比三种策略：全微调 / 只训最后层 / 逐渐解冻
- 手写IoU，验证对几个边框对的计算

#### 今日检查清单
- [ ] 三种fine-tuning策略对比实验完成
- [ ] 理解为什么ImageNet预训练的特征能迁移到其他领域
- [ ] IoU代码写完，自测通过

---

### Day 16｜NLP + Word2Vec + BERT（5h）

**今日目标**：从词向量到BERT，理解预训练语言模型的演进逻辑

#### 学习内容（2.5h）
- D2L Chapter 15.1–15.4: Word2Vec + GloVe（理解词向量的本质）
- D2L Chapter 16.1–16.4: **BERT**（精读）⭐
- HuggingFace Transformers库快速入门

#### 关键例子：Word2Vec的直觉
```python
# Word2Vec的核心：分布式假设（词义由上下文决定）
# Skip-gram：用中心词预测上下文
# CBOW：用上下文预测中心词

# 训练完后的神奇性质：
# king - man + woman ≈ queen（向量空间中的语义运算）

# BERT的两个关键创新：
# 1. 双向：同时看左右上下文（GPT只看左边）
# 2. 预训练任务：MLM（完形填空）+ NSP（下句预测）

# BERT fine-tuning（最简版）
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

model     = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize + fine-tune + evaluate（约30行代码搞定）
```

#### BERT vs GPT vs T5
| 模型 | 架构 | 预训练 | 适合 |
|------|------|--------|------|
| BERT | Encoder | MLM + NSP | 理解（分类/NER/QA） |
| GPT | Decoder | 语言模型（LM） | 生成 |
| T5 | Encoder-Decoder | 文本转文本 | 通用（seq2seq） |
| RoBERTa | Encoder | 更多MLM（去掉NSP） | BERT的改进版 |

#### 实践任务（2.5h）
- 用HuggingFace BERT在SST-2上fine-tune，精度 > 93%（只需约30行代码）
- 对比：LSTM方案 vs BERT方案在同一数据集上的精度 + 训练时间
- 可视化BERT的[CLS] token embedding（用t-SNE降维），看不同类别是否分离

#### 今日检查清单
- [ ] 能解释BERT为什么双向，GPT为什么单向
- [ ] BERT fine-tune完成，精度 > 93%
- [ ] 理解Tokenization中的WordPiece/BPE（面试有时会问）

---

### Day 17｜生成模型 + ML系统设计框架（5h）

**今日目标**：了解VAE/GAN原理；重点掌握ML系统设计——MLE面试核心

#### 上午（2h）：生成模型
- D2L Chapter 20.2–20.3: VAE + GAN基础

```python
# VAE的关键：编码成分布，而不是点
# 损失 = 重构损失（像素级） + KL散度（让latent接近标准正态）
vae_loss = reconstruction_loss + beta * kl_divergence
# KL散度的作用：让latent space连续可插值，避免"空洞"

# GAN的博弈：
# G（生成器）：想骗过D    → 最小化 log(1 - D(G(z)))
# D（判别器）：想识别真假 → 最大化 log(D(x)) + log(1 - D(G(z)))
# 纳什均衡：D(G(z)) = 0.5，即D无法区分真假

# 实践任务：VAE在MNIST生成数字（精度不重要，能看出形状即可）
```

#### 下午（3h）：ML系统设计框架
**这是MLE面试和DS最大的差距所在！**

```
7步回答框架（每题30-45分钟）：

① 澄清需求（2min）
   "这个系统的核心业务目标是什么？DAU规模？延迟要求？"
   
② 定义ML问题（3min）
   任务类型：分类/排序/回归/生成？
   在线 vs 离线？
   核心指标：offline（AUC/NDCG）和 online（CTR/留存率）各是什么？
   
③ 数据（3min）
   数据来源 + 标注策略（人工/弱监督/用户行为）
   特征：user侧 / item侧 / context侧
   数据不平衡：正负样本比例，如何处理？
   
④ 模型（5min）
   Baseline → 复杂模型（先简后繁！面试官喜欢听到这个）
   Trade-off：精度 vs 延迟 vs 可解释性
   
⑤ 训练（3min）
   离线训练频率（实时/小时/天）
   负采样策略（随机 vs hard negative mining）
   
⑥ 线上服务（3min）
   特征存储（Feature Store：在线低延迟 vs 离线批量）
   模型版本管理 + 灰度发布
   
⑦ 监控（2min）
   数据漂移检测（PSI/KS test）
   性能回退自动报警 + 回滚机制
```

**练习题（口述30分钟）**：设计"微博信息流排序系统"

#### 今日检查清单
- [ ] VAE实现完成，能生成模糊但可辨认的数字
- [ ] 7步框架背熟，能不看笔记流畅讲出来
- [ ] 针对"微博信息流"，完整走一遍框架

---

### Day 18｜模型部署 + 高频编码题（5h）

**今日目标**：打通从训练到部署的最后一公里，同时练熟编码题

#### 上午（2h）：模型部署基础
```python
# 1. 导出ONNX（跨框架部署）
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}})

# 2. 模型量化（减小体积，加速推理）
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
# 通常可减小75%体积，速度提升2-4x（CPU上）

# 3. 用FastAPI部署（最简版）
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return {"label": int(logits.argmax()), "confidence": float(logits.softmax(-1).max())}

# uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 下午（3h）：高频编码题
```python
# 必须能在20分钟内默写的题目清单：

# 题1：数值稳定的Softmax
def softmax_stable(x):
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# 题2：Cross-Entropy Loss
def cross_entropy(y_hat, y):
    return -torch.log(softmax_stable(y_hat)[range(len(y)), y]).mean()

# 题3：手写BatchNorm（见Day 14）
# 题4：手写Attention（见Day 10）
# 题5：手写ResBlock（见Day 7）
# 题6：实现Dropout
def dropout(x, p, training):
    if not training or p == 0: return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)  # 除以(1-p)保持期望值不变

# 题7：计算Conv输出shape（需要能心算）
# 题8：IoU（见Day 15）
```

#### 今日检查清单
- [ ] ONNX导出跑通
- [ ] FastAPI接口部署成功（本地测试）
- [ ] 8道编码题都能在20分钟内独立完成

---

### Day 19｜模拟面试（5h）

**今日目标**：模拟真实面试环境，找出最后的知识盲区

#### 上午（2.5h）：理论面试自测

每题口头作答，录音后回放检查：

**基础理论**（每题2-3分钟）
- 为什么用ReLU而不是sigmoid？
- Dropout的作用机制是什么？为什么有效？
- BatchNorm和LayerNorm分别在什么场景用？
- L1和L2正则化的区别？各产生什么类型的解？

**CNN/RNN/Transformer**（每题3-4分钟）
- ResNet为什么能训练很深的网络？
- LSTM是如何解决RNN梯度消失的？
- Transformer的Self-Attention时间复杂度是多少？为什么？
- BERT和GPT的根本架构区别是什么？

**ML工程**（每题3-5分钟）
- 如何处理训练集中的类别极度不平衡？
- 模型上线后发现精度下降，可能的原因和排查步骤？
- 如何debug一个不收敛的模型？（系统回答，至少5个点）

**Debug不收敛系统回答（必背）**
```
Step 1: 先用最小数据集（甚至1个batch）过拟合
        → 如果能过拟合：问题在数据/正则，不在模型
        → 如果不能过拟合：模型有bug，逐层检查
Step 2: 检查梯度（是否消失/爆炸）
        → torch.autograd.set_detect_anomaly(True)
Step 3: 检查学习率（过大震荡，过小不动）
        → LR Finder或直接试1e-1, 1e-3, 1e-5
Step 4: 检查数据（标签是否对齐？是否归一化？）
Step 5: 检查损失函数（是否适合任务？是否有数值溢出？）
```

#### 下午（2.5h）：系统设计模拟
用7步框架回答以下两题（各45分钟，含写要点）：
1. 设计"抖音视频推荐系统"
2. 设计"垃圾邮件检测系统"

#### 今日检查清单
- [ ] 所有理论题都能流畅作答（录音自评）
- [ ] 两道系统设计题走完完整框架
- [ ] 整理出最后的知识盲区，今晚补完

---

### Day 20｜最终项目（5h）

**三选一，产出GitHub上可展示的作品**

#### Option A（CV向）：图文对比学习（CLIP-lite）
```python
class CLIPLite(nn.Module):
    """用对比学习让图像和文本描述的embedding靠近"""
    def __init__(self, embed_dim=128):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        self.image_enc = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        self.text_enc  = ...  # 简化版Transformer（4层）
        self.img_proj  = nn.Linear(2048, embed_dim)
        self.txt_proj  = nn.Linear(512,  embed_dim)
    
    def forward(self, images, tokens):
        img_feat = F.normalize(self.img_proj(self.image_enc(images)), dim=-1)
        txt_feat = F.normalize(self.txt_proj(self.text_enc(tokens)),  dim=-1)
        # InfoNCE Loss：对角线是正样本，其余是负样本
        logits = img_feat @ txt_feat.T * temperature
        labels = torch.arange(len(images))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
```

#### Option B（NLP向）：情感分析系统（含API serving）
- BERT fine-tune（精度 > 94%）
- FastAPI接口 + 健康检查 + 错误处理
- 简单的HTML前端（输入框 + 结果展示）

#### Option C（推荐系统向）：双塔召回模型
```python
# 在MovieLens-1M上，评估 HR@10 和 NDCG@10
class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Embedding(n_users, embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Embedding(n_items, embed_dim),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, user_id, item_id):
        u = F.normalize(self.user_tower(user_id), dim=-1)
        v = F.normalize(self.item_tower(item_id), dim=-1)
        return (u * v).sum(-1)  # 余弦相似度
```

#### GitHub仓库必须包含
```
final_project/
├── README.md          # 清晰介绍：问题 + 方法 + 结果表格
├── requirements.txt
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── EDA.ipynb      # 含可视化
└── results/
    └── metrics.json   # 实验结果
```

#### README核心结构
```markdown
## 结果
| 方法 | 指标 | 说明 |
|------|------|------|
| Baseline | 72% | Logistic Regression |
| 本方法 | 93% | BERT fine-tune |

## 快速开始
pip install -r requirements.txt
python src/train.py

## 模型架构
（一张架构图，胜过千行文字）
```

---

## 进度衡量标准

### 每阶段通关测试

**阶段一结束（Day 12）**
- [ ] 白板默写：ResBlock / LSTM cell / Attention公式（不看资料，5分钟内）
- [ ] CIFAR-10精度 > 88%（项目在GitHub上）
- [ ] 能向非ML背景的人解释"为什么ResNet比普通CNN好"

**阶段二结束（Day 20）**
- [ ] 8道编码题全部20分钟内完成
- [ ] 7步框架能流畅讲30分钟不露怯
- [ ] GitHub上有3个有star价值的项目
- [ ] 模拟面试录音自评：能连续回答且解释清楚

---

## 初学者常见错误（7个必读）

**错误1：跳过基础直接看高阶**
`❌` 第一天就想跑BERT｜`✅` 先把线性回归每行代码理解透彻  
原因：基础不牢，遇到bug完全不知道从哪查起

**错误2：只跑通，不理解**
`✅` 删掉某一行，预测会发生什么，再实际验证——这是最好的理解方式

**错误3：忘记切换train/eval模式**
```python
# 推理时必须：
model.eval()
with torch.no_grad():
    output = model(x)
# 否则Dropout和BN行为不对，精度会莫名下降
```

**错误4：GPU设备不一致**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X, y = X.to(device), y.to(device)  # 数据也要移到同一设备！
```

**错误5：过拟合时换更复杂的模型**
`✅` 正确顺序：先加正则（Dropout/数据增强/L2） → 再考虑换模型

**错误6：学习率选择靠感觉**
```python
# Adam经验范围：
# 从头训练：1e-3  |  Fine-tuning预训练层：1e-5 ~ 1e-4
# loss震荡 → lr太大  |  loss几乎不动 → lr太小或梯度消失
```

**错误7：面试时背公式而不解释直觉**
```
❌ "BN的公式是 (x-μ)/σ * γ + β"
✅ "训练中每层输入分布不断变化，BN稳定了这个分布，
    让后面的层更容易学习。γ和β是可学习参数，
    允许模型在必要时恢复原始分布的尺度。"
```

---

## 免费工具 & 资源

### 核心教材
- D2L中文版：https://zh.d2l.ai
- D2L配套代码：https://github.com/d2l-ai/d2l-zh

### 免费算力
| 平台 | GPU | 额度 |
|------|-----|------|
| Kaggle Notebooks | P100 | 30h/周，无需科学上网 |
| Google Colab | T4/L4 | ~12h/天 |
| Hugging Face Spaces | CPU | 免费，适合部署demo |

### 常用工具
```bash
pip install torchinfo       # 打印模型结构和参数量
pip install tensorboard     # 训练可视化
pip install wandb           # 实验管理（免费个人账户）
pip install torch-lr-finder # 自动找最优学习率
pip install transformers    # HuggingFace预训练模型
pip install datasets        # HuggingFace数据集
```

### 面试资料
- ML面试圣经：https://huyenchip.com/ml-interviews-book/
- 李沐论文精读：https://space.bilibili.com/1567748478
- 系统设计：《Machine Learning System Design Interview》（书）

---

## 每日检查清单（打印贴桌面）

```
📅 开始前（10分钟）
□ 回顾昨天3个关键点（不看笔记，凭记忆）
□ 写下今天具体要完成的任务

📚 学习（2.5h / 3h）
□ 读D2L + 跑代码（边读边运行，不要纯看）
□ 遇到不懂：先想5分钟，再查
□ 关键公式/结构图：手写在纸上

💻 实践（1.5h / 2h）
□ 完成今日实践任务
□ 代码能跑通，结果符合预期
□ commit到GitHub

🌙 结束后（10分钟）
□ 写"今日顿悟"（1-3条，用自己的话）
□ 写"明天要复习"（疑问点）
```

---

*DS → MLE New Grad｜20天 88小时｜前12天4h/day，后8天5h/day*
