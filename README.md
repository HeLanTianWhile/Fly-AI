# Fly AI · 从零实现的图像生成 AI（条件变分自编码器 CVAE）

一个 **不使用任何 AI 封装库**（torch / tensorflow / keras 等）的图像生成系统，
**纯 Python + NumPy** 手工实现深度学习的前向、反向传播、训练与生成。

**核心能力**：输入一个数字 `0–9`，生成一张 **100×100** 的手写数字风格图像。
本阶段先做「数字生成」，架构是通用的，可扩展到更多类别。

---

## 1. 为什么选「条件 VAE」而不是 Diffusion / GAN

| 方案 | 数据需求 | 训练稳定性 | 算力 | 生成速度 |
|------|---------|-----------|------|---------|
| 条件 VAE（本实现） | **少** | **稳** | 低 | 一次前向，毫秒级 |
| Diffusion | 中 | 稳 | 高（多次迭代） | 慢（多次去噪） |
| GAN | 中 | 易崩 | 中 | 快 |

在你要求「训练数据少 / 算力低 / 纯 CPU」的前提下，条件 VAE 是最合适的起点。

---

## 2. 自研「全新架构」要点（都不是现成封装）

1. **微型自动微分张量** `cvae/tensor.py`
   手写反向图 + 链式法则，前向记录父节点，反向拓扑排序传梯度。整套深度学习由此搭建。
2. **自研组件** `cvae/layers.py`
   - `Dense` 全连接
   - `Conv2D` 二维卷积（im2col 矩阵乘法）
   - `Conv2DTranspose` 转置卷积（scatter 累加，精确上采样）
   - `BatchNorm2D` 批归一化（含训练/推理 running stats）
   - SiLU / ReLU 激活
3. **条件码空间** `cvae/model.py`
   - `CondEmbed`：数字标签 → 可学习嵌入向量
   - **渐进式条件调制（Progressive Conditioning）**：解码器在**每一个上采样阶段**
     再次注入标签嵌入（`FeatureMap *= (1 + cond·wa) + cond·wb`），
     让"数字类别"对整幅图（从 7×7 到 100×100）都有区分度，而不只是 latent 注入一次。
4. **自研训练目标** `cvae/losses.py`
   高斯似然重构 + KL 散度（beta-VAE 可调）。
5. **自研优化器** `cvae/optimizer.py`：Adam。

### 形状链路（100×100 输入）
```
encoder: 100→50→25→13→7   通道 16,32,64,64   → 48 维 latent (+ 类别嵌入)
   （每级：Conv2D stride2 + BatchNorm + SiLU）
decoder: 7→13→25→50→100   转置卷积 + BN + 渐进条件调制
   （latent + 条件  →  100×100 单通道灰度图 [0,1]）
```
参数量约 **0.6M**，CPU 友好。

---

## 3. 安装与运行

```bash
pip install numpy            # 唯一第三方依赖（标准库之外）
```

### 快速体验（合成数据，无需外网，几十秒）
```bash
python demo.py               # 训练一个小模型 + 生成 0..9 各一张 100x100 图像
```
`demo.py` 会展示：训练 loss 下降 → 生成 10 张 100×100 图像（PNG）→ 终端 ASCII 预览，
并直接验证「输入数字 → 输出与之对应的 100×100 图像」。

### 训练（真实手写数字 MNIST）
```bash
python main.py train \
  --epochs 30 --per_class 300 --batch 16 \
  --data_dir ./data --out ./checkpoints
```
- `--per_class`：每个数字采样的图像张数（控制数据量，缺省 300 → 共 3000 张）
- 首次运行自动下载公开手写数字数据集 MNIST（仅数据，非 AI 库）；
  若网络不通/下载失败，会自动回退到合成数据以便先跑通流程。
- 每个 `sample_every` 轮导出 `samples_e*.npy`（10 个数字各 1 张 100×100）。

### 生成
```bash
python main.py generate --digit 5 \
  --model ./checkpoints/latest.pkl --out generated/5.png
```
保存 `5.png`（100×100）并在终端打印 ASCII 预览。

### 终端 ASCII 预览
```bash
python main.py ascii --digit 3 --model ./checkpoints/latest.pkl
```

---

## 4. 目录结构

```
.
├── main.py                 # CLI：train / generate / ascii
├── demo.py                 # 快速演示（合成数据训练 + 生成 10 张 100x100）
├── cvae/
│   ├── tensor.py           # 自研自动微分张量
│   ├── layers.py           # Dense / Conv2D / Conv2DTranspose / BatchNorm
│   ├── model.py            # 自研 CVAE（含 CondEmbed + 渐进条件调制）
│   ├── losses.py           # 重构似然 + KL
│   ├── optimizer.py        # Adam
│   ├── data.py             # MNIST / 少量采样 / 重采样到 100×100 / 合成回退
│   ├── train.py            # 训练循环 + 保存/加载
│   ├── generate.py         # 生成 + 纯手工 PNG 编码 + ASCII 预览
│   └── gradcheck.py        # 数值梯度校验（验证反向传播正确）
├── requirements.txt
└── README.md
```

---

## 5. 自研反向传播已验证

`python -m cvae.gradcheck` 对每个自研组件做了**数值梯度校验**
（中心差分 vs 解析梯度），确认：
- `Conv2D`（输入/权重/偏置）梯度正确
- `Conv2DTranspose`（输入/权重/偏置）梯度正确
- `BatchNorm2D`（输入/γ/β）梯度正确（输入梯度用 float64 精密校验）
- `Dense`（权重/偏置）梯度正确

---

## 6. 实测验证结果

在纯 CPU、合成「每类一个不同大小的方框」数据、每类 250 张、12 epoch、base=8 的条件下：

| 指标 | 初值 | 收敛后 |
|------|------|--------|
| 总 loss | 1.38 | 0.014 |
| 重构 | 0.71 | 0.011 |
| KL | 0.67 | 0.003 |

生成时每个数字对应的方框大小随数字单调放大（数字 0 最小、数字 9 最大），
证明**条件机制（CondEmbed + 渐进条件调制）确实让“类别”刻画到整幅 100×100 图像**：

```
digit 0: coverage= 0.6%    digit 5: coverage=15.2%
digit 1: coverage= 1.9%    digit 6: coverage=20.1%
digit 2: coverage= 3.9%    digit 7: coverage=25.8%
digit 3: coverage= 6.1%    digit 8: coverage=31.9%
digit 4: coverage=10.1%    digit 9: coverage=38.6%
```
> 注意：本项目为**从零跑通完整深度学习训练与生成**的演示/教学用途。在少量数据
> （每类几百张）下，质量不如大规模专用扩散模型；如需真实手写数字 `100×100` 效果，
> 请先让 `main.py train` 用 MNIST 数据训练更多 epoch。

---

## 7. 扩展到新类别

1. 训练数据换成你的类别标签（`0..K-1`）。
2. 调用 `CVAE(num_classes=K)` 实例化。
3. 其余不变——解码器的条件调制会为每个类别学出一条专用刻画。

---

## 8. 参数速查

| 参数 | 含义 | 建议 |
|------|------|------|
| `--latent` | 解码 latent 维数 | 32–64 |
| `--base` | 卷积基础通道数 | 8–16（越大越精细但越慢） |
| `--beta` | KL 权重 | 0.5–2.0（小→更还原，大→更规整） |
| `--sigma` | 重构伪似然噪声 | 0.2–0.5 |
| `--z` (generate) | 采样 latent 幅度 | 0.3–0.8 |

> 说明：本项目为教学/自研演示用途，重点在于**从零跑通完整的深度学习训练与生成**，
> 不追求在少量数据下达到与大规模专用模型相当的质量。
