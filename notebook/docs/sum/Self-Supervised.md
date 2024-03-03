# 自监督学习 SSL

## 2023: PECoP

### 1 Abstract

- 先前的工作

    - 因为有标签的 AQA 数据很少，先前的工作一般基于在 large-scale domain-general dataset 上预训练的模型进行优化

        => 在存在较大 domain shift 时，模型的 generalisation 较差

    - 侧重点不同：

        - 在 Parkinson’s Disease 严重性评估中，运动节奏中的一两次间断都会对 quality score 产生极大影响

        - 而在 pretraining task（动作分类）中，轻微（甚至更加严重的）区别并不会影响 action classification

    - Continual Pretraining

        主要用于 NLP，以通过 domain-specific unlabeled data 训练任务专精的模型

        这种方法要求更新所有参数，同时存储针对所有 separate task 的 params

    - BatchNorm Tuning

        支持通过仅调节 BatchNorm Layer 参数获取 domain-specific model

        但是在 较小的 / 更加 domain-specific 的 AQA 数据集中不起作用

- 创新点

    PECoP 通过增加一个额外的预训练过程来 reduce domain shift（专注于特定 AQA 任务）：

    - 往预训练的 3D CNN 外面包一层 3D-Adapters <u>自监督学习</u> spatiotemporal & in-domain info

    - 只更新 Adapter 中的参数，预训练模型参数保持不变

        > 此处拉踩需要更新 <u>所有参数</u> 的 HPT

### 2 Relative Works

- AQA：如 Tang(USDL)，Yu（CoRe）

    大多数方法将 AQA 视为 监督学习（score）的 回归任务，并尝试降低裁判的主观影响
    
    !!! bug "这些方法都忽视了 base dataset(K400) 和 target dataset 之间的 domain gap"

- SSL for AQA

    近期的一些方法开始尝试 自监督学习（SSL）。在传统的 Regression Loss 外，在迁移学习中增加了 SSL Loss（不需要添加额外的数据标注）

    如 Liu 应用了 Self-Supervised Contrasive Loss

- Continual Pretraining

    相比于传统的迁移学习，Continual Pretraining 通过 in-domain SSL 增强了 domain shift 问题

    - Gururangan：证明了增加 in-domain data pretraining 对于文本分类性能的影响

    - Reed：证明了在更接近 target dataset 的数据集上进行 Continual Pretraining 可以加快收敛、提高鲁棒性

    - Azizi：结合了 supervised pretraining（on ImageNet） & intermediate contrastive SSL（on MedicalImages），得到了具有更佳泛化性的医学图像诊断器

- Adapters

    Transformer 架构的 lightweight bottleneck module，用于实现 parameter-efficient 迁移学习

    - Chen：
    
        - 提出 AdaptFormer 用于可量化的 img/video 识别任务

        - 提出 Conv-Adapter，使其可以应用于 2D CNN

### 3 Approach

![](./assets/PECoP%20Pipline.png)

- Training Set：

    1. $D_g$: large-scale, labelled, domain-general
    2. $D_t$: target video dataset in the AQA domain

    两个数据集分别对应具有 significant domain discrepancy 的学习任务 $T_g, T_t$

- Test Set: $D_q \subseteq D_t$, unlabelled

- Target：在 $D_g， D_t$ 上调参，以找到一个可用于 $D_q$ 的 transferable spatiotemporal feature extractor 

#### Domain-general pretraining

这一部分直接使用了 pretrained backbone model —— 在 K400 上经过监督学习得到的 I3D 模型

#### In-domain SSL continual pretraining

- 本文提出的 3D-Adapter 和 Transformer & 2D CNN 中的相应模块具有类似的结构，在此基础上添加了 3D Layers 以实现视频数据的 3D CNN 操作

- 在每一个 Inception Module 的 concatenation Layer 后插入一个 3D-Adapter 可以得到显著的性能提升

    > 事实上 R3D + 3D-Adapter 也能提升性能

<center>![](./assets/ReCoP%20Incept%20with%20Adapter.png)</center>
<center>添加 3D-Adapter 后的 Inception Module</center>

- 3D-Adapter modules 将被 **随机初始化** 

- 每一个 3D-Apater 使用了：

    - 可学习参数 $\theta_{down}$ —— downsampling · depth-wise · 3D convolution
    - 非线性激活函数 $f(.)$，如 ReLU
    - 可学习参数 $\theta_{up}$ —— upsampling · point-wise · 3D convolustion

- 相关参数如下：

    - $C_{in}, C_{out}$： 输入/输出 的 channel dimensions
    - compression factor $\lambda$： bottleneck dimension

因此，对于给定的 input feture vecor $h_{in} \in \mathbb{R}^{C_{in} \times D \times H \times W}$，3D-Adapter 将输出 $h_{out} \in \mathbb{R}^{C_{out} \times D \times H \times W}$

具体变换过程可描述为：

$$
h_{out} = \alpha \odot (\theta_{up} \otimes f(\theta_{down} \overline{\otimes} h_{in})) + h_{in}
$$

其中：

- $\otimes, \overline{\otimes}$ 分别表示 depth-wise / point-wise 卷积操作
- $\alpha \in \mathbb{R}^{C_{out}}$ 是可调超参数，初始化为 ones
- $\odot$ 表示 element-wise 乘法

---

Continual Pretraining 阶段：

!!! tip "只有 3D-Adapter 中的参数会被更新"

- 在集合 $D_q$ 中以 SSL 形式进行 —— 视频标签由模型自动生成

- 使用了 Video Segment Pace Prediction (VSPP) 进行了预处理，从而对 “以不同速度完成的动作” 进行对比，生成视 SSL 标签信息：

    - speed rate $\lambda_i$
    
    - Segment No. $\zeta_i$

!!! info "一些拉踩信息"
    1. 关于 SSL Pretext 的选取
    
        作者也尝试使用了 Constrastive-Based 的 RSPNet & 同样是 Transformatoin-Based 的 VideoPace，但表现都不及 VSPP

    2. 关于 BatchNorm Tuning（对比 Adapter）

        在小型数据库上容易过拟合


#### Supervised fine-tuning

!!! question "为啥还要 Fine-Tuning (?)"
    本文 3D-Adapter 只对 I3D 模块进行 Continual Pretraining，整个模型的参数还是需要微调

!!! tip "在这个阶段，<u>所有参数（预训练参数 & 3D-Adapter）</u> 将在 $D_t$ 上被微调"

- USDL / MUSDL

    - 特征 -> I3D backbone -> temporal Pooling -> SoftMax -> 预测分布

    - 使用 *预测分布* 与 *由 ground-truth 产生的高斯分布* 之间的 KL loss 进行优化

    > MUSDL 是 USDL 的 multi-path 版本，适用于 MTL-AQA / JIGSAWS 这种由多个裁判同时打分的数据集

- CoRe

    - 使用添加 3D-Adapter 的 I3D 模型进行特征提取，随后丢回 GART 进行回归

    - 最终结果取多个 exampler 的平均

- TSA

    使用添加 3D-Adapter 的 I3D 模型进行特征提取，随后丢回 Attention Module 进行后续操作
