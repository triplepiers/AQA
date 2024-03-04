## 2023: FSPN

### 1 Abstract

- 先前的工作

    - 大多数基于 粗粒度(coarse-grained)特征 进行训练、采用 holistic video representations，缺乏对 fine-grained intra-class variations 的捕捉

    - Parmar and Morris 认为所有的 sub-action sequences 对结果具有 <u>相等的贡献</u>

    - segmenting action sequences along with their temporal dependence remains a challenging task：

        1. 缺少预定义的 标签 & action sequence 之间的关联性

        2. sub-action sequences 具有非常细的粒度，动作间的变化十分平滑 => 难以确定其边界

        3. 由于动作十分细密、在相似的背景中进行，各 sub-action 之间有较多的共同 attributes

- 创新点

    提取 fine-grained sub-action sequence 和它们的 temporal dependencies 有助于做出更准确的估计

    > 为了降低背景的干扰，本文使用预训练模型从 input video 中提取了 actor-centric regions 

    1. 提出了由两部分组成的 FSPN：
    
        - <u>intra-sequence</u> action parsing module <span style="color:red;">**无监督**</span>
        
            对更细粒度下的 sub-actions 进行挖掘
            
            实现 semantical sub-action parsing，从而更准确的描述动作序列间的细微差别
        
        - spatiotemporal multiscale transformer module

            > 低阶特征缺乏语义信息，高阶特征难以对 sub-action 进行细粒度描述

            学习 <u>motion-oriented</u> action features、挖掘其在不同时间范围内的 long-range 依赖关系

    2. 提出了一个 group contrastive loss

    此外，由于整个动作序列可能存在组件重复 ABBBBCC，模型使用了 1D Temporal Convolution + Transformer Network 来提取 single-scale feature

    最终，各阶段特征会通过 multiscale temporal fusion 聚合生成 unified feature represen- tation，并用于最终的预测

### 2 Relative Works

1. AQA

    - Regression Formulation

    - Pairwise Ranking Formulation

2. (Fine-grained) Action Parsing 

    - Zhang: Temporal Query Networks => 通过 query 找出相关的 segments

    - Dian: TransParser =>  对 sub-action 进行挖掘（无监督）

3. Vision Transformer

    从 低分辨率图片 & 较小的通道数量 开始，逐渐增加通道并减少 spatial resolution

### 3 Approach

![](./assets/FSPN%20Pipeline.png)

#### 问题定义

对于给定 input video $x_i \in \mathbb{R}^{T \times H \times W \times C}$ 及其对应的分数标签 $y_i$，AQA 问题可以认为是一个回归问题：

> T, H, W, C 分别为 clip 长度、视频宽高、通道数

1. 使用预训练模型 $D(.)$ 得到运动员所在的 BBox $x_a$

    $$
        x_a = D(x_i) \in \mathbb{R}^{T \times H \times W \times C}
    $$

2. 将 原始输入 和 BBox 都输入 (相同的)I3D 来提取 spatiotemporal 特征 $(f_i, f_a)$

    $$
    f_i = E_v(x_i),\ f_a = E_v(x_a)
    $$

3. 使用 FSPN $\mathbb{F}_\Theta(.)$ 提取特征，并最终进行回归运算

    $$
    \overline{y}_i = R_{\theta}(\mathbb{F}_\Theta(E_v(x_i)),\mathbb{F}_\Theta(E_v(x_a)))
    $$


#### Intra-Sequence Action Parsing

##### 1 Intra-Sequence Action Parsing (IAP)

!!! tip "确定每个 sub-action 的 起始帧 & 结束帧"

- 给出的 Parser 可以对 $S$ 个 sub-action 的分布概率进行预测，同时指出 “转变” 发生的具体帧编号 $f^{th}$：

    features -> probability vec $A_s$ (对 $s^{th}$ sub-action 的 middle-level 表示)

    $$
    [A_1, ..., A_s] = IAP(f_i,f_a)
    $$

- 使用 up-sampling decoder + MLP layers projection head 构建 “分布概率预测器”

    - 上采样包含四个 spatial-temporal dimensions 分别为：(1024, 12), (512, 24), (256, 48), and (128, 96) 的子块
    
        - temporal axis 会被卷积操作扩充

        - spatial dimensions 会被 Max Pooling 削减

    - 使用了 3 Layer MLP

- $A_s(\vec{t})$ 表示 $t^{th}$ 帧可能对应的 sub-action 概率分布；$\vec{t}_s$ 是对 $s^{th}$ 跳 action sequence 的预测结果

    > $\text{argmax } A_s(\vec{t})$ 即为该帧最可能对应的 sub-action 类型
    >
    > 此时 t 与 t+1 必然对应不同的 sub-aciton => 新的 sub-action instance 从 $(t+1)^{th}$ 帧开始

    $$
    \vec{t}_s = \text{argmax } A_s(\vec{t}),\ \frac{T}{S}(s-1) \leq \vec{t} \leq \frac{T}{S}s
    $$

    上式保证了 $\vec{t}_1 \leq ... \leq \vec{t}_s$

##### 2 Group Contrastive Learning

- 上一步得到的 $(f_i,f_a)$ 共享了较多的语义信息，直接对其进行比较学习会导致 <u>对具有相同语义的动作序列学习得到不同的表示</u>

    => 使用 Group Contrastive Learning，对具有相似 sub-action seq 的视频进行对比

- 输入的 $(f_i,f_a)$ 会被

    - 赋予与其动作、语义具有最大相似度组别的 Pseudo Label $p$
    - 最终生成 sub-action sequence $(\overline{f}_i,\overline{f}_a)$

- 具有相同 Pseudo Label 的 feature 将组成如下的 group：

    $$
    G^p_k = \frac{\sum_{i=1}^B g(A_s^k)}{T_B},\ \text{where } p=A_s^k
    $$

    - $g(.)$ 是序列 $A_s^k$ 的 logits

    - $T_B$ 是 Group $B$ 的序列总量

- 我们使用 Group 的 average representation $G_f^g$，定义：

    - positive-pair：$G_a^p, G_i^p$ （上标组别相同）

    - negative-pair：$G_a^p, G_k^q$ （上标组别不同）

        > Action Seq 相同，但 Sub-action Seq 不同


#### Spatio-Temporal Multiscale Transformer

!!! tip "输入 sub-action sequence $(\overline{f}_i,\overline{f}_a)$，并在不同 scale 上挖掘 long-range dependencies"

##### 1 Actor-Centric Multiscale Transformer

- 本文提出的 Transformer 各阶段具有各异的 channel resolution => channel & scale 逐渐增加

- 该 Multiscale Transformer，由 3 * stage（结构相同）构成
    
    - 每个 stage 

        - 在 early Layer 处理粗粒度特征，并在更深层处理细粒度特征
    
        - 各包含了 3 个 transformer block + 8 attention head，用于处理相同 scale 的信息并生成 Attention 值

            $$
            \hat{f}_i += MLP(LN(attention)), attention = Multihead(LN(\frac{Q_a(K_i)^T}{\sqrt{d_k}})V_i) + \overline{f}_i
            $$

            <center>$Ln(.)$ 为 Layer Norm 操作，$MLP$ 为两层使用 GELU 激活函数</center>

            <center><img src="../assets/FSPN%20block.png" style="max-height:250px"></center>
            
            <center>block 结构：$\overline{f}_a = query, \overline{f}_i=memory$</center>

    - 对于输入特征 $f \in \mathbb{R}^{T' \times C'}$:

        1. 进行一次 1D 卷积 (kernel=3, stride=1)

        2. 使用包含 $L$ 个 block 的 Multiscale Transformer，其中 stage n 的 output_shape = $T' \times 2^nC'$（放大）

            > channel dimesion 会在 stage 切换时通过 MLP 扩大 2 倍

##### 2 Multiscale Temporal Fusion

<center><img src="../assets/FSPN%20fusion.png" style="max-width:500px;"></center>

对于第 $n$ 层的 output shape 为 $F_n = T' \times 2^nC'$，而第 $n+1$ 层则为 $F_{n+1}= T' \times 2^{n+1}C'$

- 为了成功进行 aggregate，我们需要进行 upsampling: $U_\varphi(F_n) = \text{Upsampling}(R_n\mathcal{W}^n)$

- 随后使用 element-wise addition 更新 $F_{n+1} = U_\varphi(F_n) + F_{n+1}\mathcal{W}^{n+1}$

最终通过 fuse 的到的 intergrated feature $\mathcal{F} = \text{Concat}(F_1, ..., F_{N})$

### 4 Optimization

#### Overall Training Loss

最终将由 2-Layer MLP 对 $\text{MaxPool}(\mathcal{F})$ 预测得到分组标签 $\gamma_i$ 和 回归分数 $y_i$

- 对于回归预测，有：

    $$
    \begin{align*}
    L_{bce} &= - \sum_{i=1}^I(\gamma_i log(\overline{\gamma_i}) + (1-\gamma_i)log(1-\overline{\gamma}_i)) \\
    L_{reg} &= \sum_{i=1}^I \| \overline{y}_i - y_i\|^2,\ \text{where } \gamma_i = 1
    \end{align*}
    $$

- 对 Group Contrastive，有：

    $$
    L_{gc} = - \log{\frac{
        h(G_a^p,G_i^p) / \tau
    }{
        h(G_a^p,G_i^p) / \tau + \sum_{q=1,k}^S  h(G_a^p,G_k^q) 
    }}, \text{ where }p\neq q
    $$

    - $h(.) = \exp{(\text{cosine similarity})}$ 

    - $\tau$ 是 teperature 超参数
