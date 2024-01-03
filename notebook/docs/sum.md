# 汇总

## 1 TSA-Net

### 1-1 Abstract

- 现有工作的不足：

    1. 直接迁移 Action Recognition，忽略了最本质的特征 —— foreground & background info

        - HAR 的重点在于 —— 区分 *不同的动作*
        - QAQ 的重点在于 —— 对于 *特定动作* 的优缺点进行评估

        => 对两者采用相同的 特征提取 方式显然是不对的

    2. Feature Aggregasion 效率低

        受卷积操作的感受野大小限制，导致损失了 long-range dependencies (?)

        > 看起来是缺少时间信息

    3. RNN 可以通过隐藏态存储上下文，但是 *不能并行计算*

    <span style="color:red;">所以，AQA 需要一个 <u>高效的 Feature Aggregation 算法</u> ！</span>

- 创新点：

    1. 引入 single object tracker

    2. 提出 Tube Self-Attention Module(TSA)

        - 基于 tube 机制 和 self-attention 机制，实现高效的 feature aggregation

        - 通过 adopting sparse feature interactions 生成丰富的时空上下文信息

#### Attention-Based Feature Aggregation

1. Non-Local Opertaion

    每一个 position 会和 **所有 feature** 做 Dense Correlation

2. Tube Self-Attention Operation

    仅对各时刻下同一 spatio-temporal tube 内的每一个 postion 做 Sparse Correlation

    - tube 使模型专注于 feature map 的一个子集 —— 运动员（忽略背景）
    - self-attention 生成了 时间维度 上的 上下文信息

### 1-2 Approach

![](./assets/TSA%20Pipline.png)

1. 包含两个步骤：

    - 使用已有的 Visual Object Tacking 模型生成 tracking box，然后通过 feature selection 生成 tube（ST-Tube）

        假设输入的视频共有 $L$ 帧，$b_l$ 表示第 $l$ 帧中的 Bounding-Box

        $$
            V:\text{L-frames Video} \rightarrow VOT_{SiamMask} \rightarrow B:\text{BBoxs}\{b_l\}_{l=1}^L
        $$

    - 将视频划分为 N-Clip 交给 I3D-s1（立体卷积）生成 Feature Map

        假设 $L=MN$，我们将视频划分为 $N$ 个包含 $M$ 个连续帧的 Clip

        使用 I3D 算法可以从中生成 $N$ 个 features

        $$
            V:\text{N-fold Video} \rightarrow I3D \rightarrow X:\text{features}\{x_n\}_{n=1}^N
        $$

2. 对 Feature Map 中 tube 框定的部分应用 self-attention 机制，从而实现 feature aggregation

    ```text
        B: BBoxs    \
                     |--> TSA Module --> 包含上下文时空信息的 X': features (N个)
        X: features /
    ```

    > X 与 X' 等大：TSA Module 没有改变 Feature Map 的 shape
    >
    > => 这一特性允许我们<u>堆叠 TSA</u>以获得更丰富的时空信息

3. 将 Aggregated Feature 交给 I3D-Stage2 生成 $H$

    $$
        X' \rightarrow I3D_{s2} \rightarrow H: \text{features}\{h_n\}_{n=1}^N
    $$

4. Network Head

    1. 使用 AvgPooling 在 Clip 纬度上进行 fuse

        $$
            \overline{h} = \frac{1}{N} \sum_{n=1}^N h_n
        $$

    2. 迁移训练 MLP_Block 对 $H$ 进行打分

        $$
            \overline{h} \rightarrow \text{MLP-Block} \rightarrow Score
        $$


#### TSA Module

根据 Bounding-Box 对 I3D 生成的 feature-map 进行过滤，减少 self-attention 处理的数据量

> 他说这能去除背景噪声的干扰 => Non-local Module 会使用整个 Feature-Map，从而引入背景噪声

1. Tube 生成

    - 由于 I3D-s1 使用了 2 * temporal Pooling ($/2^2$)，Bounding-Box:feature-map X = 4:1

    - 问题：SiamMask 会产生 *平行四边形* 边框 -> 我们不好直接在 feature-map 里框方格

        解决： 使用如下的 alignment 策略生成 mask

        1. 将连续的 4 个 BBox 缩放到和 feature-map 一样的大小
        2. 分别生成 4 个 Mask：覆盖面积超过阈值 $\tau = 0.5$ 记 1，否则为 0
        3. 使用并集作为最终的 Mask' $M_{c,t}^{l\rightarrow l+3}$

    - 最终参与 self-attention 是所有 mask=1 的 feature: $\Omega_{c,t}$

        $$
            \Omega_{c,t} = \{(i,j)|M_{c,t}^{l\rightarrow l+3}(i,j)=1\}
        $$

2. Self-Attention

    生成了和 feature-map $x$ 等大的 $y$，输出两者的 residual $x' = Wy + x$
 
    <center>![](./assets/TSA%20Module.png)</center>
    <center>本文中的 TSA 算法</center>

3. Nertwork Head

    > 通过修改 MLP_Block 的输出大小 & Loss Func，Network Head 就可以兼容不同的任务

    - Classification：输出由 n_class 决定 + BCE Loss
    - Regression：输出 1 维向量 + MSE Loss
    - Score distribution prediction：
    
        - 在 USDL 中嵌入 TSA Module
        
        - Loss = Kullback-Leibler (KL) divergence of predicted score distribution and ground-truth (GT) score distribution

## 2 CoRe

### 2-1 Abstract

!!! info "对比学习 Constrastive Learning"
    学习一个 representation space，通过比较两个（same class）样本在 representation space 中的距离来衡量其 语义联系 (semantic relationship)

    > 因为只有同组的才能直接对比，所以需要 Group-aware

    本文训练模型用于 <u>回归预测 relative score</u> 并对两个 video 的 score 进行对比，从而学习其 diff 用于最终分数估计

- 现有工作：

    从 single video 回归得到一个 quality score（针对单一视频的回归任务），面临三个挑战：

    1. Score Label 是由人类裁判给出的，其主观评判会给分数估计造成极大困难

        suffter a lot from <u>large inter-video score variation</u>

    2. video 之间的差异是十分微小的：运动员往往在相似的环境下作出类似的动作

    3. 大多数解决方案使用 Spaerman's Rank 进行评估，这种方法可能不能正确反应评估表现

- 创新点：

    - 基于 reference to another video that has shared attributes 进行分数预测（不是直接从单个视频下手）

    - 提出了 CoRe：
    
        通过 pare-wise comparision，强调视频之间的 *diff*，并引导模型学习评估的 key hint

        $$
            \text{Video} \rightarrow Model \rightarrow \text{Relative Score Space}
        $$

    - 通过 GART 将传统的分数回归预测划分为两个子问题：

        > group-aware regression tree

        1. coarse-to-fine classification: 判断 “好坏” 的二分类

            将 relative score 空间划分成多个相离区间，随后通过二叉树将 relavtive score 划分到对应的 group

        2. small interval regression

            <u>在 relative score 被预测的 class 内</u> 预测 final score


    - 提出 Relative L2-distance，在模型评估中考虑组间差异

### 2-2 Relative Works

- Gordan：使用骨架轨迹评估体操跳跃动作质量

- Pirsiavash：

    - 离散余弦变换（DCT）被用于从 body pose 图像到 input feature 的编码

    - SVR 被用于从 feature 到 final score 的映射

- Parmar：使用 C3D 对视频数据进行编码以获得 spatio-temporal features

- Xu：使用两个 LSTM 来学习 multi-scale feature

- Pan：

    - 使用 spatial & temporal relation graphs，在其相交处建模（？）

    - 使用 I3D 提取 spatio-temporal features

- Tang：提出 USDL 来降低人类裁判打分造成的歧义

    > uncertainty-aware score distribution learning

### 2-3 Approach

![](./assets/CoRe%20Pipline.png)

#### 1 Contrastive Regression

- Problem Formulation

    - 大多数方法将 AQA 视为一个回归问题：输入包含目标动作的 video，输出对应的 quality score

    - 部分方法提出了 difficulty-score（已知），最终结果变为 qulity * difficulty

    !!! bug "由于视频往往在相似的环境下拍摄，模型很难从微小的变化中学习巨大的分数差异"

    ---

    - 重新定义问题为：
    
        Regress relative score between the Input Video $v_m=\{F_m^i\}_{i=1}^{L_m}$ & an Exemplar $v_n=\{F_n^i\}_{i=1}^{L_n}$ with Score Label $s_n$
    
    - 我们可以将回归问题写为如下形式：

        $$
        \hat{s}_m = R_{\Theta}(F_W(v_m), F_W(v_n)) + s_n
        $$

        <center>其中 $R_{\Theta}$ 为回归模型，$F_W$ 为特征提取模型（参数相同的I3D）</center>

        > 计算 CurInput & Exampler 之间的偏差值，然后基于 Exampler 的分数进行生成评分

- Exemplar-Based Score Regression

    !!! question "如何挑选具有可比性的 Exampler"

    1. 使用 I3D 模型对 input & exampler 进行特征提取，得到 $f_m, f_n$

    2. 将两者特征与 exampler score 进行聚合：$f_{(n,m)} = concat([f_n, f_m, \frac{s_n}{\epsilon}])$

        $\epsilon$ 是一个 norm constant，用于确保 $\frac{s_n}{\epsilon} \in [0,1]$

    3. 使用回归模型对 $pair(n,m)$ 的 score-diff 进行预测：$\Delta s = R_{\Theta}(f_{(n,m)})$

#### 2 Group-Aware Regression Tree

- contrastive regression 存在的问题

    比较回归可以预测 relative score $\Delta s$，但通常具有较大的值域

    => 直接预测 $\Delta s$ 具有一定困难

- GART 实现分治

    1. 将 $\Delta s$ 的值域划分为 2<sup>d</sup> 个 non-overlapping class

    2. 建立 $d-1$ 层的 二叉回归树（有 2<sup>d</sup> 个 leaf nodes）

    3. 分类

        整个过程遵循 coarse-to-fine manner：第一层决定 input 比 exampler 好/坏，并在后续具体量化 better/worse 的程度

- Tree Architecture

    - Input：
    
        - Root Node: 将聚合特征 $f_{(n,m)}$ 放进多层神经网络 MLP，并将 $\text{MLP}(f_{(n,m)})$ 作为 GART 的输入

        - Other Nodes: 父节点的输出

    - Output:

        - Internal Nodes: 产生通向左右子节点的概率

            通向某个具体 leaf node 的概率 = $\prod p_{\text{nodes in path}}$

        - Leaf Nodes: $sigmoid(P_{\text{leaf}}) \in [0,1]$ 代表了 Input 和对应 class 的 score-diff

- 边界划分策略

    > 等长划分 class 会导致 unbalance

    1. 收集测试集中所有 pair 的 score-diff： $\delta = [\delta_1, ... ,\delta_T]$，对其进行升序排列后得到 $\delta^*$

    2. 给定 n_class = $R$，遵循以下策略得到各组边界 $G^r = (min^r, max^r)$:

        $$
        \begin{align*}
            min^r &= \delta^*\left[\lfloor(T-1) \times \frac{r-1}{R}\rfloor\right] \\
            max^r &= \delta^*\left[\lfloor(T-1) \times \frac{r}{R}\rfloor\right]
        \end{align*}
        $$

        <center>其中，$\delta^*[i]$ 表示数组中的第 i 个元素</center>

- 优化策略

    > GART 包含 classification & in-class regression，所以总的 Cost 要拆成两部分。

    当 pair $\delta$ 的 score-diff 被分类至 Class i 时:

    - 将 One-Hot 分类标签的对应位置置 1: $l[i] = 1$ 

    - 另 regression 标签 $\sigma_i = \frac{\delta - min^i}{max^i - min^i}$

    此时，每个 pair 都拥有 One-Hot Classification Label $l$ & Regression Label $\sigma$:

    $$
    \begin{align*}
    J_{cls} &= - \sum_{r=1}^{R}(l_r \log(P_r) + (1 - l_r)\log(1-P_r))\\
    J_{reg} &= \sum_{r=0}^R(\hat{\sigma}_r - \sigma_r)^2, \text{where } l_r=1\\
    J &= J_{cls} + J_{reg}
    \end{align*}
    $$

    <center>其中 $P_r, \hat{\sigma}_r$ 分别是 Leaf Probability & Regression Result</center>

- 前向推导（仅 GART 部分）

    $$
    R_{\Theta}(f_{(n,m)}) = \hat{\sigma}_{r^*} (max^{r^*}-min^{r^*}) + min^{r^*}
    $$

    <center>$r^*$ 是具有 Max Probability 的 class 编号</center>

    !!! info "Multi-Exampler Voting Strategy"
        对于给定输入 $v_{\text{test}}$，选定 $M$ 个 exampler $\{v^m_{\text{train}}\}_{m=1}^M$ 及其对应分数标签 $\{s^m_{\text{train}}\}_{m=1}^M$

        Voting 策略可以使用如下公式进行描述：

        $$
            \begin{align*}
                \hat{s}^m_{\text{test}} &= R_{\Theta}(F_W(v_{\text{test}}, v_{\text{train}}^m)) + s_{\text{train}}^m \\
                \hat{s}_{\text{test}} &= \frac{1}{M} \sum \hat{s}_{\text{test}}^m
            \end{align*}
        $$

### 2-4 Evaluation Protocol

1. Spearman's Rank Correlation 

    为了能和之前的工作进行比较，此处采用了 Spearman's Rank Correlation 作为评估标准：

    $$
    \rho = \frac{\sum(p_i-p)(q_i-q)}{\sqrt{\sum(p_i-\overline{p})^2 \sum(q_i-\overline{q})^2}}
    $$

    其中 $p,q$ 分别表示样本在两个序列中的 ranking

2. Fisher’s z-value

    用于评估 across-action 的平均表现

3. Relative L2-Distance

    对于特定动作的 max/min score，R-l<sub>2</sub> 遵循如下定义：

    $$
    R-l_2(\theta) = \frac{1}{K} \sum\left(\frac{|s_k - \hat{s}_k|}{s_{max} - s_{min}}\right)^2
    $$

    其中 $s_k,\hat{s}_k$ 分别表示第 k 个样本的 ground-truth 和 预测得分

    > 传统的 l<sub>2</sub> 距离对于从属于不同 class 的 action 是无意义的；
    >
    > 相比于 Spearman's Rank Correlation 注重 RANK，R-l<sub>2</sub> 更加重视具体数值

## 3 PECoP

### 3-1 Abstract

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

### 3-2 Relative Works

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

### 3-3 Approach

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

- 使用了 Video Segment Pace Prediction (VSPP) 进行了预处理，从而对 “以不同速度完成的动作” 进行对比

#### Supervised fine-tuning

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


## 4 Video-based AQA (综述)

### 4-1 Definition & Challenges

- 相关领域：Human Action Recognition & Analysis

    - 现有的技术支持 action classification of short/long-term videos, temporal action segmentation and spatial-temporal action
location
    
    - 在 video sruveillance, vedio retrieval & human-computer interaction 领域有着广泛应用

    !!! bug "只能对动作进行 粗粒度的分类/定位，并不能对特定动作的质量进行客观评价"
    
    - 强调识别、捕捉不同种类动作之间的 external differences

    > AQA 则聚焦于同一种类动作间的 internal differences

- AQA 问题目标

    得到一个能 *自动做出客观评价的智能系统*，从而减少在动作评估中投入的人力物力、并降低主观影响。

#### Definition & Form

- Video-based AQA 是一个基于视频数据生成对特定动作质量客观评价的 internal differences 任务

- 用于 AQA 和 HAR 任务的 **模型** 具有一定程度的相似性：

    首先进行 feature extraction，随后通过 network head 实现复杂任务

    - 传统方法会采取 <u>DFT/DCT/linear combination</u> 实现 feature aggregation

    - 深度学习方法的发展则使得通过 <u>深层卷积网络(DCN)/RNN</u> 进行 video embedding 成为可能

- AQA 任务大致可被划分为以下三种：

    1. Regression Scoring：常见于运动领域
    
        - 一般直接使用 <u>SVR/FCN</u> 直接进行预测
        
        - 以 MSE 作为优化目标

    2. Grading：常见于对手术操作的评分

        - 实际上是个分类任务，输出的是诸如 `novice`, `medium`, `expert` 的标签

        - 一般使用 classification accuracy 进行评估

    3. Pairwise Sorting

        - 从测试集里面随手抓两个（一对）视频进行拉踩

        - 使用 pairwise sorting accuracy 进行评估

#### Challenges

- Area Specifed

    - Medical Care：由于医疗动作具有较高的时间复杂度、语言信息量和低容忍度，医疗相关的 AQA 解决方案需要有较强的语义理解能力

    - Sports：body distortion & motion blur

- Common Challenges

    计算效率、视角遮挡、模型可解释性  etc.

### 4-2 Datasets & Evaluation

<center>Datasets for AQA</center>

<table>
<tr>
    <th>类型</th>
    <th>名称</th>
    <th>Desc.</th>
</tr>
<tr>
    <td rowspan="8">Sport</td>
    <td>MIT-Diving</td>
    <td>60fps, 平均每个视频包含 150 帧，分数范围为 [20,100]</td>
</tr>
<tr>
    <td>MIT-Skiing</td>
    <td>24fps, 平均每个视频包含 4200 帧，分数范围为 [0,100]</td>
</tr>
<tr>
    <td>UNLV Dive & UNLV Vault</td>
    <td>平均每个视频包含 75 帧，分数范围为 [0,20]</td>
</tr>
<tr>
    <td>Basketball Performance Assessment Dataset</td>
    <td>24 Train + 24 Test，有 250 + 250 个 pair label</td>
</tr>
<tr>
    <td>AQA-7</td>
    <td>包含7种运动的视频，803 train + 303 test</td>
</tr>
<tr>
    <td>MTL-AQA</td>
    <td>16种不同类别的跳水视频，每个由7位裁判打分</td>
</tr>
<tr>
    <td>FisV-5</td>
    <td>平均时长为 2min50s，由9位裁判评判 TES & PCS</td>
</tr>
<tr>
    <td>Fall Recognition in Figure Skating</td>
    <td>276 顺利落冰 + 141 摔倒</td>
</tr>
<tr>
    <td>Medical Care</td>
    <td>包含缝合、穿针、打结三部分，有部分分和总评分</td>
</tr>
<tr>
    <td rowspan="3">Others</td>
    <td>Epic skills 2018</td>
    <td>包含揉面团、绘画、使用筷子三个子集</td>
</tr>
<tr>
    <td>BEST</td>
    <td>平均时长为 188s，包含5种不同日常活动的视频</td>
</tr>
<tr>
    <td>Infinite grasp dataset</td>
    <td>包含了94个婴儿抓东西的视频，时长在 80-500s，有 pair label</td>
</tr>
</table>

<center>Performance Metrics</center>

<center>
<table>
<tr>
    <th>Task</th>
    <th>Metric</th>
</tr>
<tr>
    <td>Regression Scoring</td>
    <td>均方误差 MSE</td>
</tr>
<tr>
    <td>Grading</td>
    <td>Classification Accuracy</td>
</tr>
<tr>
    <td>Pairwise Sorting</td>
    <td>Spearman Correlation Coefficient</td>
</tr>
</table>
</center>

!!! info "Spearman Correlation Coefficient $\rho$"

    $$
    \rho = \frac{\sum_i(p_i - \overline{p})(q_i - \overline{q})}{\sqrt{\sum_i(p_i - \overline{p})^2 \sum_i(q_i - \overline{q})^2}}
    $$

### 4-3 Models (截至 2021)

- Medical Skill Evaluation

    由于对医疗动作评估问题的研究早于深度学习方法的发展，大多数医疗相关的模型都使用了 <u>traditional feature</u>


- Sport AQA

    由于起步相对较晚，体育相关的研究使用 CNN 和 RNN 实现了较好的成果

    1. based on Deep Learning

        - 通常使用 2D-CNN/3D-CNN/LSTM 来进行 feature extract & aggregate
        
        - 通过 Network Head 来适配不同类型的任务

        根据关注点不同可划分为: Structure Design / Loss Desgin 两类

    2. based on Handcrafted Features (before 2014)

- Medical Care

    由于较强的专业性，医疗领域没有一个可以作为统一 benchmark 的数据集

    - GIT 聚焦于 OSATS 系统下的外壳手术技能评估

    - JHU 聚焦于 机器人微创手术(RMIS)

    - ASU 聚焦于 腹腔镜手术(laparoscopic surgery)

### 4-4 发展前景

- Dataset

    目前存在的 AQA 数据集规模较小，且包含的语义信息较少

    => 希望在未来推出更大规模、包含更多语义信息的数据集

- Model: more Efficient & Accurate
    
    - 更好的利用 temporal info 对动作进行建模

    - 使用 Unsupervised 方法减少数据标注、降低主观影响

    - 对 复杂、长期 动作的质量进行评估