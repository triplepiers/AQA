# 狠狠切割

## 2018 Stacking Segmental P3D

!!! info "首个 Stage by Stage 评分的方法"

### 1 Abstract

- 先前的工作

    - C3D (Convolutional 3D Network)
  
        - 优势：可以同时捕捉 appearance & subtle motion cues
        - 缺点：训练消耗会消耗较多内存，只能处理等帧切割的 clip

    - 更多工作将 C3D 分解为 “两步走”：
  
        1. 在图像上训练的 2D-CNN（ResNet）
        2. 基于 2D feature 生成时序特征的 1D-CNN（LSTM）

        > 但除了 Two-stream 外的方法无法捕捉 motion cue

    - 信号处理领域的研究表明，一个 filter 往往可以由若干更简单的 filter 相乘得到

        P3D 和 I3D 工作已经证明在大规模数据集(Kinetics)上训练的3D卷积可被隐式分解

        => 但我们很难解释具体是哪一个部分起到了关键作用


- 本文工作

    - 基于 ED-TCN，提出了 Segment-based P3D-fused Network (S3D)

        对于每个 segment 分别应用 P3D，再进行 aggregation

    - 证明 segment-aware 的训练方式强于 full-video 训练

        发现 full-video P3D 和只观察 “水花” 的结果类似（但没有关注入水前的动作序列）
        
    - 证明 temporal segmentation 可以以较小的代价完成
    - benchmark: UNLV-Dive

### 2 Related Works

#### Diving Skill Assessment

- 早期有使用 approximate entropy features 的工作
- Pose + SVR on MIT-Diving
- C3D + SVR on UNLV-Dive

#### 3D-CNN & STN

- 3D-CNN

    - 3D-CNN 并不是特别为了视频分析任务设计的（patch-level 3D-CNN 是医学图像分析的重要手段）

        事实上，3D-CNN 是 STN 的替代品（毕竟时间和空间存在本质上的不同）

    - 在预设帧数视频上训练的 3D-CNN 要求输入也有同样的帧数（每个 clip 都得是 16-frame）

        小的 clipSize 一是出于内存占用的考虑，同时也保证了特征的 locality

    - 和捕获 spatial local connectivity 的 2D-CNN 相比，3D-CNN 天然支持对 temporally local motion and coherence 的捕捉

- TCN: 在 segmenting fine-grained actions 上具备 SOTA 表现

    - TCN 实际上是 2D-CNN + LSTM 的简化版本 —— 在 2D-CNN 的倒数第二层上构建 cross-frame 的 1D-CNN
    
        其输入是由 2D-CNN 逐帧提取的一维特征

    - Encoder-Decoder TCN (ED-TCN) 通过 下采样-上采样 实现 Encode-Decode，随后通过 `softmax` 为每一帧预测动作类型标签

### 3 Approach

!!! info "Diving Video Segmentation"
    作者认为 Diving 视频可以划分为 5 个片段：

    > 实际需要评分的只有 2-5 (4个片段)

    1. Preparation（不评分）
    2. Jumping: 离开跳台 -> 手第一次碰到身体
    3. Dropping: 手第一次碰到身体 -> 手最后一次离开身体
    4. Entering into the water
    5. Water-spray decaying: 水花逐渐消失


#### 1 full-video P3D

- 改装 P3D

    - 原装的 P3D 适用于 action classification 任务，但同时考虑了 appearance & local motion

    - 为了完成适配 regression 任务，作者把最后一层换成了 Fully-Connected + Dropout

- 拉踩 C3D

    - C3D 的 `3 * 3 * 3` kernel size 是硬性要求，训练成本大
    - P3D 的 `3 * 3 * 3` kernel 是 **可拆分的**，可以看成 `3*3*1 + 1*1*3` 的简单 kernel 组合

        这同时允许网络独立进行 2D-Conv 和 1D-Conv

    - 与 ResNet 的思想类似，P3D 允许将这些卷积层做成 residual units，根据 2D($S(·)$)-1D($T(·)$) 的混合策略不同，可以讲单元划分为三种：
  
        1. P3D-serial: $x_{t+1} = x_t + T(S(x_t))$
        2. P3D-parallel: $x_{t+1} = x_t + T(x_t) + S(x_t)$
        3. P3D-composition: $x_{t+1} = x_t + T(S(x_t)) + S(x_t)$

        其中 $x_t, x_{t+1}$ 分别是 x-th 单元的输入输出

    - P3D 还采用了 bottleneck 的设计：在每个 residual units 前后加上 `1*1*1 Conv`，分别用于 降低输入纬度 & 增长输出纬度
  
#### 2 Segment-level P3D

!!! question "预训练的 P3D 模型只能处理 16-frame 输入，我们应该怎么对 segment 提取特征呢？"
    只要在每个 segment center-frame 的左右 16-frame 提取特征就好啦

- 我们能够 independently 的对每一个 stage 进行特征提取/分数预测：

    - use features: 取 $Avg$ => single feature

    - use subscores: 合并所有 subscores => feature vec `[s1, s2, ..., sn]`

- 通过上面的方法，我们可以获得 segmentation+P3D 处理后的 feature set $x$

    加上整个视频的 ground-truth label $y$，我们可以得到一个 training-sample $(x,y)$

- 最终训练一个 LR(逻辑回归) / SVR 模型进行分数的回归预测

#### 3 Teomporal Segmentation using TCN

- 此处的 Temporal Segmentation 其实可以视为一个 *逐帧五分类* 任务

    同时需要保证 intra-class continuity（每个阶段是连续的）

- 输入：frame-level 2D-CNN feature

    - 假设输入视频共有 $K$ 帧，2D-CNN 的输出纬度为 $D$，我们可以将 2D-CNN 提取的特征表示为 $X_0 \in \mathbb{R}^{D \times K}$

    - 此时每一层 TCN 的输入可以被记为 $X_i(i \geq 0)$

- 输出：5 segments

- ED-TCN: 我们可以讲 l-th Layer 的 temporalConv 过程记为以下形式

    $$
    X_l = \text{Activate}(W_l * X_{l-1} + b)\\
    $$

    $$
    X_l \in \mathbb{R}^{N_l \times T_l},\ W_l \in \mathbb{R}^{d_l \times N_{l-1}},\ b \in \mathbb{R}^{N_l}
    $$

    - 初始条件为 $(N_0, T_0) = (D,K)$
    - $N_l$ 是 l-th Layer 中的 `n_Conv_Filters`，$T_l$ 是 `len(feature)`，$d_l$ 是 l-th Layer 的 `len(filter)`




## 2023 IRIS

!!! info "本文聚焦的问题：(花滑)单人短节目"
    - 节目时长约为 3min

    - 相比于自由滑，节目编排受制于更多的规则

### 1 Abstract

- 创新点

    > XAI 开发者应该学习 <u>具体应用场景（运动）的评分标准(rubrics)</u>

    本文提出的 Interpretable Rubric- Informed Segmentation：

    - 其给出评分的过程是 **可解释** 的，依照 rubric 决定 what to consider

        使用 4 个 key feature 进行 预测&解释: Score, Sequence, Segments, and Subscores

    - 对输入进行分割，以定位对应特定 criteria 的特殊 section

    - 本文仅对 *figure skating* 进行测定，但方法同样适用于其他 Video-based AQA 问题

### 2 Relative Works

1. Action Quality Assessment

    目前已经有许多针对 Sport 及其他领域的 AQA AI 解决方案被提出，通过训练一个 Regression Model 对最终分数进行预测。
    
    - 这些方法往往通过 3D CNN (C3D / I3D) 进行特征提取 —— 通过在 2D(spatial) & 1D(temporal) 进行卷积，将**较短的**视频切片转换为 feature vector

    - 为了处理**更长的**输入视频：

        - Parmar 将 3DCNN 和 LSTM 结合，提出了 C3D-LSTM

        - Zheng 基于 Graph Convolutional Network 提出了 Context-aware 的模型
        
            对 Static Posture & Dynamic Movement 进行建模来捕捉不同时间跨度上的联系

        - Nekoui 提出了一种 CNN-based 的方法同时对粗细粒度的 temporal dependencies 进行捕捉
        
            使用 video feature & pose estimation heatmap，并堆叠不同 kernel size 的 CNN 模块来捕捉不同时间跨度的 pattern 进行捕捉

        - Xu 针对 *figure skating* 场景提出了 multi-scale and skip-connected CNN-LSTM
        
            通过不同大小的 CNN kernel 对短期 temporal dependencies 进行捕捉 & 通过 LSTM + self-attention 对长期 temporal dependencies 进行捕捉

2. Explaining Action Quality Assessment

    虽然上述方法卷赢了准确性（预测结果和裁判评分之间的 Spearman’s rank correlation），但却忽略了用户应该如何应用和解释这些预测结果

    > 老师 chua 的一下给你的卷子批了个总评分，又不告诉你扣哪了

    - Yu 通过 Grad-CAM saliency maps 对 *diving* 问题的评分过程进行解释
    
        => 但 saliency maps 的本意是拿来 debug 的 orz

    - Pirsiavash 通过计算 由姿态估计得到的 relative score 的梯度 来对 *diving* 问题的评分过程进行解释

        => 基于数据（而非基于人类裁判如何做出裁决）

    - 使用数据传感器 + 浅层模型的方法

        - Khan 借鉴了电子板球游戏，使用可穿戴设备收集数据并对板球击球数据进行分析，并人为规定了若干 low-level sub-actions

        - Thompson 在量化评分标准后，提出了对 “盛装舞步” 项目的可视化方法

3. Rubrics for Figure Skating

    - “评分标准” 的两要素：

        1. $\geq 1$ * trait / dimension + 对应的解释案例

        2. 各 dimension 的 评分范围 / 分段标准

    - ISU Judging System: 对每个维度给 sub-score，最后折算总分

        1. TES（技术得分）

            运动员的技术动作序列将被提前列出，每个动作的得分 = Base Value(难度分) + GOE(偏差值)

            > $GOE \in [-5,5]$

            - 动作序列将由：Jump, Spin, Step Seq 及之间的过渡动作组成

            - Jump 可被分为六种：Toe Loop (T), Salchow (S), Loop (Lo), Flip (F), Lutz (Lz), and Axel (A)

                根据起跳、落冰动作进行区分，不同的圈数会给不同的分

            - Rotation 可被分为三种：Upright (USp), Sit (SSp), and Camel (CSp)

                根据难度、流畅度、稳定度进行评分

            - Step Sequence 可被分为三种：Straight line step sequence (SISt), Circular step sequence (CiSt), and Serpentine step sequence (SeSt)

                步伐必须与音乐相匹配，并且干净利落的完成shuo ta

        2. PCS (Program Component Score) 由五部分组成：

            - Skating Skills: 对节目内容的丰富程度、技术是否干净、滑速进行评分
            
            - Tran- sition / Linking Footwork: 对整体步法和姿态转变流畅度进行评分
            
            - Performance / Execution: 是否从肢体动作和情绪上传达了配乐的情感

            - Choreography / Composition: 节目动作编排是否与音乐契合
            
            - Interpretation: 用于表达音乐的动作是否具有创新性


### 3 Approach

<center><img src="../../assets/IRIS%20Pipeline.png" style="max-width:500px;"></center>

??? info "事前可解释性建模 Ante-hoc"
    > IRIS 在对模型进行训练前，将可解释性结合到模型结构中，以实现模型内置的可解释性

    - It is informed by a rubric to determine what to consider when calculating its judgement.

    - It performs seg- mentation to specify moments of a performance to judge specific criteria.

IRIS 使用了 4 个 Rubric-Feature 以预测和解释其作出的判断：

1. Score: Final Judgement

2. Sequence: 描述了用于评判的动作 (technical elements)

3. Segments: 每个动作由哪一条规则评判

4. Subscores: 描述 TES & PCS 各项得分

#### 0 Data Preparation
> 使用 MIT-Skate 数据集

- 将 3min 左右的视频转化为 4D tensor: `[x, y, time, colorChannel]`

- 提取 ISU 发布的 PDF 文件中的 skater name, TES base, GOE, total subscores, PCS subscore。将其与视频数据相对应。

- <span style="color:red;">手动</span> 标记每个动作的起止时间 （train segmentation）

- 为了确保每个类别下的都有充足的数据，这里队 Jump, Spin, Step Sequence, Transition 的划分较为宽泛

    > 建议 Future Work 能够用更多数据来划分 more specific labels

#### 1 Base Embedding
> 生成 video 的 vector presentation

训练 3D-CNN（I3D） 模型 $M_0$：

- 输入: video tensor $x$

- 输出: timeseries embedding $\hat{z}_t$: 2D tensor

    $M_0$ 会对每 0.534s 生成一个 vector embedding，每个视频将对应约 356 个（zero-padding）

#### 2 Action Segments

- 训练 multi-stage temporal convolutional network (MS-TCN) $M_t$：

    - 训练集：人工标注的 segments 作为 ground truth (监督学习)

    - 输入：$\hat{z}_t$

    - 输出：action sequence embedding $\hat{z}_{\tau}$ (Seq-to-Seq)
    
        标注了每个 action 的起止时间 (zero-padding)

- 由于 TCN 可能预测出 over-segmentation，本文使用了以下方法进行优化

    1. 使用 truncated mean squared err 对 loss function 进行平滑操作

        > 合并较小的 segmentation

        $$
        L_{\mu} = \frac{1}{T} \sum_t^T \max(\varepsilon_t, \varepsilon)
        $$

        - $\varepsilon_t = (\log\hat{m}(t)- \log\hat{m}(t-1))^2$

        - $\varepsilon$: truncation 超参数

    2. 使用启发式的方法

        1. 从裁判表中数处每个动作类别 $a$ 包含的动作数量，总量为 $n_a$ 个

        2. 使用 <u>最长的</u> $n_a$ 个 segmentation，将其分配给各类别

        3. 剩下的较短 segmentations 全都打上 `transition` 标签

#### 3 Predict Subscores

IRIS 预测的 subscores 分为以下两个部分：

1. TES (7): 对每个技术得分点进行评估

    1. 对 action sequence 中的每一个动作 $\tau$ 分别预测 TES（仅 GOE 部分）

        为每个 seq step $\tau$ 各训练一个传统 CNN 模型 $M_{\Delta \tau}$:

        - 输入：seq embedding $\hat{z}_{\tau}$ & 真实动作标签 $a_{\tau}$(可以预先得知)

        - 输出：单个动作的 GOE 得分 $\hat{y}_{\Delta \tau}$


    2. 计算单个动作的 TES 得分 $\hat{y}_{\tau} = \hat{y}_{\tau_0} + \hat{y}_{\Delta \tau}$ (Base + GOE)

    3. 计算所有动作的总 TES 得分 $\hat{y}_{total} = \sum_r \hat{y}_r$

2. PCS (5): 对节目 *整体表现* 进行评估

    训练 multi-task CNN 模型 $M_{\pi}$：

    > 因为各项总评之间存在 correlation，训练 multi-task 比训练 multi-indipendent 更准确

    - 输入： **整个视频** 的 time series embedding $\hat{z}_t$

    - 输出：预测 PCS 的各分量 $\hat{y}_p^i$ (multi-task)，将其和 $\hat{y}_{\pi}$ 作为总 PCS

#### 4 Predict Final Score

简单的将 TES & PCS 总分相加即可：

$$
\hat{y} = \hat{y}_{total} + \hat{y}_p
$$

<center><img src="../../assets/IRIS%20visual.png" style="max-width:500px;"></center>
<center>IRIS 评分过程可视化</center>

### 4 Evaluation

1. 计算 Final Score 的 Spearman Rank Correlation

2. (New) 分别计算 TES & PCS 的 Spearman Rank Correlation

3. Dice Coefficient: 计算 segmentation 的准确性

    对预测分割序列 $\hat{a}$ 和真实动作序列 $a$，计算其交叠程度 $2(a \cdot \hat{a})/(|a|^2 + |\hat{a}|^2)$
