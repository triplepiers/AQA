## 2 IRIS

!!! info "本文聚焦的问题：(花滑)单人短节目"
    - 节目时长约为 3min

    - 相比于自由滑，节目编排受制于更多的规则

### 2-1 Abstract

- 创新点

    > XAI 开发者应该学习 <u>具体应用场景（运动）的评分标准(rubrics)</u>

    本文提出的 Interpretable Rubric- Informed Segmentation：

    - 其给出评分的过程是 **可解释** 的，依照 rubric 决定 what to consider

        使用 4 个 key feature 进行 预测&解释: Score, Sequence, Segments, and Subscores

    - 对输入进行分割，以定位对应特定 criteria 的特殊 section

    - 本文仅对 *figure skating* 进行测定，但方法同样适用于其他 Video-based AQA 问题

### 2-2 Relative Works

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


### 2-3 Approach

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

<center><img src="../assets/IRIS%20visual.png" style="max-width:500px;"></center>
<center>IRIS 评分过程可视化</center>

### 2-4 Evaluation

1. 计算 Final Score 的 Spearman Rank Correlation

2. (New) 分别计算 TES & PCS 的 Spearman Rank Correlation

3. Dice Coefficient: 计算 segmentation 的准确性

    对预测分割序列 $\hat{a}$ 和真实动作序列 $a$，计算其交叠程度 $2(a \cdot \hat{a})/(|a|^2 + |\hat{a}|^2)$
