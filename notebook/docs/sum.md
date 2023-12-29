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