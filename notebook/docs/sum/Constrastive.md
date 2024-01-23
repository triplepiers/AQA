## 1 CoRe

### 1-1 Abstract

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

### 1-2 Relative Works

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

### 1-3 Approach

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

### 1-4 Evaluation Protocol

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

## 2 TPT

### 2-1 Abstract

- 先前的 SOTA 方法

    使用 ranking-based pairwise comparison，或 regression-based methods 

    => 通过对 backbone 输出做 global Pooling，基于 *整个视频 (holistic video)* 进行 regression / ranking

    !!! bug "限制了对 细粒度·类内差异 (fine-grained intra-class variation) 的捕捉"

- 创新点

    regression-based，可以在避免 part-level 监督的情况下学习更细粒度的特征

    1. 使用 Temporal Parsing Transfomer 将 "整体特征(holistic)" 拆解为 temporal part-level 形式

        !!! example "将 Diving 拆解为 approach -> take off -> flight 等多个阶段"

        - 使用一系列 queries 来表示 <u>特定动作的 atomic temporal patterns</u>

        - 在 Decoding 阶段，将 frames 转换为 长度固定的·按照时序排列的 part representations

        - 基于 (relative pairwise) part representations 和 对比回归 得到最终的 Quality Score

    2. 提出了两种新的 Loss Function 用于 Decoder

        > 因为当前的数据集都没有提供 temporal part-level labels / partitions

        1. Ranking Loss: Cross Attenion 阶段学习的参数符合时序

        2. Sparsity Loss: 让学习的 part representation 更具 "判别性(discriminative)"

!!! info "Temporal Action Parsing: 细粒度动作识别"
    - Zhang: Temporal Query Network adopted query-response functionality
    - Dian: TransParser

    上述工作均聚焦于 "frame-level" 的特征增强，而本文则侧重于提取 "更具备语义信息的·part representation"

### 2-2 Approach

![](./assets/TPT%20Pipeline.png)

#### Overview

1. 使用滑动窗口将 input 划分为 $T \times M \text{frames}$ 的 <u>重叠 Clips</u>

2. 对于分割得到的 Clips，使用 I3D (backbone) 处理得到 $V = \{v_t \in \mathbb{R}^D\}_{t=1}^T$

    - $D$ 为 feature dimension

    - 每个 $v_t$ 经由空间上的 Average Pooling 得到（本模型并不关注 spatial patterns）

3. Contrastive Regression

    1. <u>Clip-Level</u> $V$ $\rightarrow$ <u>Temporal Part-Level</u> representations $P =\{p_k\in \mathbb{R}^d\}_{k=1}^K$

        > $d$ 为 Part Feature-dimension, $K$ 为 query 总数
        > 
        > 操作分别对 Input & Exampler 进行，生成 $P, P_0$

    2. 基于 Part-aware Contrastive Regressor 计算 Part-wise Relative representations
    3. Fuse Part-aware representation 以预测 Relative Score $\Delta s = R(P,P_0)$
    4. Final Score = $s + \Delta s$（$s$ 为 exampler 的实际分数）

#### Temporal Parsing Transformer
> 输入：Clip-Level Representation $V$，输出：quries (Part Representation)

!!! info "区别于 DETR 的 Transformer 架构"
    1. 因为 Encoder 不能提高本方法的准确性，本方法 **仅包含 Decoder**：
        
        可能是因为：
        
        - clip-level self-attention smooths the temporal representations

        - 本方法 cannot decode part presentations without part labels

    2. 在 cross attention block 添加了参数 $temperature$ 以控制内积的放大（？）

    3. 没有把位置信息（clip id）embed 到输入数据中
    
        => query 用于表示 atomic patterns 而非作为时间锚点

对于 $i^{th}$ Decoder Layer，有：

- Decoder Part Feature $p^{(i)}_k \in \mathbb{R}^d$

- Learnable Atomic Patterns $q_k \in \mathbb{R}^d$

- embedded Clip Rrpresentation $v_t \in \mathbb{R}^d$

<center>先用 $p^{(i)}_k + q_k$ 得到 query，随后和 $v_t$ 做 cross attention 得到输出 $\alpha_{k,t}$</center>

$$
\alpha_{k,t} = \frac{
    \exp{((p_k^{(i)} + q_k)^T \cdot \frac{v_t}{\tau})}
}{
    \sum_{j=1}^T \exp{((p_k^{(i)} + q_k)^T \cdot \frac{v_j}{\tau})}
}
$$

- $\alpha_{k,t}$ 表示 query<sup>k</sup> 和 clip<sup>t</sup> 之间的 attention value
- $\tau \in \mathbb{R}$ 是可调参数，用于放大内积以使得 attention 更加 discriminative.

---

由于本模块的目的在于：将 Clip Repre 聚合到 Part Repre 中，我们需要根据如下策略对 Part Repre 进行更新：

$$
p^{(i)}_k += \sum_{j=1}^{T} \alpha_{k,j} v_j + p_k^{(i)}
$$

#### Part-Aware Contrastive Regression

> 在 Temporal Parsing Transformer 模块，我们已经成功将输入转化为 Part Repre $P = \{p_k\}$;
> 
> 此时我们需要对 Input 和 Exampler 的 $P,P_0$ 进行比较，并生成 Relative Score $\Delta s$

!!! tip "我们可以分别计算每个 Part 的相对分数，最后把他们 fuse 到一块儿"

对于 $k^{th}$ Part，我们通过 *多层感知机(MLP)* $f_r(.)$ 生成对应的 relative pairwise representation $r_k \in \mathbb{R}^d$：

> 所有的 Parts <u>共用一个 MLP</u>

$$
r_k = f_r(\text{concat}([P_k;P_k^0]))
$$

---

为了提高准确性，此处使用 Group-aware Regression strategy 生成 Relative Score $\Delta s$：

- 对 Training Set 中所有可能的 pair 生成 $B$ 个 $\Delta s$ 取值区间（类似 CoRe Tree 里的区间非偏区间划分策略）

- 生成 One-Hot Label $\{l_n\}$，表示 $\Delta s[i]$ 所处的区间编号

---

对 Input Video 的预测：

1. 对 Relative Part Repre $\{r_k\}$ 使用 Average Pooling
2. 使用 2 * 2-Layer MLP 对输入视频的 classification label $l$ & 回归结果 $\gamma$ 进行预测

### 2-3 Optimization

- 假设：每一类动作都可以按照相同顺序进行阶段切分，并通过 transformer query 进行表示

- 在 Cross Attention 阶段，$k^{th}$ query 的 attention center $\\overline{\alpha}_k$：

    $$
    \overline{\alpha}_k = \sum_{t=1}^T t \cdot \alpha_{k,t} \in [1,T]
    $$

    - $\{\alpha_{k,t}\}$ 是已经 normalized 的 attention responses
    - $k^{th}$ query 对于所有 clip 的 attention 总和为 1，即 $\sum_{t=1}^T \alpha_{k,t} = 1$

#### Cross Attention Block

1. Ranking Loss

    为了鼓励各 query 聚焦于 <u>different temporal region</u>，我们对 attention center 使用 Ranking Loss

    !!! tip "理想情况下，Part Repre 在（同类）不同视频下有 <u>相同时序</u>"

    $$
    L_{rank} = \sum_{k=1}^{K-1} \max{(0,\ \overline{\alpha}_k - \overline{\alpha}_{k+1} + m)} + \max{(0,\ 1-\overline{\alpha}_1 + m)} + \max{(0,\ \overline{\alpha}_k - T + m)}
    $$

    <center>$m$ 是用于控制惩罚力度的超参数</center>

    - 第一项用于确保顺序 $\overline{\alpha}_k \lt \overline{\alpha}_{k+1}$ 成立

    - 后两项分别用于确定首位顺序 $\overline{\alpha}_0 = 1$ 和 $\overline{\alpha}_{k+1} = T$ 成立
    （$\overline{\alpha}_k \in [1,T]$）

2. Sparsity Loss

    鼓励每一个 query 聚焦于靠近 center $\mu_k$ 的那些切片：

    $$
    L_{sparsity} = \sum_{k=1}^K\sum_{t=1}^T|t - \overline{\alpha}_k| \cdot \alpha_{k,t}
    $$

#### Contrastive Regressor

基于比较学习的回归器需要预测 分组标签 $l$ & 相对偏差 $\gamma$，我们：

- 对各组使用 BCE Loss

    $$
    L_{cls} = - \sum_{n=1}^N [l_n \log{(\vec{l}_n)} + (1-l_n) \log{(1 - \vec{l}_n)}]
    $$

- 对由 ground-truth 生成的 interval 信息使用 Square Error

    $$
    L_{reg} = \sum_{n=1}^N (\gamma_n \ \vec{\gamma}_n)^2, \text{ where } l_n=1
    $$


#### Overall Training Loss

$$
L = \lambda_{cls} L_{cls} + \lambda_{reg} L_{reg} + \lambda_{rank} \sum_{i=1}^L L^i_{rank} + \lambda_{sparsity} \sum_{i=1}^L L^i_{sparsity}
$$
