## 2022: Skeleton-based Deep Pose Feature Learning

!!! warning "好像只是把 ST-GCN 叠了 10 层，再加一个 LSTM"

### 1 Abstract    

- 现有工作

    - Evaluate **single / sequential-defined** action in **short-term** videos

        > Sample: diving(跳水), vault(跳马)

    - Extract features **directly** from RGB videos through **3D-ConvNets**

        !!! bug "导致 feature 和 scene info 混淆在一起"
            - 但另一篇文章又批判了 skeleton-based 方法忽视了水花等环境因素，导致准确度下降
            - 大概得考虑一下 trade off？

- Long-duration Video 面临的挑战

    - Contain multiple chronologically(时序上) inconsistent actions

        e.g. 不同花滑短节目对于 滑行/跳跃/旋转 的编排顺序并不一致

    - Actions only have slight difference in a few frames

        e.g. 3Flip & 3Lutz 其实长得很像

- WHY Skeleton-based ？

    > Yes, but: 性能没有 RGB-Based 方法好

    - 不应该只提供一个 final score，还应该提供 meaningful feedbacks 帮助人们进行改进
  
    - Robust to changes in: appearance, lighting, surrounding env

- 本文工作：`deep pose feature learning` `long-duration videos (花滑/艺术体操)`

    - 特征提取：使用 **Spatial-Temporal Pose Extraction (STPE)** Module

        - Capture **subtle** changes
  
        - Obtain **skeletal data** in **space & time** dimensions
  
    - 时序特征表示：使用 **Inter-action Temporal Relation Extraction (ITRE)** Module

        通过 RNN 对骨架数据的时序特征进行建模

    - Score Regression：使用 FCN（全卷积网络）实现

    - Benchmarks: MIT-Skate, FIS-V

### 2 Relative Works

#### Skeleton-based AQA

在 Sports 领域的研究：

1. Pirsiavash 最早提出将 pose feature 应用于 AQA 领域

    通过 Discrete Cosine Transform (DCT) 对 pose feature 进行编码，随后使用 SVR 进行回归预测

2. Venkataraman 提出计算 multivariate approximate entropy (多变量近似熵)，对 单个关节的变化 & 关节间联系 进行建模

3. Nekoui 建立了 two-stream(双流) 网络，对 appearance & pose feature 分别进行建模

4. Pan 提出了 Graph-based Model 对 关节间协方差 & 身体局部动作 进行建模

---

与 GCN 相关的方法：（只在 diving 这样的短时任务中验证过）

> GCN 相比于其他卷积网络，在骨架数据这种 graph-structured 的特征上具有更好的泛化性

1. Bruce 提出了一个 two-task GCN 对 deep pose feature 进行提取

    应用于老年痴呆症的异常检测与质量评估

2. Nekoui 提出了数据集 ExPose，并使用了 ST-GCN 从提取的关节序列中提取 pose feature

### 3 Approach

??? info "Figure Skating Grading Rule"
    > 这里也标榜了一下 "based on the rules of Figure-Skating"，然后水了好长一段

    $$ \text{Final Score} = TES + PCS - TDS $$

    - 每个动作的 $TES = \text{basic score} + GOE$，$TDS \geq 0$ 是失误扣分
    - 一般有 9 位裁判，去掉 最高&最低 后取平均值

#### 1 骨架信息获取 & 预处理

记共有 $N$ 个视频的 Labled RGB 视频数据集 $V = \{v_i, l_i\}_{i= 1 \sim N}$

- 其中 i-th 具有 $m$ 帧的视频记为 $v_i = \{I_j\}_{j=1\sim m}$
- $l_i$ 是 i-th 视频的 ground-truth label

---

1. 对 i-th 视频进行 Pose Estimation 后，得到骨架数据 $v_i \rightarrow \{S_j\}_{j=1\sim m}$

    这篇文章使用了 OpenPose 提供的 18-joint Model

2. 对所有的 Skeleton Seq 采取相同的采样策略：只处理前 $T$ 帧，使 $s = \{S_j\}_{j=1\sim m} \rightarrow \{S_j\}_{j=1\sim T}$

3. 将 $s$ 划分为 $M$ 个不重叠的子序列 $s \rightarrow \{P_k\}_{k = 1 \sim M}$，每个子序列对应长度 $Z = \frac{T}{M}$

4. 考虑单个子序列 $P = \{p^i = \{(x_j^i, y_j^i, \text{ac}_j^i)\}_{j=1}^{18}\}_{i=1}^Z$

    其中： $(x_j, y_j)$ 是 j-th joint 在笛卡尔坐标系下的坐标，$\text{ac}_j$ 是该坐标的置信度

5. BatchNormalization： $x' = \frac{x - \mu}{\sigma}$

#### 2 时空姿态特征提取 (STPE)

<center><img src="../../assets/STPE.png" style="max-height: 200px;"></center>

熟悉的类 ST-GCN 思路：
> 好的，直接拿 ST-GCN 当 backbone 了

- Spatial Dimension: 使用 skeleton-graph 来表示关节及其连接关系
- Temporal Dimension:  把相邻 frame 里的同一个 joint 连起来就算完事

---

小小改进：

- Basic Block = SpatialConv layer $\rightarrow$ TemporalConv Layer $\rightarrow$ Dropout Layer
  
    SConv & TConv 的输出都有 BatchNorm + ReLU 的处理

- 堆叠 10 个 Basic Block：

    - 使用 $C$ 表示 feature Channel，$Z$ 为子序列时长，$J = 18$ 为关节总数
    - 令 temporal kernel size = 9 && $L_4, L_7$ strides = 2，则 `4-3-3` 层的输出通道数分别为 `64-128-256`

    $$
    \text{input}^{C \times Z \times J} \rightarrow \text{STPE} \rightarrow \text{output}^{256 \times Z' \times J} = f_p
    $$

#### 3 动作间时序联系提取 (ATRE)
> - 对于花滑来说 action 之间的衔接会影响 PCS 得分
> - 这里通过使用了 LSTM 的 RNN 实现

经过 STPE 模块的处理，我们得到了 pose feature $F_p = \{f_p^k\}_{k=1\sim m}$

1. 使用 全连接(FC) 层来

    - Remove redundant information
    - Reduce dimension of Pose Feature

2. 使用 BatchNorm 层提升泛化能力

---
> 关于 LSTM 的实现

使用一个 $M$ time steps 的 LSTM 网络（因为 segmentation 的数量是固定的）

- 共有 $M$ 个 Memory-cells 用于存储 info * output feature
  
- 每个 cell：
    - 输入 = i-th skeleton subSeq + 上一个 cell 的 output
    - layer 数量为 1，hidden size of layer = 256

- 最终输出 $f_t$（两种方案）：

    1. ✅ 最后一个 LSTM cell 的输出
    2. 所有 LSTM cells 输出的平均值

#### 4 回归预测

- 使用 3 Layers FCN(全连接神经网络) 进行特征降纬

    由于 STPE 的输出为 $C' \times Z' \times J$，3 Layers 的节点数分别为 `[C'*Z'*J, 2048, 1]`

- 使用 2 Layers FCN 进行回归预测，节点数分别为 `[256, 1]`

整个过程可被描述为：

$$
\hat{l} = \text{Activation}(\text{FC}(f_s)), f_s \in \{f_p, f_t\}
$$

!!! tip "Finding"
    由于两个 Benchmark 之间的分数分布不太一样，作者发现使用 `original data + ReLU()` 的效果 > `norm(data) + sigmoid()`

- Loss Function

    不同于其他方法复杂的损失函数，这里只用预测值与实际值之间的 MSE 误差

    $$
    L_{MSE} = \frac{1}{N}\sum_{i=1}^N(l_i - \hat{l}_i)^2
    $$