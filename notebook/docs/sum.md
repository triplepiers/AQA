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

### 1-2 TSA-Net 流程

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


#### TSA Module 实现

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