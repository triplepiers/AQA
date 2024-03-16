## DataSets

- MIT-Skate

- Fis-V

    [S-LSTM + M-LSTM (with C3D)](https://arxiv.org/pdf/1802.02774.pdf) 同样将 TES 和 PCS 回归作为独立任务

    !!! warning "可解释性"
        S-LSTM 的 attention weight 可以表明不同 clip 对最终预测值的贡献程度

    - S-LSTM 用于局部信息建模:

        - 常见方案使用最大或平均池化运算符将 C3D 特征合并为视频级表示
        - 然而，并非所有视频片段/帧对回归最终分数的贡献相同
            
            use self-attentive embedding scheme to generate the video-level representations.

    - M-LSTM（多尺度卷积跳跃）采用了几个具有不同核大小的并行1D卷积层，学习以多个尺度建模顺序信息

        通过这种M-LSTM架构，我们实际上可以兼顾两全其美 —— 多尺度卷积结构可以从视频中提取局部和全局特征表示
        
        - 修订后的跳跃LSTM可以高效地跳过/丢弃学习局部和全局信息中不重要的冗余信息
        - 不同尺度的最终LSTM输出仍然被串联并通过非线性全连接层进行学习回归
        

- [RFSJ](https://dl.acm.org/doi/pdf/10.1145/3581783.3613774): a Replay Figure Skating Jumping dataset

    !!! info "Fis-V 把重播片段剪掉了，但是这里保留了"

- FR-FS: 判定是不是摔倒了

- [2023 FineFS](https://dl.acm.org/doi/pdf/10.1145/3581783.3613795): [Fine-grained Figure Skating dataset](https://github.com/yanliji/FineFS-dataset)

    - 细粒度的分数标签，从粗到细的技术子动作类别标签，技术子动作的开始和结束时间

        最粗粒度是 Spin, Jump, StepSeq（没有单独划分接续步，但也切成了7段）

        甚至给每个动作标了 BV & GOE（我哭死）

    -  Video swin transformer (VST) 提取的视频特征 => 对比 I3D 和 C3D 后选用

    - 使用 MHForemer 提取 的 17-joints 3D 骨架模型（并经过后续手动修正）

    ![](https://githubraw.com/yanliji/FineFS-dataset/main/imgs/DatasetFigureNew.png)

    ---

    [论文](LUSD-NET) 提出的 对 TES 和 PCS 分别进行了回归预测

    - 使用 ACM-Net 逐个确定 clip 中是否包含技术动作（仅利用视频级别标签来定位动作实例的时间边界并识别相应的动作类别）
    - Uncertainty Score Disentanglement (USD)：不确定性分数解缠。

        学习独立的面向PCS和面向TES的特征，并利用这些特征来预测PCS和TES。

    - 通过 Encoder + 线性回归器分别预测分值（参考了 uncertainty regression 思路）

    - benchmarks: FineFS & Fis-V 上 TES & PCS 的 Spearman corr.

## Short-Program 短节目

**短节目** 规定了动作细节内容：3种跳跃 + 3种旋转 + 1种步法 + 接续步

- Spin 主要就三种：蹲踞旋转（sit spin）、燕式旋转（camel spin）和直立旋转（upright spin）
- 定级步法可以分为：转体步法和滑行步法，比赛中裁判要**根据运动员的动作编排和临场发挥确定等级**。
- 编排步法包括三类：滑行步法、旋转步法和跳转步法。（但光是滑行步法就一堆）

| |男单 | 女单 |
|:--|:--|:--|
|时间 | 2min40s $\pm$ 10s| 2min40s $\pm$  10s|
|   Jump-1  |两周/三周半跳跃 |两周/三周半跳跃  |
|   Jump-2  | 包含步法的三周跳| 包含步法的三周跳|
|   Jump-3  | 3+2以上的组合跳 | 3+2以上的组合跳|
|Spin-1|跳接旋转|跳接旋转|
|Spin-2|一次换脚的燕式/蹲转|弓身旋转|
|Spin-3|一次换脚的组合旋转|一次换脚的组合旋转|
|步法|两套（10起为1套）|一套普通 + 一套螺旋（10起取消）|

## Grading Policy

9 名裁判对 TES & PCS 进行打分：

- 每个技术动作的评分 $score \in [-5, +5]$，该动作最后的 $GOE = (\sum^9_{i=1} score_i - MAX - MIN) * 0.1 * BV$

$$
Score = TES_{技术分}( BV_{基础分} + GOE_{执行分} ) + PCS_{节目分} - 扣分项
$$