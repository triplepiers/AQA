# 汇总

## 2021: A Survey of Video-based AQA

### 1 Definition & Challenges

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

### 2 Datasets & Evaluation

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

### 3 Models (截至 2021)

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

### 4 发展前景

- Dataset

    目前存在的 AQA 数据集规模较小，且包含的语义信息较少

    => 希望在未来推出更大规模、包含更多语义信息的数据集

- Model: more Efficient & Accurate
    
    - 更好的利用 temporal info 对动作进行建模

    - 使用 Unsupervised 方法减少数据标注、降低主观影响

    - 对 复杂、长期 动作的质量进行评估

## 2022: 视频理解中的动作质量评估方法综述

### 1 AQA 方法分类

1. 以质量分数为评估结果
2. 以等级类别为评估结果
3. 以质量等级排序为评估结果
   
### 2 数据集 & 评价指标
!!! comment "21 那篇按照数据集的 “内容” 聚类，这篇按 “指标” 聚类了"

1. 以质量分数为评估结果

    - 评价指标：Spearman Rank Correlation
    - 数据集：（一张表，比 2021 的有增加）
    - 实验结果对比：把各个方法的 Spearman Rank Corr. 列在了同一张表里，然后对比分析

2. 以 等级类别/ 质量等级排序 为评估结果（二合一）

    - 评价指标：

        - 等级类别：$\text{accuracy} = \frac{n_{分类正确}}{N}$
        - 质量等级排序：
            - 考虑一对样本 $(p_i,p_j)$，如果预测值的相对大小符合真实值的相对大小关系，则认为分类正确 `m+=1`
            - 对于整个数据集，若存在 $N$ 对样本，则 $\text{accuracy}_{rank} = \frac{m}{N}$

        - （新）NDCG：常用于搜索算法评估，评估预测排名序列与实际排名序列的相似程度

    - 数据集
    - 实验结果对比：把 accuracy 列了一下

### 3 限制与展望

1. 仅面向特定动作

    对每种动作单独训练模型进行评估，不能对复杂动作进行评估

2. 对分数的预测分析过于单一：只有回归模型

3. 虽然存在 “分阶段评估” 的方法，但切割方式简单

    - 大部分方法直接 **等分** 视频
    - 部分采用 **时序分割**，但可能导致时间信息丢失，分割准确率也会影响后续回归准确率

4. 对于以 **排序** 作为评估依据的方法，只有 pair-wise 的相对排序

    而在 **排序学习算法** 中，还有 Point-Wise Ranking / ListWise Sorting 两种

    缺少使用其他排序思想进行评估的指标

5. 数据集仍然欠缺

    对特定领域进行专家级注释的成本巨大，而 复杂动作的·多阶段 评价更是如此

    大部分模型以基于 image/video 的大规模数据进行预训练，急需基于 弱监督/半监督 的方法

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
