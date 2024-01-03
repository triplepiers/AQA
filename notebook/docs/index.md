# AQA: 动作质量评估

- 李渊明 的 GitHub 仓库：[Awesome AQA](https://github.com/Lyman-Smoker/Awesome-AQA)

## 1 Requirement

> Explainable learning for sports movement optimization

利用深度神经网络模型对体育动作进行打分以实现运动员水平的提升是一个值得研究的问题：

- 然而，由于<u>深度学习模型的可解释性差</u>，导致打分只能做到知其然不知其所以然。

- 本课题提出 **体育动作优化的可解释学习**：

    尝试研究一种可解释的打分模型，使运动员更容易理解需要改进的技术点，提高运动成绩。

## 2 Msg Logs

!!! note "2023-12-28"
    我个人觉得比较好的文章就是 [TSA-Net](./ref/2021/TSA-Net.pdf) & [CoRe](./ref/2021/CoRe.pdf)，这两篇是两种流派的基础方法：

    1. Backbone + attention + 回归头 的传统结构

    2. 对比学习框架 的结构

    然后我最近比较关注 [PECoP](./ref/2023/PECoP.pdf) 这一篇，因为感觉架构上已经水了不少了。下面还有做的空间：
    
    - 怎么优化表示
    
    - 找一个持续学习和自监督、半监督的方法

!!! note "2023-12-30"
    > 终于见到黎叔了，好耶！

    - 主要聚焦于”可解释性“，accuracy 接近 SOTA 即可

        输出可比较的 <u>视频片段</u>（对比 2D 中的热力图）

    - 可以不用即时出成果（25fps），可以输入图像 - 后续输出结果

    - 可以先基于 skeleton 做，如果后续有更好的就可以进行批判

        骨架不太细致：手部动作评委能看到，但是骨架识别不出来



## 3 References

### 2021

- [CoRE](./ref/2021/CoRe.pdf): Group-aware Contrastive Regression for Action Quality Assessment

    **(ICCV 2021)** [[Paper]](http://openaccess.thecvf.com//content/ICCV2021/papers/Yu_Group-Aware_Contrastive_Regression_for_Action_Quality_Assessment_ICCV_2021_paper.pdf) [[Code]](https://github.com/yuxumin/CoRe)

- [TSA-Net](./ref/2021/TSA-Net.pdf): Tube Self-Attention Network for Action Quality Assessment 

    **(ACM-MM 2021 Oral)** [[Paper]](https://arxiv.org/pdf/2201.03746) [[Code]](https://github.com/Shunli-Wang/TSA-Net)

- [A Survey of Video-based Action Quality Assessment](./ref/2021/Video-based.pdf)

### 2022

- [TPA](./ref/2022/TPA.pdf): Action Quality Assessment with Temporal Parsing Transformer

### 2023

- [PECoP](./ref//2023/PECoP.pdf): Parameter Efficient Continual Pretraining for Action Quality Assessment 

    **(WACV 2023 Oral)** [[Paper]](https://arxiv.org/pdf/2311.07603.pdf) [[Code]](https://github.com/Plrbear/PECoP)