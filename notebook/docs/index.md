# AQA: 动作质量评估

- 李渊明 的 GitHub 仓库：[Awesome AQA](https://github.com/Lyman-Smoker/Awesome-AQA)

## 1 Requirement

> Explainable learning for sports movement optimization

利用深度神经网络模型对体育动作进行打分以实现运动员水平的提升是一个值得研究的问题：

- 然而，由于<u>深度学习模型的可解释性差</u>，导致打分只能做到知其然不知其所以然。

- 本课题提出 **体育动作优化的可解释学习**：

    尝试研究一种可解释的打分模型，使运动员更容易理解需要改进的技术点，提高运动成绩。

## 2 References

### 2021

- [CoRE](./ref/2021/CoRe.pdf): Group-aware Contrastive Regression for Action Quality Assessment

    **(ICCV 2021)** [[Paper]](http://openaccess.thecvf.com//content/ICCV2021/papers/Yu_Group-Aware_Contrastive_Regression_for_Action_Quality_Assessment_ICCV_2021_paper.pdf) [[Code]](https://github.com/yuxumin/CoRe)

- [TSA-Net](./ref/2021/TSA-Net.pdf): Tube Self-Attention Network for Action Quality Assessment 

    **(ACM-MM 2021 Oral)** [[Paper]](https://arxiv.org/pdf/2201.03746) [[Code]](https://github.com/Shunli-Wang/TSA-Net)

- [A Survey of Video-based Action Quality Assessment](./ref/2021/Video-based.pdf)

### 2022

- [TPT](./ref/2022/TPA.pdf): Action Quality Assessment with Temporal Parsing Transformer [[Code]](https://github.com/baiyang4/aqa_tpt)

### 2023

- [PECoP](./ref//2023/PECoP.pdf): Parameter Efficient Continual Pretraining for Action Quality Assessment 

    **(WACV 2023 Oral)** [[Paper]](https://arxiv.org/pdf/2311.07603.pdf) [[Code]](https://github.com/Plrbear/PECoP)

    - 为 I3D Module 引入 3D-Adapter 进行 Continual Pretraining

        - 更好的学习 domain-specific 任务、减少需要调整的参数

        - 使用 VSPP (transform-based) 作为 SSL Pretext Task

    - 证明在 R3D 中引入 3D-Adapter 同样有效

- [FSPN](./ref/2023/STPN.pdf): Fine-Grained Spatio-Temporal Parsing Network for Action Quality Assessment 

    **(TIP 2023)** [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10317826)

- [IRIS](./ref/2023/IRIS.pdf): Interpretable Rubric-Informed Segmentation for Action Quality Assessment 

    **(ACM-IUI 2023)** [[Paper]](https://arxiv.org/pdf/2303.09097.pdf)