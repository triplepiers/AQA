# 复现结果

## 1 CoRe

??? info "Source Code"
    
    - Src

        - GitHub repo for [Group-aware Contrastive Regression for Action Quality Assessment](https://github.com/yuxumin/CoRe)

        - [Kinetics pretrained I3D model](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)

    - Dataset

        > 当前代码仅支持在 MTL-AQA / AQA-7 上进行训练

        - MTL-AQA
        
            - [Original](https://github.com/ParitoshParmar/MTL-AQA)

            - Prepared Dataset: [百度云 (`smff`)](https://pan.baidu.com/s/1ZUHyvYFia0cJtPp7pTfAbg)

        - AQA-7
        
            - Original

                ```bash
                mkdir AQA-Seven & cd AQA-Seven
                wget http://rtis.oit.unlv.edu/datasets/AQA-7.zip
                unzip AQA-7.zip
                ```

            - Prepared Dataset: [百度云 (`65rl`)](https://pan.baidu.com/s/1mcXo6g1XXhm9f0qL5lsNNw)

        - [JIGSAWS](https://cs.jhu.edu/~los/jigsaws/info.php)

    - Prerained CoRe
    
        - [for MTL-AQA](https://cloud.tsinghua.edu.cn/f/2dc6e1febc0e49fdb711/?dl=1)

??? quote "Prepareation"
    - 服务器: `10.214.211.135`

    - 准备工作：

        - 适用于 `NVIDIA RTX A6000` 的 Pytorch 环境

            ```bash
            pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
            ```

        - 补充依赖 `pyyaml` & `torch_videovision`

            ```bash
            pip install git+https://github.com/hassony2/torch_videovision
            ```

        - 下载预训练 I3D backbone

            ```bash
            wget https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_rgb.pth
            ```

        - `~/CoRe/MTL-AQA/new/new_total_frames_256s` 软链接至 `/nfs4-p1/zzy/AQA/CoRe-MTL/*`

    - 模型训练（在 `~/CoRe` 根路径下）

        - On MTL-AQA `bash ./scripts/train.sh 0,1 MTL try`

        <center>![](./assets/rep/CoRe%20log1.png)</center>
        <center>虽然只是第一条，但有一种“孩子生了”的美感</center>


## 2 TSA-Net

!!! info "Source Code"

    - Src

        - GitHub repo for [TSA-Net](https://github.com/Shunli-Wang/TSA-Net)

        - [Kinetics pretrained I3D model (i3dm)](https://pan.baidu.com/s/1L1MqzlTDFtbOKLYm1b1GpQ)

    - Dataset: [FR-FS (star)](https://pan.baidu.com/s/1Nkl6FlM2PcvbofegNjCIGA)

        > 新的一天从发现学长没下过这个 dataset 开始崩溃


## 3 TPT

## 4 PECoP

!!! info "Source Code"

    - Src

        - GitHub repo for [PECoP](https://github.com/Plrbear/PECoP)

    - Dataset

??? quote "Prepareation"
    - 服务器: `10.214.211.137`

    - 准备工作：

        - 补充依赖 `tensorboardX`

        - 在 `~/PECoP` 路径下，下载预训练 I3D backbone（同 CoRe）