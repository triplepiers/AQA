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

        - 将 `/nfs4-p1/zzy/AQA` 中的 Seven.tar 复制至用户路径，并使用 `tar -xf [tarFile]` 解压，后创建软链接

### 模型训练

- On MTL-AQA

    ```bash
    bash ./scripts/train.sh 0,1 MTL try
    ```

    - 跑了两天多，然后在第 118 Epoch 因为忘记插电然后寄了，这是 [Training Log](./assets/rep/CoRe%20Train.log)

    <center>![](./assets/rep/CoRe%20log1.png)</center>
    <center>第一条 log，有一种“孩子生了”的美感</center>

    - giao 我以为断了就没了，读了下 src 发现还是有中间存档的（你不输出谁知道啊）

        > 模型在 `~/CoRe/experiments/CoRe_RT/MTL/try/best.pth`

- On AQA-7

    > 为了能快点跑完，把 MAX_EPOCH 改成 60 了

    ```bash
    bash ./scripts/train.sh 0,1 Seven try --Seven_cls 1
    ```

    这次挂后台了，log 在 ~/CoRe/seven.log

### 模型验证

- On MTL-AQA

    1. 作者的[预训练模型](https://cloud.tsinghua.edu.cn/f/2dc6e1febc0e49fdb711/?dl=1)

        ```bash
        bash ./scripts/test.sh 0 MTL try --ckpts ./MTL_CoRe.pth
        ```

        ```text title="验证结果"
        ckpts @ 59 epoch( rho = 0.9541, L2 = 25.6865 , RL2 = 0.0024)

        [TEST][0/353] 	 Batch_time 112.51 	 Data_time 134.18 
        [TEST][40/353] 	 Batch_time 0.98 	 Data_time 78.54 
        [TEST][80/353] 	 Batch_time 1.07 	 Data_time 90.37 
        [TEST][120/353]  Batch_time 0.98 	 Data_time 0.00 
        [TEST][160/353]  Batch_time 0.98 	 Data_time 0.00 
        [TEST][200/353]  Batch_time 0.98 	 Data_time 0.00 
        [TEST][240/353]  Batch_time 0.98 	 Data_time 0.00 
        [TEST][280/353]  Batch_time 0.98 	 Data_time 0.00 
        [TEST][320/353]  Batch_time 0.97 	 Data_time 0.00 

        [TEST] correlation: 0.953281, L2: 26.893236, RL2: 0.002463
        ```

    2. 自己跑了 117 个 Epoch [中道崩徂的模型 (2fcd)](https://pan.baidu.com/s/1vuDndWhKk5b4tFO4lO3Uyw)

        ```bash
        # 只是改了一下参数路径
        bash ./scripts/test.sh 0 MTL try --ckpts ./experiments/CoRe_RT/MTL/try/best.pth
        ```

## 2 TSA-Net

??? info "Source Code"

    - Src

        - GitHub repo for [TSA-Net](https://github.com/Shunli-Wang/TSA-Net)

        - [Kinetics pretrained I3D model (i3dm)](https://pan.baidu.com/s/1L1MqzlTDFtbOKLYm1b1GpQ)

    - Dataset: [FR-FS (star)](https://pan.baidu.com/s/1Nkl6FlM2PcvbofegNjCIGA)

        > 新的一天从发现学长没下过这个 dataset 开始崩溃

??? quote "Prepareation"

    - 服务器: `10.214.211.137`

    - 准备工作：

        - 在 `~/dataset` 中下载 FR-FS 数据集，并建立软链接至 `/TSA/data/FRFS`

        - 在 `~/TSA/data` 路径下，下载预训练 backbone

        - `AttributeError: module 'distutils' has no attribute 'version'`

            tensordBoard 版本与 setuptool 不匹配，重装 `setuptools==58.0.4`

        - `Descriptors cannot be created directly.`

            将 protobuf 降级至 3.20.X

### 模型训练

```bash title="使用 nohup 进行后台训练"
# train TSA-Net >> TSA.log
nohup python test.py --gpu 0 --pt_w ~/TSA/Exp/TSA-USDL/best.pt --TSA >> TSA.log  2>&1 &
# train Plain-Net >> plain.log
nohup python test.py --gpu 0 --pt_w ~/TSA/Exp/USDL/best.pt >> plain.log  2>&1 &
```

训练结束后将在：`~/TSA/Exp/TSA-USDL` 和 `~/TSA/Exp/USDL` 产生对应的 `best.pt`（其实还有 `train.log`）

### 模型验证

??? bug "WTM 笑死，他这个 test 的代码有 bug"
    > 另外它 `README` 里给的测试代码是不对的: 
    >
    > `weight_logger` 用的不是相对 CWD 的路径 & `train.py` 生成的模型后缀是 `.pt` (而不是 `.pth`)

    - Bug Tip: `AttributeError: 'Namespace' object has no attribute 'type'`

    - Fix:

        ```python
        # "/home/syf/TSA/test.py", line 206
        # from
        base_logger = get_logger(f'{args.model_path}/{args.type}.log', args.log_info)

        # to (随便起啥名都行)
        base_logger = get_logger(f'{args.model_path}/test.log', args.log_info)
        ```

```bash title="用 nohup 挂后台"
nohup python test.py --gpu 0 --pt_w Exp/TSA-USDL/best.pth --TSA >> TSA.log  2>&1 &
nohup python test.py --gpu 0 --pt_w Exp/USDL/best.pth >> plain.log  2>&1 &
```

下面是验证时的通用参数：

```text
dp: False                           weight_decay: 1e-05
num_workers: 8                      num_epochs: 20
test_batch_size: 8                  temporal_aug: 0
std: 5                              lr: 0.0001
```

<center>验证结果</center>

<center>
<table>
<tr>
<th>Model</th>
<th>Accuracy</th>
</tr>
<tr>
<td>TSA-Net</td>
<td>98.08%</td>
</tr>
<tr>
<td>Plain-Net</td>
<td>95.19%</td>
</tr>
</table>
</center>


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

        