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

    - 跑了两天多，然后在第 118 Epoch 因为忘记插电然后寄了，这是 [Training Log](./assets/rep/CoRe%20Train%20MTL.log)

    <center>![](./assets/rep/CoRe%20log1.png)</center>
    <center>第一条 log，有一种“孩子生了”的美感</center>

    - giao 我以为断了就没了，读了下 src 发现还是有中间存档的（你不输出谁知道啊）

        > 模型在 `~/CoRe/experiments/CoRe_RT/MTL/try/best.pth`

- On AQA-7

    > 为了能快点跑完，把 MAX_EPOCH 改成 60 了

    ```bash
    bash ./scripts/train.sh 0,1 Seven try --Seven_cls 1
    ```

    - 修改了 MAX_EPOCH 跑了 60 个 Epoch，这是 [Training Log](./assets/rep/CoRe%20Train%20Seven.log)


### 模型验证

- On MTL-AQA

    1. 作者的[预训练模型](https://cloud.tsinghua.edu.cn/f/2dc6e1febc0e49fdb711/?dl=1)

        ```bash
        bash ./scripts/test.sh 0 MTL try --ckpts ./MTL_CoRe.pth
        ```

        ```text title="验证结果"
        ckpts @ 59 epoch( rho = 0.9541, L2 = 25.6865 , RL2 = 0.0024)

        [TEST] correlation: 0.953281, L2: 26.893236, RL2: 0.002463
        ```

    2. 自己跑了 117 个 Epoch [中道崩徂的模型 (2fcd)](https://pan.baidu.com/s/1vuDndWhKk5b4tFO4lO3Uyw)

        ```bash
        # 只是改了一下参数路径
        bash ./scripts/test.sh 0 MTL try --ckpts ./experiments/CoRe_RT/MTL/try/best.pth
        ```

        ```text title="验证结果"
        ckpts @ 108 epoch( rho = 0.9534, L2 = 30.5425 , RL2 = 0.0028)

        [TEST] correlation: 0.951908, L2: 31.714521, RL2: 0.002904
        ```

- On AQA-7 (自己跑了 60 个 Epoch 的模型)

    ```bash
    bash ./scripts/test.sh 0 MTL try --ckpts ./experiments/CoRe_RT/Seven/try/best.pth
    ```

    ```text title="验证结果"
    ckpts @ 58 epoch( rho = 0.8681, L2 = 63.5571 , RL2 = 0.0064)

    [TEST] correlation: 0.840793, L2: 89.697052, RL2: 0.008214
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

??? info "Source Code"

    - Src

        - Github repo for [TPT](https://github.com/baiyang4/aqa_tpt)

    - 准备工作

        > 因为用的是 CoRe 的同款数据集，所以在 135 那台服务器上
        >
        > 好的因为显卡内存不够，换到 137 了

        1. 在项目根目录下创建路径 `[TPT_root]/data/`

        2. 按照指定格式创建软链接

            ```bash title="@ [TPT_root]/data"
            ln -s /home/syf/CoRe/MTL-AQA/model_rgb.pth ./model_rgb.pth
            ln -s /nfs4-p1/zzy/AQA/AQA-TPT/data_preprocessed ./data_preprocessed
            ```

        3. 补充依赖 `einops, pandas, tensorboard, tqdm, matplotlib, seaborn`

        4. 设置环境变量 `export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:2048"`

            > 2048 还是不够

### 模型训练

- On MTL-AQA

    ```bash
    # 默认端口 29500 already in use，此处指定端口 29501
    # 逐渐放弃多卡训练 => 我超，动了！我永远爱 137 这台机子
    nohup python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train_pairencode1_decoder_1selfatt_self8head_ffn_sp_new.py --epoch_num=250 --dataset=MLT_AQA --bs_train=2 --bs_test=2 --use_pretrain=False --num_cluster=5 --margin_factor=3.2 --encode_video=False --hinge_loss=True --multi_hinge=True --d_model=512 --d_ffn=512 --exp_name=sp_new_103_5_3 >> MTL.log  2>&1 &
    ```

    !!! bug "降到 batch_size = 2 还是内存不够，仙逝了"

### 模型验证
> 作者暂未提供 test script，蹭一下隔壁 CoRe 的

- On MTL-AQA

    ```bash
    # @ ～/CoRe
    python test.py --gpu 0 --pt_w [path_to_model] --TSA
    ```

## 4 PECoP

??? info "Source Code"

    - Src

        - GitHub repo for [PECoP](https://github.com/Plrbear/PECoP)

        - [Kinetics pretrained I3D](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)

    - Dataset

        - PD4T (需要填写申请表orz)

        - MTL-AQA (CoRe ver.): 软链接至 `~/dataset/MTL-AQA`

??? quote "Prepareation"
    - 服务器: `10.214.211.137`

    - 准备工作：

        - 补充依赖 `tensorboardX, torch_videovision, ptflops`

            ```bash
            pip install git+https://github.com/hassony2/torch_videovision
            pip install ptflops
            ```

        - 在 `~/PECoP` 路径下，下载预训练 I3D backbone（同 CoRe）

### 模型训练

- On MTL-AQA

    修改了 `data_list` (数据标签文件路径) & `rgb_prefix` (数据集路径)

    ```bash
    python train.py --bs 16 --lr 0.001 --height 256 --width 256 --crop_sz 224 --clip_len 32 --rgb_prefix /home/syf/dataset/MTL-AQA/ --data_list /home/syf/dataset/MTL-AQA/final_annotations_dict_with_dive_number.pkl
    ```

    !!! bug "缺少必要的 list 文件"
        `./datasets/ucf101.py` 中规定对应的 list (UTF-8)文件格式为:

        ```
        sample_name(str)   action_label(int)    num_frames(int)    num_p(str)
        ```

        因为 MTL-AQA 缺少对应格式的文件，暂时没法动