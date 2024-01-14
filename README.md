# Tips
Action Quality Assessment（毕设救一下啊毕设）
  
## Server

- 使用的公钥是 `~/.ssh/id_rsa.pub`

- 服务器系统：Ubuntu 20.04.6 LTS (x86_64)

- CUDA Version: 12.2  

```bash
ssh syf@10.214.211.106 # * woc 原来是这台特别慢

ssh syf@10.214.211.135 # * /nfs4-p1/zzy, 数据库在 /nfs4-p1/zzy/AQA/CoRe-MTL

ssh syf@10.214.211.137

exit # log out
```

## Miniconda3 Envs

> 我超，3.6 版本的 wheel 构建需要交叉编译（我又没有 sudo 权限给服务器装 G++）

| EnvName | PythonVer | Desc. |
| :-----: | :-------: | ----- |
| torch39 | 3.9.18 | for CoRe |

## Tip

- 后台执行命令

    ```bash
    nohup [command] & # 命令将在断开 ssh 连接后继续执行，输出被写入 nohup.out

    # 查看并杀死后台运行的 wget 命令
    ps aux | grep wget
    kill -9 [pid]
    ```

- 软连接

    ```bash
    # 创建软链接
    ln -s [源文件或目录] [目标文件或目录]

    # 删除软链接
    rm -rf [软链接名字]

    # 修改软链接
    ln -snf [新的源文件或目录] [目标文件或目录]
    ```

- [bypy](https://github.com/houtianze/bypy) 下载百度云文件

    ```bash
    # 安装
    pip install bypy

    # 使用: 打开链接并复制验证码
    bypy info

    # 我的应用数据/bypy/* => currentDir
    bypy syncdown -v

    # 下载指定文件
    bypt downfile [pth/to/remoteFile] [path/to/localFile]
    ```


- pip 

    ```bash
    # 临时换源
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [some-package]

    # 从列表安装指定依赖
    pip install -r requirements.txt
    ```

- 查看服务器下的显卡信息

    ```bash
    nvidia-smi
    ```

- 远程连接时 bash 没有颜色

    ```bash
    force_color_prompt=yes # 取消 ～/.bashrc 中该行注释
    ```

- Miniconda3 安装

    ```bash
    # 下载脚本文件
    wget https://mirrors.ustc.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh

    # 为文件添加执行权限
    chmod u+x Miniconda3-latest-Linux-x86_64.sh

    # 执行安装脚本
    ./Miniconda3-latest-Linux-x86_64.sh
    # 一直 enter 到接受条款，默认安装在当前用户路径下
    ```