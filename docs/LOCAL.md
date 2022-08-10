# ローカル環境でのトレーニング

## 概要

`docker-compose` を利用してコンテナを準備し、実行環境をコード化した機械学習を行う方法を解説します。入っているパッケージやライブラリなどのバージョンをコード化することで、手元の環境以外で再現できるようにしましょう。

## 事前準備

1. [docker with NVIDIA container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
2. [docker-compose](https://docs.docker.com/compose/install/)

## 最初の一歩

サービス(レポジトリ)として `mmdetection` を利用する例を示します。　

### 1. コンテナをビルドする

```
$ cd [GIT-ROOT]
$ docker-compose build mmdetection
```

### 2. コンテナを立ち上げる

```
$ docker-compose up mmdetection
```

### 3. コンテナに入る

```
$ docker-compose exec mmdetection /bin/bash
root@5933e43a1587:/workspace#
```

### 4. コードを実行する

```
root@5933e43a1587:/workspace# python mmdetection/tools/clearml_train.py --config=configs/mmdetection/example/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
...
```

### 5. コードをコンテナ外から実行する

```
$ cd [GIT-ROOT]
$ docker-compose exec mmdetection python mmdetection/tools/clearml_train.py --config=configs/mmdetection/example/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
...
```

### 6. 解説

`docker-compose.yaml` には以下のようにコンテナを立ち上げ続けるようなループコマンド( `command` の部分)が入っています。またgpuをコンテナ内で利用できるような指定をしています( `shm_size` と `- driver: nvidia` の部分)。

`volumes` でソースコード( `mmdetection` フォルダ)やデータセットを配置する `data` 、設定ファイルを配置する `configs` などをコンテナ内で参照できるようにしています。

`dockerfile` で `docker/mmdetection` を指定していますが、これは一般的な `Dockerfile` であり、コンテナを定義しています。

```
(抜粋)
  mmdetection:
    container_name: mmdetection
    build:
      context: .
      dockerfile: ./docker/mmdetection
    shm_size: '8gb'
    volumes:
      - "./mmdetection:/workspace/mmdetection"
      - "./data:/workspace/data"
      - "./configs:/workspace/configs"
      - "./pretrain:/workspace/pretrain"
    command: /bin/sh -c "while sleep 1000; do :; done"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

`Dockefile` の記述方法は

```
docker/mmdetection
--
#
# NVIDIA NGC コンテナを使う
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# 動作確認されているので、独自にコンテナをビルドするより手間が少ないです
#
FROM nvcr.io/nvidia/pytorch:22.02-py3

#
# 日本時間に設定する (ログを見やすくするため)
#
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev tzdata \
&&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

#
# プロジェクトに必要なパッケージをインストールする
#
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    mmcv-full

#
# requirements/runtime.txt をインストールする
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    matplotlib \
    numpy \
    pycocotools \
    six \
    terminaltables \
    'mmcls>=0.20.1' \
    packaging \
    prettytable

#
# ClearML関連のパッケージ
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    clearml-agent \
    boto3
```

## コンテナの作り方

### 1. 動作を確認する

プロジェクトページなどを参考にコンテナ内に入った状態で通常通りに学習できるか確認をします。pytorchやCUDAのバージョンなどの違いにより動作しないことが分かったら、コンテナのイメージを変えてやり直します。

コンテナはNVIDIA NGCが便利です。pytorchやCUDAライブラリなどが動作確認され、毎月更新が行われています。

環境変化が早いのでpytorchのバージョンやCUDAのバージョンに合わせてコンテナを選択しましょう。[PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) を見て、**CUDA Toolkit** と **PyTorch** のバージョンが合う **Container Version** を採用します。

`Dockerfile` の `FROM` を調整して使用イメージを決定します。

```
#
# NVIDIA NGC コンテナを使う
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# 動作確認されているので、独自にコンテナをビルドするより手間が少ないです
#
FROM nvcr.io/nvidia/pytorch:22.02-py3
```

### 2. 必要なパッケージを特定する

コンテナに入って行った作業はコード化するためには、何をしたかを把握しておく必要があります。

一般には `requirements.txt` に記載されることが多いですが、`setup.py` 内に記載されている場合もあります。また `README.md` などのテキストに手順が書かれている場合などがあります。

動作確認を急ぐあまり何をしたのか忘れないようにしましょう。

`Dockerfile` の `RUN` を利用してパッケージを追加するコードを作成します。

```
#
# requirements/runtime.txt をインストールする
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    matplotlib \
    numpy \
    pycocotools \
    six \
    terminaltables \
    'mmcls>=0.20.1' \
    packaging \
    prettytable
```

※1 動作させるための最小のパッケージになるように注意しましょう

※2 `pip freeze` でパッケージを確認する場合は、NGCコンテナは最初からパッケージが多く入っていることに注意して差分を上手く見つけましょう

※3 [pipdeptree](https://pypi.org/project/pipdeptree/) を使うと最小のパッケージを出力できます

### 3. コンテナをゼロから作る

これまでの実行手順で必要十分か確かめるためには、一旦コンテナを消してから再作成して動作を検証します。

```
$ docker-compose up mmdetection
[Ctrl+C]
$ docker-compose rm mmdetection
$ docker-compose build mmdetection
$ docker-compose up mmdetection
(別端末から)
$ docker-compose exec mmdetection python mmdetection/tools/clearml_train.py --config=configs/mmdetection/example/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```
