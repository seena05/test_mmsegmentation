## 運用

### 初期実験の管理

基本的に特定のモデルを追加する場合を除いて、各自ブランチを作成して実験を行なって下さい。

そのうちで指定のデータセットでの評価結果が良く、共有されるべきものについてはハイパーパラメータや細かい調整の検討に移します。

### 最適化の検討

初期実験で良い成績のコンフィグと実行スクリプトをPRとして提出して下さい。
実行結果を確認して `main` ブランチにマージを行います。

### その他 (未決定事項)

* 最適化検討時のコード規約・スタイルややり方
* 横断的な実験コードやツール、便利機能の共有方法

## 使い方

```
$ clearml-task --project clearml-example-project \
               --name mmsegmentation_pspnet_r50-d8_512x512_4x4_40k_coco-stuff10k \
               --script mmsegmentation/tools/clearml_train.py \
               --branch main \
               --docker registry.dev.ai-ms.com/yumba/mmsegmentation:latest \
               --queue gpu01 \
               --args \
               dataset_id=862c283004cc45a08bd2b7ee97e7a954 \
               dataset_path=data/coco_stuff10k \
               config=configs/mmsegmentation/example/pspnet/pspnet_r50-d8_512x512_4x4_40k_coco-stuff10k.py \
               standalone=True \
               nproc_per_node=2
```

## レポジトリ一覧

| レポジトリ | 概要 | 備考 |
|---|---|---|
| [mmdetection](https://github.com/open-mmlab/mmdetection) | Object Detection Toolbox | mmdet v2.24.1 |
| [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | Segmentation Toolbox | mmseg v0.24.1 |
| [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | Object Detection Toolbox | |
| [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) | Segmentation Toolbox | |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Object Detection | |
| [CBNetV2](https://github.com/VDIGPKU/CBNetV2) | Object Detection | mmdet v2.14.0, mmcv v1.3.8 |
| [FocalNet](https://github.com/microsoft/FocalNet) | [Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) / [Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) | mmdet v2.11.0, mmcv v1.3.8 / mmseg v0.11.0, mmcv v1.3.0 |
| [PyCenterNet](https://github.com/Duankaiwen/PyCenterNet) | Object Detection | mmdet v2.11.0, mmcv v1.3.8 |

## 追加予定

| レポジトリ | 概要 |
|---|---|
| [DAB-DETR](https://github.com/IDEA-opensource/DAB-DETR) | Object Detection |
| [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch) | Segmentation |

## A. コンテナイメージの作り方

[docker-compose](https://docs.docker.com/compose/) を利用してコンテナをビルドしたり、ローカルでテストできるようにします。

### 方針

各レポジトリ毎にコンテナイメージを用意し、必要最小限の環境で行えるようにします。

以下は共有して利用するためのルールとして適合方法を記載します。

### 利用方法

`docker-compose build [サービス名]` や `docker-compose up [サービス名]` などレポジトリ名で個別に起動します。

### Dockerfileの置く場所

`docker/`以下にレポジトリ名に合わせて配置する

レポジトリ名が `PaddleDetection` であれば `docker/PaddleDetection` という命名にします。

### Dockerfileの書き方

基本的には以下のテンプレートに従って作成して下さい。

古いCUDAやpytorchのバージョンを必要とする場合はコメントを記載してください。
(A100などのマシンでは古いと動作しない可能性が高いですし、逆に古いマシンでは新しすぎるとドライバーなどの関係性で動作しないケースがあります)

```
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
# XXX: ここをカスタマイズする :XXX

#
# ClearML関連のパッケージ
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    clearml-agent \
    boto3
```

### docker-composeの書き方

`docker-compose.yaml`の`services`に追記する形で準備して下さい。

service名は大文字やハイフンが利用できないので、`container_name`でレポジトリ名を設定してください。

### 例 (PaddleDetection)

```
  paddle_detection:
    container_name: PaddleDetection
    build:
      context: .
      dockerfile: ./docker/PaddleDetection
    shm_size: '8gb'
    volumes:
      - "./PaddleDetection:/workspace/PaddleDetection"   # 最小限のマウントになるようにしてテストできるようにする
      - "./data:/workspace/data"
      - "./configs:/workspace/configs"
    command: /bin/sh -c "while sleep 1000; do :; done"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
```

### 起動方法

`docker-comopse up [サービス名]` で立ち上げ、別のシェルで `docker-compose exec [サービス名] /bin/bash` でコンテナ内に入る

## B. スタートアップスクリプト

統一的な使い方として、学習の開始は `clearml_train.py` という名前のファイルをエントリースクリプトに用意します。

※ `torch.distributed.launch` や `torchrun` のコード、または独自の分散処理を実装がされている場合がありますが、`clearml-task` から引数を渡す方法を上手く用意して下さい

## C. 動作確認スクリプト

新しいモデル/ライブラリを使う場合は、動作確認用のスクリプトが用意されているとスムーズに利用可能になります。

### example

`example/` に `[レポジトリ名].sh`の形式で単独実行可能なスクリプトを用意します。

#### 使い方

`$ bash example/[レポジトリ名].sh`

#### 作り方

手軽に実行を試すために、小さいサイズの公開データセットを利用したもので、元々のレポジトリにあるものを実行できるようにします。

プロジェクト名は `clearml-example-project` とします。

`[ClearML] public-datset` には `coco2017-min` や `voc0712-min` などを用意しています。

コンテナを作成し `registry.dev.ai-ms.com/yumba/[レポジトリ名]:latest` として*push*しておきます。

スクリプトは複数GPUで動作することを期待して `nproc_per_node=2` のような引数で開始できるようにしておきます。
