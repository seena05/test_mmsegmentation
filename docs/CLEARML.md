# ClearMLへの対応

## 概要

ClearML上で動作させるためには以下の対応が必要です。

1. エントリーコードをpythonで用意する (NOT シェルスクリプト)
2. clearml-dataでのデータセットの登録
3. 指定したデータのダウンロードをできるようにする (データセットとpretrainedモデル)
4. registryへのコンテナの登録
5. モデル/トレーニング中の成果物の保存

## 1. エントリーコードをpythonで用意する (NOT シェルスクリプト)

公開されているレポジトリや高速にトレーニングさせたい環境では分散学習が用意されています。

### 例 (mmdetection)

`bash mmdetection/tools/dist_train.sh [CONFIG] [GPU-NUM]` とすることで複数GPUで動作させることができます。

### 1. pytrochでの分散学習の仕組み

[TORCH DISTRIBUTED ELASTIC](https://pytorch.org/docs/stable/elastic/run.html) を参照してほしいのですが、`torch.distributed.launch` を使う旧来の方法と `torchrun (torch.distributed.run)` を使う新方式があります

`torchrun` は分散学習時にエラーなどでプロセスが停止した場合に復旧する機能を持たせ `torch.distributed.launch` を発展させたライブラリです

pytorchの公式の機能以外にも分散学習はスピードを改善させる方法が様々にあるので、ライブラリ毎に個別に実現されている場合があります (例: Paddleプロジェクト、YOLOXプロジェクト)

いづれにしても `clearml-task` に対応させるにはエントリーコードにpythonが必要で、かつ実行時に分散するGPU数などを指定できるようにするため一工夫必要です

### 2. エントリーコードを作る

pytorchのバージョンに合わせてコードを加工しないと動作しないことに注意して下さい。

以下はmmdetectionの `dist_train.sh` で使用されている pytorch 1.9以降に実装された `torchrun (torch.distributed.run)` を参照して作る方法を説明します。

#### エントリーポイントを特定する

`mmdetection/tools/dist_train.sh` は以下のようなコードになっていて、`torch.distributed.launch` を実行していることを確認します。他にはGPU数の指定が `--nproc_per_node=$GPUS` で行われている事を覚えておきます

```
#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
```

#### 実装コードを確認する

[torch.distributed.launch](https://github.com/pytorch/pytorch/blob/v1.11.0/torch/distributed/launch.py) の実装をGITレポジトリで見てみると、[torch.distributed.run](https://github.com/pytorch/pytorch/blob/v1.11.0/torch/distributed/run.py) が引数処理後に呼び出されている事が分かります。

[argparseを後から減らすのは難しい](https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse) ので、`parse_args` を書き換える方針で進めます。

#### 実装コードを元に改変する

`clearml-task` は引数に `args` を指定した内容がエントリースクリプトに渡されます。`argparse.ArgumentParser` の[書式のみ対応](https://github.com/allegroai/clearml/blob/master/clearml/task.py)しています。

`args` には `key=value` の組み合わせで指定する必要があるので、[positional arguments](https://docs.python.org/3/library/argparse.html) に該当するオプションをエントリーコードでは書けないことに注意します。

そこでコード中の `training_script` と `training_script_args` をコメントアウト(もしくは削除)します。

```
mmdetection/tools/clearml_train.py:L202
--
    #
    # Positional arguments.
    #

    # parser.add_argument(
    #     "training_script",
    #     type=str,
    #     help="Full path to the (single GPU) training program/script to be launched in parallel, "
    #     "followed by all the arguments for the training script.",
    # )

    # Rest from the training program.
    # parser.add_argument("training_script_args", nargs=REMAINDER)
```

最小限この変更だけでもエントリースクリプトとして起動させる事はできますが、configを渡さないと元の `mmdetection/tools/train.py` を呼び出せないので `config` オプションを追加します。

```
mmdetection/tools/clearml_train.py:L216
--
    # ClearML configs
    parser.add_argument(
        '--config',
        type=str,
        required=True,
    )
--
mmdetection/tools/clearml_train.py:L418
--
    args.training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    args.training_script_args = ['--seed', '0', '--launcher', 'pytorch', args.config]
```

この変更で元のシェルスクリプト(`mmdetection/tools/dist_train.sh`)を呼び出すように調整しています。

ここまでできたら、コンテナ内で `clearml_train.py` を実行できるか確かめます。データはローカルにあるフォルダーなどを `volumes` でマウントしたしたものを利用すれば動作確認できます。

## 2. clearml-dataでのデータセットの登録

[ClearML Data](https://clear.ml/docs/latest/docs/clearml_data/clearml_data) を利用してデータセットをオンデマンドに利用できるようにします。

通常のS3やNFSを使うのと比較してキャッシュ機能とClearMLの純正の機能であることからコード中で参照しやすくて便利です

### 1. データセットを準備する

できればCOCO形式に変換するなどすると使い易いでしょう

COCO2017を `data/coco` に配置した例を示します

```
$ cd data
$ ls coco
annotations  train2017  val2017
```

### 2. データセットプロジェクトを作る

`clearml-data create` によりデータセット用のプロジェクトをClearML上に作成できます

```
$ clearml-data create --project public-dataset --name coco2017
clearml-data - Dataset Management & Versioning CLI
Creating a new dataset:
New dataset created id=227750fe4a824e2e9cfe214d6d27b250
```

`id=25d3c555841a4e5583861bc74d44b0de` が出力されますが、以後このIDでデータセットの定義を行います

### 3. データを追加する

`clearml-data add` により一括でデータを追加できます

```
$ clearml-data add --files coco
clearml-data - Dataset Management & Versioning CLI
Adding files/folder to dataset id 227750fe4a824e2e9cfe214d6d27b250
Generating SHA2 hash for 2533 files
Hash generation completed
2533 files added
```

※ 説明用にデータ数を削減しています

### 4. データをアップロードする

アップロードする前に `~/clearml.conf` を編集して、`storage.dev.ai-ms.com` にアクセスできるようにしておきます

```
sdk {
    aws {
        s3 {
            credentials: [
                {
                     host: "storage.dev.ai-ms.com:9000"
                     bucket: "datasets"
                     key: "HW50AU3UR84S14TZHZV2"
                     secret: "D5RjOinInevAT9m7ioJVn6GItJOKCjsIwI3pVnGt"
                     multipart: false
                     secure: false
                }
            ]
        }
    }
}
```

`clearml-data upload` により一括でデータを追加できます

`--storage` に `s3://storage.dev.ai-ms.com:9000/datasets/` を指定して、NASにアップロードします

```
$ clearml-data upload --storage s3://storage.dev.ai-ms.com:9000/datasets/
clearml-data - Dataset Management & Versioning CLI
uploading local files to dataset id 227750fe4a824e2e9cfe214d6d27b250
Compressing local files, chunk 1 [remaining 2533 files]
File compression completed: total size 412.73 MB, 1 chunked stored (average size 412.73 MB)
Uploading compressed dataset changes 1/1 (2533 files 412.73 MB) to s3://storage.dev.ai-ms.com:9000/datasets
2022-05-07 17:24:13,603 - clearml.storage - INFO - Uploading: 5.00MB / 393.61MB @ 65.64MBs from /tmp/dataset.227750fe4a824e2e9cfe214d6d27b250.zip
2022-05-07 17:24:13,681 - clearml.storage - INFO - Uploading: 10.00MB / 393.61MB @ 64.08MBs from /tmp/dataset.227750fe4a824e2e9cfe214d6d27b250.zip
(...snip...)
```

※ デフォルトだとCleraMLが動作しているファイルサーバーにアップロードされますが、容量はそこまでないので必ずこちらを指定して下さい

### 5. データを固定する

`clearml-data close` によりデータセットを固定することができます。

```
$ clearml-data close
clearml-data - Dataset Management & Versioning CLI
Finalizing dataset id 227750fe4a824e2e9cfe214d6d27b250
2022-05-07 17:28:53,270 - clearml.Task - INFO - Waiting to finish uploads
2022-05-07 17:28:53,338 - clearml.Task - INFO - Finished uploading
Dataset closed and finalized
```

これでデータセットが利用可能になりました。
APIを通じて誤ってデータを変更することが無くなります。
新しいデータセットを作る場合は、継承して別のデータセットとして作成します。

※ `--id` を `create` 以降の全てのコマンドに記載しましたが、`close` まで状態として保持されるので一連の作業を連続して行う場合は省略可能です

## 3. 指定したデータのダウンロードをできるようにする (データセットとpretrainedモデル)

エントリースクリプトを実現するには、もう一つ学習に必要なデータを取得できるようにしなければなりません。

データをCLIから取得する場合は以下のコマンドで実行できます

```
$ pip install clearml boto3
$ clearml-data get --id d6507bf2a2d54f66952a8cdd540f3a09 --copy data/coco
```

この動作と同じコードは以下で実現できます

```
mmdetection/tools/clearml_train.py:L222
--
    parser.add_argument(
        '--dataset_id',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
    )
--
mmdetection/tools/clearml_train.py:L414
--
    if args.dataset_id and args.dataset_path:
        Dataset.get(dataset_id=args.dataset_id).get_mutable_local_copy(args.dataset_path)
```

引数に `dataset_id` と展開する名前 `dataset_path` を用意し、[clearml.Dataset](https://clear.ml/docs/latest/docs/references/sdk/dataset/) を利用してデータを取得します。

clearmlのコードを追加する際は、単独の場合(ClearML以外での実行)でも動作するように注意して実装しましょう。

※ clearml.Taskが有効でない場合などの処理

## 4. registryへのコンテナの登録

リモートマシンで作成したイメージを利用できるようにします

### 1. Web UIにログインする

[https://registry.dev.ai-ms.com/](https://registry.dev.ai-ms.com/) にログインして下さい(IPAのユーザー名でログインできます)

※ [Portus](http://port.us.org/) は `docker registry` に認証をつけるサービスですが、今回はIPA(LDAP)認証に対応しています

Webインターフェースからは作成したイメージなどを確認することができます

### 2. Docker Registryにログインする

一般の `docker registry` と同様に `docker login` での認証が必要となります

```
$ docker login registry.dev.ai-ms.com
Username: [IPAのユーザー名]
Password: [IPAのパスワード]
Authenticating with existing credentials...
WARNING! Your password will be stored unencrypted in /home/ykato/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

※ `docker logout registry.dev.ai-ms.com` でログアウトすることができます

### 3. Dockerイメージをアップロードする

作成したイメージの `IMAGE ID` を確認し、`registry.dev.ai-ms.com` 上でのイメージ名を付与します

```
$ docker images
REPOSITORY                                                TAG         IMAGE ID       CREATED          SIZE
clearml-example-mmdetection_clearml_example_mmdetection   latest      0254cfe68a0a   20 minutes ago   14.8GB
(...snip...)
```

`0254cfe68a0a` が作成したイメージのIDです

```
$ docker tag 0254cfe68a0a registry.dev.ai-ms.com/yumba/mmdetection:latest

$ docker push registry.dev.ai-ms.com/yumba/mmdetection:latest
Using default tag: latest
The push refers to repository [registry.dev.ai-ms.com/yumba/mmdetection]
02a697271783: Preparing
(...snip...)
latest: digest: sha256:839bdc18c4c8b3a881de0df9ad6276b10e2ea162d43b1e77fa656c357a40499a size: 10437
```

これでDockerイメージをリモートから参照することができるようになりました

※ registry上での命名規則として、`registry.dev.ai-ms.com/yumba/[サービス名]:latest` とします

## 5. モデル/トレーニング中の成果物の保存

コンテナ内で実行された結果は何もしない場合は保存されないので、学習結果としてのACCやloss、モデル(pth)などは適宜保存コードを用意する必要があります。

ここから先はレポジトリのコードに対して編集をしてClearML上でデバッグする必要があります。

### 1. CONFIGなど設定値を保存する (Taskの使い方)

`Task.upload_artifact` を利用するとテキストファイルなどをWebUI上から閲覧、ダウンロードできるようになります。

```
mmdetection/tools/train.py
--
    # save training config
    from clearml import Task
    task = Task.current_task()
    if task:
        task.upload_artifact(name='Config', artifact_object=cfg.pretty_text)
```

WebUI上では `ARTIFACTS > OTHER` に `Config` として表示されるようになります

※ 詳しくは[公式サイト(Artifacts)](https://clear.ml/docs/latest/docs/fundamentals/artifacts/)をご確認ください

### 2. ACC/LOSSなど学習途中の値を保存する (Loggerの使い方)

WebUI上では `RESULTS > CONSOLE` として残るので、究極的にはLoggerは不要であるとも言えますが、グラフになっていると視覚的に手早く確認できるので共有するならばある程度は可視化できるようにした方が良いでしょう

#### Tensorboardを有効にする

一番簡単なのは [Tensorboard](https://www.tensorflow.org/tensorboard) のログの設定があるならば、それを有効にすることです。

ClearMLは`Tensorboard`のログについては自動的にグラフにする機能があります。

例えば `mmdetection` では **log_config** に対して `TensorboardLoggerHook` を有効にするだけで対応できます。

```
configs/mmdetection/example/_base_/schedules/default_runtime.py:L2
--
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### テキストログからLoggerを仕込む

コンソールに学習過程の結果が出力されている事をヒントに、値を指定している箇所を探して `clearml.Logger` を入れます。

例えば `PaddleDetection` では **LogPrinter** の `logger.info` でlossなどの出力がされているので、この場所で以下のようにコードを追加します

```
PaddleDetection/ppdet/engine/callbacks.py:L146
--
                    clearml_logger = Logger.current_logger()
                    if clearml_logger:
                        for loss_name, loss_value in training_staus.get().items():
                            clearml_logger.report_scalar('train', loss_name, iteration=(steps_per_epoch * epoch_id + step_id), value=loss_value)
```

WebUI上では `RESULTS > SCALARS` にグラフで表示されるようになります。

※1 詳しくは[公式サイト(Logger)](https://clear.ml/docs/latest/docs/fundamentals/logger)をご確認ください

※2 構造的なプロジェクトであれば複数のLoggerを切り替えられるように作られているのでインターフェースを合わせて作る方法もありますが、個別に作り込むのは大変なのでやり易さを優先しましょう

### 3. MODEL(pth)など学習したモデルを保存する (OutputModelの使い方)

モデルデータは肥大化する環境で全部を残すようにするとストレージに優しくないというのとトレーニング時間がアップロード時間で伸びてしまうので、bestモデルと、last checkpointを残すような方法が良いでしょう。

こちらもコードを辿って `torch.save` が行われているポイントを探します。ただし、一般にモデルの保存前に色々な処理が行われるのでloggerなどの出力や実装名などから推論する方が早い場合が多いです。

`mmdetection` では `mmdet.core.evaluation.eval_hooks` でbestモデルの判定を行なっています。configで `evaluation = dict(save_best='auto', interval=1, metric='bbox')` のように `save_bast` を指定することで有効になるようになっています。

以下のコードを追加して、`self.best_ckpt_path` にあるモデルを `OutputModel` を使ってアップロードするようにしています。

```
mmdetection/mmdet/core/evaluation/eval_hooks.py:L67
--
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)
            if self.output_model and self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                step = runner.epoch if self.by_epoch else runner.iter
                self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)
```

WebUI上では `RESULTS > ARTIFACTS > OIUTPUT MODELS` でダウンロードできるようになります。

※1 連続してモデルをアップロードすると先のモデルのアップロードがキャンセルされてしまうので注意点して下さい

※2 最新のmmdetの事例です。数年前はベストモデルを残すような機能はなかったので、自前で用意する必要があります
