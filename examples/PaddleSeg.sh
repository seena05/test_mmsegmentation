# dataset: ADEChallengeData2016
# GPUメモリーの不足でA100以外では動作しません
clearml-task --project clearml-example-project \
             --name PaddleSeg_fastfcn_resnet50_os8_ade20k_480x480_120k \
             --script PaddleSeg/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/paddle_seg:latest \
             --branch main \
             --queue a100 \
             --args \
               dataset_id=0ad6856a2b044deca6682304517ebe1a \
               dataset_path=data/ADEChallengeData2016 \
               config=configs/PaddleSeg/example/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml \
               standalone=True \
               nproc_per_node=2 \
               do_eval=True
