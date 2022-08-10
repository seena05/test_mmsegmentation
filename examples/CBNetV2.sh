# dataset: coco2017-min
# GPUメモリを多く使用するのでA100などでしか動作しません
clearml-task --project clearml-example-project \
             --name CBNetV2_cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco \
             --script CBNetV2/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/cbnetv2:latest \
             --branch main \
             --queue a100 \
             --args \
               dataset_id=d6507bf2a2d54f66952a8cdd540f3a09 \
               dataset_path=data/coco \
               config=configs/CBNetV2/example/cbnet/cascade_rcnn_cbv2d1_r2_101_mdconv_fpn_20e_fp16_ms400-1400_coco.py \
               nproc_per_node=2
