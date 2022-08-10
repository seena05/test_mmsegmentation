# dataset: coco2017-min
clearml-task --project clearml-example-project \
             --name swin_transformer_object_detection_mask_rcnn_focalnet_tiny_patch4_mstrain_480-800_adamw_1x_coco_srf \
             --script Swin-Transformer-Object-Detection/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/swin_transformer_object_detection:latest \
             --branch main \
             --queue gpu01 \
             --args \
               dataset_id=d6507bf2a2d54f66952a8cdd540f3a09 \
               dataset_path=data/coco \
               config=configs/FocalNet/example_mmdet/focalnet/mask_rcnn_focalnet_tiny_patch4_mstrain_480-800_adamw_1x_coco_srf.py \
               nproc_per_node=2
