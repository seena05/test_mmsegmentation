# dataset: coco2017-min
clearml-task --project clearml-example-project \
             --name mmdetection_faster_rcnn_r50_fpn_1x_coco \
             --script mmdetection/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/mmdetection:latest \
             --branch main \
             --queue gpu01 \
             --args \
               dataset_id=d6507bf2a2d54f66952a8cdd540f3a09 \
               dataset_path=data/coco \
               config=configs/mmdetection/example/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
               standalone=True \
               nproc_per_node=2
