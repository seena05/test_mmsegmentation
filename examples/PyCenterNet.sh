# dataset: coco2017-min
clearml-task --project clearml-example-project \
             --name PyCenterNet_pycenternet_r50_fpn_giou_1x_coco \
             --script PyCenterNet/code/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/pycenternet:latest \
             --branch main \
             --queue gpu01 \
             --args \
               dataset_id=d6507bf2a2d54f66952a8cdd540f3a09 \
               dataset_path=data/coco \
               config=configs/PyCenterNet/example/pycenternet/pycenternet_r50_fpn_giou_1x_coco.py \
               nproc_per_node=2
