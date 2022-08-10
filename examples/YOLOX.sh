# dataset: coco2017-min
clearml-task --project clearml-example-project \
             --name YOLOX_yolox_s_default \
             --script YOLOX/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/yolox:latest \
             --branch main \
             --queue gpu01 \
             --args \
               dataset_id=d6507bf2a2d54f66952a8cdd540f3a09 \
               dataset_path=data/coco \
               batch_size=4 \
               devices=2 \
               name=yolox_s
