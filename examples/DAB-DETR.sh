# dataset: coco2017-min
clearml-task --project clearml-example-project \
             --name dab-detr \
             --script DAB-DETR/clearml_train.py \
             --docker registry.dev.ai-ms.com/tsaito/clearml-yumba-model-search_dab_detr:latest \
             --branch dev/dab-detr \
             --queue gpu02 \
             --args \
               dataset_id=5b62d00e535e45bdaadea6f3dc0aea28 \
               dataset_path=data/coco \
               config=configs/DAB-DETR/example.yml \
               standalone=True \
               nproc_per_node=2

# dataset: 2022-05-09_yumba-dataset-v1-3class-wli
# clearml-task --project clearml-example-project \
#              --name dab-detr \
#              --script DAB-DETR/clearml_train.py \
#              --docker registry.dev.ai-ms.com/tsaito/clearml-yumba-model-search_dab_detr:latest \
#              --branch dev/dab-detr \
#              --queue a100 \
#              --args \
#                dataset_id=5b62d00e535e45bdaadea6f3dc0aea28 \
#                dataset_path=data/coco \
#                config=configs/DAB-DETR/yumba-dataset-v1-3class-wli.yml \
#                standalone=True \
#                nproc_per_node=2
