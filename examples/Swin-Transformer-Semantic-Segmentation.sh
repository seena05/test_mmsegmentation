# dataset: ADEChallengeData2016
clearml-task --project clearml-example-project \
             --name swin_transformer_semantic_segmentation_upernet_focalnet_tiny_patch4_512x512_160k_ade20k_srf \
             --script Swin-Transformer-Semantic-Segmentation/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/swin_transformer_semantic_segmentation:latest \
             --branch main \
             --queue gpu01 \
             --args \
               dataset_id=0ad6856a2b044deca6682304517ebe1a \
               dataset_path=data/ADEChallengeData2016 \
               config=configs/FocalNet/example_mmseg/focalnet/upernet_focalnet_tiny_patch4_512x512_160k_ade20k_srf.py \
               nproc_per_node=2
