# dataset: coco_stuff10k
clearml-task --project orita-el \
             --name test \
             --script mmsegmentation/tools/clearml_train.py \
             --docker registry.dev.ai-ms.com/yumba/mmsegmentation:latest \
             --branch main \
             --queue orita \
             --args \
               dataset_id=9277bc6da7f74dfeb365c05e779f9908 \
               dataset_path=data/esophageal_dataset_3class \
               config=mmsegmentation/configs/swin/upernet_swin_small_patch4_window7_512x512_20k_esophageal_3class_dataset_pretrain_224x224_1K.py \
               standalone=True \
               nproc_per_node=2
