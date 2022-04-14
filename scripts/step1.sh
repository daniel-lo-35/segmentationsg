#!/bin/bash
#SBATCH --job-name="PretrainOD"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --chdir=.
#SBATCH --output=step1out1.txt
#SBATCH --error=step1err1.txt
###SBATCH --test-only

sbatch_pre.sh

module load python/tensorflow-2-gpu

# python3 -m torch.distributed.launch pretrain_object_detector_rebar.py --config-file ../configs/pretrain_object_detector_rebar_coco.yaml --num-gpus 4 --resume DATASETS.VISUAL_GENOME.IMAGES ../datasets/vg/images DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY ../datasets/vg/VG-SGG-dicts-with-attri.json DATASETS.VISUAL_GENOME.IMAGE_DATA ../datasets/vg/image_data.json DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 ../datasets/vg/VG-SGG.h5 DATASETS.MSCOCO.ANNOTATIONS ../datasets/coco/annotations/ DATASETS.MSCOCO.DATAROOT ../datasets/coco/ OUTPUT_DIR ../output-step1
python3 pretrain_object_detector_rebar.py --config-file ../configs/pretrain_object_detector_rebar_coco.yaml --num-gpus 1 --resume DATASETS.VISUAL_GENOME.IMAGES ../datasets/vg/images DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY ../datasets/vg/VG-SGG-dicts-with-attri.json DATASETS.VISUAL_GENOME.IMAGE_DATA ../datasets/vg/image_data.json DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 ../datasets/vg/VG-SGG.h5 DATASETS.MSCOCO.ANNOTATIONS ../datasets/coco/annotations/ DATASETS.MSCOCO.DATAROOT ../datasets/coco/ OUTPUT_DIR ../output-step1

sbatch_post.sh
