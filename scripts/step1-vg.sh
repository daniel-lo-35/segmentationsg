#!/bin/bash
#SBATCH --job-name="PretrainOD"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --chdir=.
#SBATCH --output=step1out-vg.txt
#SBATCH --error=step1err-vg.txt
###SBATCH --test-only

sbatch_pre.sh

module load python/tensorflow-2-gpu

python3 pretrain_object_detector_withcoco.py --config-file ../configs/pretrain_object_detector_coco.yaml --num-gpus 1 --resume DATASETS.VISUAL_GENOME.IMAGES ../datasets-real/vg/images DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY ../datasets-real/vg/VG-SGG-dicts-with-attri.json DATASETS.VISUAL_GENOME.IMAGE_DATA ../datasets-real/vg/image_data.json DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 ../datasets-real/vg/VG-SGG-with-attri.h5 DATASETS.MSCOCO.ANNOTATIONS ../datasets-real/coco/annotations/ DATASETS.MSCOCO.DATAROOT ../datasets-real/coco/ OUTPUT_DIR ../output-step1-vg \
DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES 5000 MODEL.ROI_HEADS.MASK_NUM_CLASSES 80 MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES 80 \
SOLVER.REFERENCE_WORLD_SIZE 4 \
SOLVER.IMS_PER_BATCH 8 SOLVER.MAX_ITER 10000


sbatch_post.sh
