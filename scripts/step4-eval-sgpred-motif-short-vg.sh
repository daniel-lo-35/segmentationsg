#!/bin/bash
#SBATCH --job-name="VGMotif"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --chdir=.
#SBATCH --output=step4mouts-vg.txt
#SBATCH --error=step4merrs-vg.txt
###SBATCH --test-only

sbatch_pre.sh

module load python/tensorflow-2-gpu

python3 train_SG_segmentation_head.py --eval-only --config-file ../configs/sg_dev_masktransfer.yaml --num-gpus 1 --resume DATALOADER.NUM_WORKERS 2 \
MODEL.WEIGHTS ../output-step1-vg/resnet_pretrain_weights.pth \
OUTPUT_DIR ../output-step3-motif-short-vg \
DATASETS.VISUAL_GENOME.IMAGES ../datasets-real/vg/images DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY ../datasets-real/vg/VG-SGG-dicts-with-attri.json DATASETS.VISUAL_GENOME.IMAGE_DATA ../datasets-real/vg/image_data.json DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 ../datasets-real/vg/VG-SGG-with-attri.h5 \
DATASETS.MSCOCO.ANNOTATIONS ../datasets-real/coco/annotations/ DATASETS.MSCOCO.DATAROOT ../datasets-real/coco/ \
MODEL.MASK_ON True  \
MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX False MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION True MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE 'Weighted' \
MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION True TEST.EVAL_PERIOD 50000 \
MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS False \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK True \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK False SOLVER.IMS_PER_BATCH 16 DATASETS.SEG_DATA_DIVISOR 2 \
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'MotifSegmentationPredictorC' MODEL.ROI_HEADS.REFINE_SEG_MASKS False TEST.DETECTIONS_PER_IMAGE 40 \
SOLVER.MAX_ITER 2000 \
DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES 5000 MODEL.ROI_HEADS.MASK_NUM_CLASSES 80 MODEL.ROI_HEADS.NUM_OUTPUT_CLASSES 80 MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES 50 \
SOLVER.REFERENCE_WORLD_SIZE 4 \
MODEL.ROI_SCENEGRAPH_HEAD.ZERO_SHOT_TRIPLETS /home/r09521612/segmentationsg/segmentationsg/evaluation/datasets/vg/zeroshot_triplet.pytorch
# DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP False # This is default true, which will remove all the intersections

sbatch_post.sh
