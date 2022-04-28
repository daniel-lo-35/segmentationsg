import enum
import sys
import os
from typing import OrderedDict
import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from segmentationsg.engine import SceneGraphSegmentationTrainer
from segmentationsg.data import add_dataset_config, VisualGenomeTrainData, register_datasets
from segmentationsg.modeling.roi_heads.scenegraph_head import add_scenegraph_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from segmentationsg.modeling import *

import logging
import time
import datetime
import copy
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds

parser = default_argument_parser()

def register_coco_data(args):
    # annotations = json.load(open('/h/skhandel/SceneGraph/data/coco/instances_train2014.json', 'r'))
    # classes = [x['name'] for x in annotations['categories']]
    # classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    classes = ['intersection', 'spacing']
    MetadataCatalog.get('coco_train_2014').set(thing_classes=classes, evaluator_type='coco')
    annotations = args.DATASETS.MSCOCO.ANNOTATIONS
    dataroot = args.DATASETS.MSCOCO.DATAROOT
    register_coco_instances("coco_train_2017", {}, annotations + 'instances_train2017.json', dataroot + '/train2017/')
    register_coco_instances("coco_val_2017", {}, annotations + 'instances_val2017.json', dataroot + '/val2017/')
    # MetadataCatalog.get('coco_train_2017').set(thing_classes=classes, evaluator_type='coco')

def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert(cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']) , "Mode {} not supported".format(cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)
    register_coco_data(cfg)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")
    return cfg

def main(args):
    torch.cuda.empty_cache()

    cfg = setup(args)
    # if args.eval_only:
        
    model = SceneGraphSegmentationTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # res = SceneGraphSegmentationTrainer.test(cfg, model)
    res = scene_graph_visualize(cfg, model)
    # if comm.is_main_process():
    #     verify_results(cfg, res)
    return res
        
    # trainer = SceneGraphSegmentationTrainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()

def scene_graph_visualize(cfg, model):
    logger = logging.getLogger(__name__)

    model.eval()

    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        if dataset_name != ("VG_test" or "VG_train"):
            continue

        data_loader = SceneGraphSegmentationTrainer.build_test_loader(cfg, dataset_name)
        output_folder = os.path.join(cfg.OUTPUT_DIR, "visualize")
        os.makedirs(output_folder, exist_ok=True)

        # Code below adapted from evaluator.py, scenegraph_inference_on_dataset()
        num_devices = get_world_size()
        logger = logging.getLogger('detectron2')
        logger.info("Start visualize on {} images".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0

        ground_truths = []
        predictions = []

        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                if len(inputs[0]['instances']) > 40:
                    continue
                start_compute_time = time.perf_counter()

                outputs = model(inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                ground_truths, predictions = process(inputs, outputs, ground_truths, predictions)

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start

                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    # logger.info("Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta)))
                    log_every_n_seconds(
                        logging.INFO,
                        "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)
                        ),
                        n=5,
                        name='detectron2'
                    )

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        # Code below adapted from sg_evaluation.py, class SceneGraphEvaluator(DatasetEvaluator), evaluate()
        _distributed = True
        _zero_shot_triplets = torch.load(cfg.MODEL.ROI_SCENEGRAPH_HEAD.ZERO_SHOT_TRIPLETS , map_location=torch.device("cpu")).long().numpy() + 1
        _mode = _mode_from_config(cfg)

        if _distributed:
            comm.synchronize()
            logger.info("Gathering data")
            predictions = comm.gather(predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            ground_truths = comm.gather(ground_truths, dst=0)
            ground_truths = list(itertools.chain(*ground_truths))

            if not comm.is_main_process():
                return {}
        # else:
        #     predictions = predictions
        #     ground_truths = ground_truths

        logger.info("Predictions Gathered")

        if len(predictions) == 0:
            logger.warning("[SceneGraphEvaluator] Did not receive valid predictions.")
            return {}

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "scenegraph_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save({'groundtruths':ground_truths, 'predictions':predictions}, f)
        # self._logger.info("Saving output prediction")

        logger.info("Computing Scene Graph Metrics")
        num_rel_category = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
        iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD

        logger.info("Preparing Global Container")
        #Prepare Global container
        global_container = {}
        global_container['zeroshot_triplet'] = _zero_shot_triplets
        global_container['result_dict'] = {}
        global_container['mode'] = _mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres

        
        for i , (groundtruth, prediction) in tqdm(enumerate(zip(ground_truths, predictions)),desc='Computing recalls'):
            visualize_one_image(cfg, groundtruth, prediction, global_container, i, output_folder)

def visualize_one_image(cfg, groundtruth, prediction, global_container, i, output_folder):
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth['relation_tuple'].long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth['gt_boxes'].tensor.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth['labels'].long().detach().cpu().numpy()           # (#gt_objs, )
    # import ipdb; ipdb.set_trace()
    # about relations
    local_container['pred_rel_inds'] = prediction['rel_pair_idxs'].long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction['pred_rel_scores'].detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction['instances'].pred_boxes.tensor.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction['instances'].pred_classes.long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction['instances'].scores.detach().cpu().numpy()              # (#pred_objs, )
    # # import pdb; pdb.set_trace()
    # # to calculate accuracy, only consider those gt pairs
    # # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # # for sgcls and predcls
    # if mode != 'sgdet' and 'SGPairAccuracy' in self._metrics:
    #     self._evaluators['SGPairAccuracy'].prepare_gtpair(local_container)

    # # to calculate the prior label based on statistics
    # if 'SGZeroShotRecall' in self._metrics:
    #     self._evaluators['SGZeroShotRecall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    #### TODO: change below code for getting image and objects ####
    # image_idx = 2
    local_container['image_id'] = prediction['image_id']
    image_idx = local_container['image_id']

    # image_path = os.path.join('images' ,str(image_idx) + '.jpg')
    image_path = os.path.join(cfg.DATASETS.VISUAL_GENOME.IMAGES, str(image_idx) + '.jpg')

    # objects = json.load(open('objects.json'))
    # relationships = json.load(open('relationships.json'))
    image = Image.open(image_path)
    # boxes = objects[image_idx-1]['objects']
    boxes = local_container['pred_boxes']
    box_labels = None
    relationships = local_container['pred_rel_inds']

    # rel = []
    # for n in range(len(relationships[image_idx-1]['relationships'])):
    #     rel.append([relationships[image_idx-1]['relationships'][n]['subject']['object_id'], relationships[image_idx-1]['relationships'][n]['object']['object_id']])
    rel = relationships

    viz_scene_graph(image_idx, image, boxes, box_labels, output_folder, None, rel, pred=True)

    return

def process(inputs, outputs, _ground_truths, _predictions):

    for input, output in zip(inputs, outputs):
        ground_truth = {}
        prediction = {}

        _cpu_device = torch.device("cpu")

        ground_truth['relation_tuple'] = input['relations'].to(_cpu_device) #Relation tupe (obj_id, sub_id, relation label)
        ground_truth['gt_boxes'] = input['instances'].gt_boxes.to(_cpu_device) #Ground truth object boxes
        ground_truth['labels'] = input['instances'].gt_classes.to(_cpu_device) #Ground truth object classes
        ground_truth['rel_pair_idxs'] = input['relations'][:,:2].to(_cpu_device) #Realtion pair index (shape: (num of relations, 2))

        if "instances" in output:
            instances = output["instances"].to(_cpu_device)
            prediction["image_id"] = input["image_id"]
            prediction["instances"] = instances
            prediction['rel_pair_idxs'] = output["rel_pair_idxs"].to(_cpu_device)
            prediction['pred_rel_scores'] = output["pred_rel_scores"].to(_cpu_device)

        ground_truth_cp = copy.deepcopy(ground_truth)
        prediction_cp = copy.deepcopy(prediction)
        del ground_truth 
        del prediction
        _ground_truths.append(ground_truth_cp)
        _predictions.append(prediction_cp)

    del outputs
    del inputs

    return _ground_truths, _predictions

def _mode_from_config(cfg):
    '''
    Estimate mode from configuration
    '''
    if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    return mode

# Adapted from
# https://github.com/danfeiX/scene-graph-TF-release/blob/master/lib/datasets/viz.py

def viz_scene_graph(image_idx, im, rois, labels, output_folder, inds=None, rels=None, preprocess=False, pred=True):
    """
    visualize a scene graph on an image
    """
    # if inds is None:
    #     inds = np.arange(rois.shape[0])
    # viz_rois = rois[inds]
    # viz_labels = labels[inds]
    viz_rois = rois
    viz_labels = labels
    viz_rels = None
    if rels is not None:
        viz_rels = []
        for rel in rels:
            # if rel[0] in inds and rel[1] in inds:
                # sub_idx = np.where(inds == rel[0])[0][0]
                # obj_idx = np.where(inds == rel[1])[0][0]
                # viz_rels.append([sub_idx, obj_idx, rel[2]])
                # viz_rels.append([sub_idx, obj_idx])
            viz_rels.append([rel[0], rel[1]])
        viz_rels = np.array(viz_rels)
    return _viz_scene_graph(image_idx, im, viz_rois, viz_labels, output_folder, viz_rels, preprocess, pred)


def _viz_scene_graph(image_idx, im, rois, labels, output_folder, rels=None, preprocess=False, pred=True):
    # if preprocess:
    #     # transpose dimensions, add back channel means
    #     im = (im.copy() + cfg.PIXEL_MEANS)[:, :, (2, 1, 0)].astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    if rels.size > 0:
        rel_inds = rels[:,:2].ravel().tolist()
    else:
        rel_inds = []
    # draw bounding boxes
    for i, bbox in enumerate(rois):
        # if int(labels[i]) == 0 and i not in rel_inds:
        #     continue
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                        #   bbox['w'], bbox['h'],
                          fill=False,
                          edgecolor='red', linewidth=1)
            )
        # label_str = cfg.ind_to_class[int(labels[i])]
        # ax.text(bbox[0], bbox[1] - 2,
        #         label_str,
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # draw relations
    for i, rel in enumerate(rels):
        # if rel[2] == 0: # ignore bachground
        #     continue
        # sub_box = rois[rel[0]-rois[0]['object_id']]
        # obj_box = rois[rel[1]-rois[0]['object_id']]
        sub_box = rois[rel[0]]
        obj_box = rois[rel[1]]
        # obj_ctr = [obj_box['x']+obj_box['w']/2, obj_box['y']+obj_box['h']/2]
        # sub_ctr = [sub_box['x']+sub_box['w']/2, sub_box['y']+sub_box['h']/2]
        obj_ctr = [(obj_box[0]+obj_box[2])/2, (obj_box[1]+obj_box[3])/2]
        sub_ctr = [(sub_box[0]+sub_box[2])/2, (sub_box[1]+sub_box[3])/2]
        line_ctr = [(sub_ctr[0] + obj_ctr[0]) / 2, (sub_ctr[1] + obj_ctr[1]) / 2]
        # predicate = cfg.ind_to_predicate[int(rel[2])]
        ax.arrow(sub_ctr[0], sub_ctr[1], obj_ctr[0]-sub_ctr[0], obj_ctr[1]-sub_ctr[1], color='green')

        # ax.text(line_ctr[0], line_ctr[1], predicate,
        #         bbox=dict(facecolor='green', alpha=0.5),
        #         fontsize=14, color='white')

    ax.set_title('Scene Graph Visualization', fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_folder, "{}-{}.jpg".format(image_idx, "prediction" if pred else "groundtruth")))

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        # use the last 4 numbers in the job id as the id
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # all ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception:
        default_port = 59482
    
    args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print(args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        # dist_url=args.dist_url,
        dist_url='auto',
        args=(args,),
    )
