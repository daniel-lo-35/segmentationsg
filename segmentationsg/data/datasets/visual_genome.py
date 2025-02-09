import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random
import os
import torch
import numpy as np
import pickle
import yaml
from detectron2.config import get_cfg
from detectron2.structures import Instances, Boxes, pairwise_iou, BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging

class VisualGenomeTrainData:
    """
    Register data for Visual Genome training
    """
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        if split == 'train':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TRAIN_MASKS
        elif split == 'val':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.VAL_MASKS
        else:
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TEST_MASKS
        self.mask_exists = os.path.isfile(self.mask_location)
        self.clamped = True if "clamped" in self.mask_location else ""
        self.clipped = cfg.DATASETS.VISUAL_GENOME.CLIPPED
        self.precompute = False if (self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS or self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP) else True
        
        # self._process_data()
        self.dataset_dicts = self._fetch_data_dict()
        self.register_dataset()
        try:
            self.get_statistics()
        except:
            pass
        
    def register_dataset(self):
        """
        Register datasets to use with Detectron2
        """
        DatasetCatalog.register('VG_{}'.format(self.split), lambda: self.dataset_dicts)
        
        #Get labels
        self.mapping_dictionary = json.load(open(self.cfg.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY, 'r'))
        self.idx_to_classes = sorted(self.mapping_dictionary['label_to_idx'], key=lambda k: self.mapping_dictionary['label_to_idx'][k])
        self.idx_to_predicates = sorted(self.mapping_dictionary['predicate_to_idx'], key=lambda k: self.mapping_dictionary['predicate_to_idx'][k])
        self.idx_to_attributes = sorted(self.mapping_dictionary['attribute_to_idx'], key=lambda k: self.mapping_dictionary['attribute_to_idx'][k])
        MetadataCatalog.get('VG_{}'.format(self.split)).set(thing_classes=self.idx_to_classes, predicate_classes=self.idx_to_predicates, attribute_classes=self.idx_to_attributes)
    
    def _fetch_data_dict(self):
        """
        Load data in detectron format
        """
        fileName = "tmp/visual_genome_{}_data_{}{}{}{}{}.pkl".format(self.split, 'masks' if self.mask_exists else '', '_oi' if 'oi' in self.mask_location else '', "_clamped" if self.clamped else "", "_precomp" if self.precompute else "", "_clipped" if self.clipped else "")
        if os.path.isfile(fileName):
            #If data has been processed earlier, load that to save time
            print("loading cached file: ", fileName)
            with open(fileName, 'rb') as inputFile:
                dataset_dicts = pickle.load(inputFile)
        else:
            #Process data
            os.makedirs('tmp', exist_ok=True)
            dataset_dicts = self._process_data()
            #TODO: this can cause problems, if it is excecuted in a distributed setup
            print("creating cache file: ", fileName)
            with open(fileName, 'wb') as inputFile:
                pickle.dump(dataset_dicts, inputFile)
        return dataset_dicts
            
    def _process_data(self):
        self.VG_attribute_h5 = h5py.File(self.cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5, 'r')
        
        # Remove corrupted images
        image_data = json.load(open(self.cfg.DATASETS.VISUAL_GENOME.IMAGE_DATA, 'r'))
        # self.corrupted_ims = ['1592', '1722', '4616', '4617']
        self.corrupted_ims = []
        self.image_data = []
        for i, img in enumerate(image_data):
            if str(img['image_id']) in self.corrupted_ims:
                continue
            self.image_data.append(img)
        # assert(len(self.image_data) == 108073)
        self.masks = None
        if self.mask_location != "":
            try:
                with open(self.mask_location, 'rb') as f:
                    self.masks = pickle.load(f)
            except:
                pass
        dataset_dicts = self._load_graphs()
        return dataset_dicts

    def get_statistics(self, eps=1e-3, bbox_overlap=True):
        num_object_classes = len(MetadataCatalog.get('VG_{}'.format(self.split)).thing_classes) + 1
        num_relation_classes = len(MetadataCatalog.get('VG_{}'.format(self.split)).predicate_classes) + 1
        
        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes), dtype=np.int64)
        bg_matrix = np.zeros((num_object_classes, num_object_classes), dtype=np.int64)
        for idx, data in enumerate(self.dataset_dicts):
            gt_relations = data['relations']
            gt_classes = np.array([x['category_id'] for x in data['annotations']])
            gt_boxes = np.array([x['bbox'] for x in data['annotations']])
            for (o1, o2), rel in zip(gt_classes[gt_relations[:,:2]], gt_relations[:,2]):
                fg_matrix[o1, o2, rel] += 1
            for (o1, o2) in gt_classes[np.array(box_filter(gt_boxes, must_overlap=bbox_overlap), dtype=int)]:
                bg_matrix[o1, o2] += 1
        bg_matrix += 1
        fg_matrix[:, :, -1] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.idx_to_classes + ['__background__'],
            'rel_classes': self.idx_to_predicates + ['__background__'],
            'att_classes': self.idx_to_attributes,
        }
        MetadataCatalog.get('VG_{}'.format(self.split)).set(statistics=result)
        return result

    def _load_graphs(self):
        """
        Parse examples and create dictionaries
        """
        data_split = self.VG_attribute_h5['split'][:]
        split_flag = 2 if self.split == 'test' else 0
        if self.split == 'val':
            split_flag = 1
        split_mask = data_split == split_flag
        
        #Filter images without bounding boxes
        split_mask &= self.VG_attribute_h5['img_to_first_box'][:] >= 0
        if self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS:
            split_mask &= self.VG_attribute_h5['img_to_first_rel'][:] >= 0
        image_index = np.where(split_mask)[0]
        
        # if self.split == 'val':
        #     image_index = image_index[:self.cfg.DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES]
        # elif self.split == 'train':
        #     image_index = image_index[self.cfg.DATASETS.VISUAL_GENOME.NUMBER_OF_VALIDATION_IMAGES:]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[image_index] = True
        
        # Get box information
        all_labels = self.VG_attribute_h5['labels'][:, 0]
        try:
            all_attributes = self.VG_attribute_h5['attributes'][:, :]
        except:
            all_attributes = np.zeros((self.VG_attribute_h5['labels'].shape[0], 0))
        all_boxes = self.VG_attribute_h5['boxes_{}'.format(self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE)][:]  # cx,cy,w,h
        assert np.all(all_boxes[:, :2] >= 0)  # sanity check
        assert np.all(all_boxes[:, 2:] > 0)  # no empty box
        
        # Convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
        
        first_box_index = self.VG_attribute_h5['img_to_first_box'][split_mask]
        last_box_index = self.VG_attribute_h5['img_to_last_box'][split_mask]
        first_relation_index = self.VG_attribute_h5['img_to_first_rel'][split_mask]
        last_relation_index = self.VG_attribute_h5['img_to_last_rel'][split_mask]

        #Load relation labels
        all_relations = self.VG_attribute_h5['relationships'][:]
        all_relation_predicates = self.VG_attribute_h5['predicates'][:, 0]
        
        image_indexer = np.arange(len(self.image_data))[split_mask]
        # Iterate over images
        dataset_dicts = []
        for idx, _ in enumerate(image_index):
            record = {}
            #Get image metadata
            image_data = self.image_data[image_indexer[idx]]
            record['file_name'] = os.path.join(self.cfg.DATASETS.VISUAL_GENOME.IMAGES, '{}.jpg'.format(image_data['image_id']))
            record['image_id'] = image_data['image_id']
            record['height'] = image_data['height']
            record['width'] = image_data['width']
    
            #Get annotations
            boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
            gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
            gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]

            if first_relation_index[idx] > -1:
                predicates = all_relation_predicates[first_relation_index[idx]:last_relation_index[idx] + 1]
                objects = all_relations[first_relation_index[idx]:last_relation_index[idx] + 1] - first_box_index[idx]
                predicates = predicates - 1
                relations = np.column_stack((objects, predicates))
            else:
                assert not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS
                relations = np.zeros((0, 3), dtype=np.int32)
            
            if self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP and self.split == 'train':
                # Remove boxes that don't overlap
                boxes_list = Boxes(boxes)
                ious = pairwise_iou(boxes_list, boxes_list)
                relation_boxes_ious = ious[relations[:,0], relations[:,1]]
                iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
                if iou_indexes.size > 0:
                    relations = relations[iou_indexes]
                else:
                    #Ignore image
                    continue
            #Get masks if possible
            if self.masks is not None:
                try:
                    gt_masks = self.masks[image_data['image_id']]
                except:
                    print (image_data['image_id'])
            record['relations'] = relations
            objects = []
            # if len(boxes) != len(gt_masks):
            mask_idx = 0
            for obj_idx in range(len(boxes)):
                resized_box = boxes[obj_idx] / self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE * max(record['height'], record['width'])
                obj = {
                      "bbox": resized_box.tolist(),
                      "bbox_mode": BoxMode.XYXY_ABS,
                      "category_id": gt_classes[obj_idx] - 1,
                      "attribute": gt_attributes[obj_idx],           
                }
                if self.masks is not None:
                    if gt_masks['empty_index'][obj_idx]:
                        refined_poly = []
                        for poly_idx, poly in enumerate(gt_masks['polygons'][mask_idx]):
                            if len(poly) >= 6:
                                refined_poly.append(poly)
                        obj["segmentation"] = refined_poly
                        mask_idx += 1
                    else:
                        obj["segmentation"] = []
                    if len(obj["segmentation"]) > 0:
                        objects.append(obj)
                else:
                    objects.append(obj)
            record['annotations'] = objects
            dataset_dicts.append(record)
        
        return dataset_dicts

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter