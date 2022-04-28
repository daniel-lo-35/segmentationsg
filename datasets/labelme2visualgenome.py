""" 
Written by https://github.com/daniel-lo-35

This script transforms a folder of labelme .json files into the following Visual Genome format: 
1. images (uncompressed folder)
2. image_data.json
3. objects.json
4. relationships.json

Other files that need to be produced manually:
5. object_alias.txt (An empty file is ok, as no alias in rebar dataset)
6. predicate_alias.txt (An empty file is ok, as no alias in rebar dataset)
7. object_list.txt (I type "intersection" in it)
8. predicate_list.txt (I type "spacing" in it)

It can then be transformed to the .h5 format that many scene graph repo uses. More info:
- http://visualgenome.org/api/v0/api_readme
- https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md
- https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md

This currently only support "spacing" and "intersection" label.
TODO: Add support of overlap and hook angle
"""

import argparse
import os
import glob
import imgviz
import sys
import json

import numpy as np
from scipy.spatial.distance import cdist

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


# ---------------------------------------------
#  ALGORITHM PART
#  https://github.com/CMMAi/AIiS/blob/main/aicore/engines/rgbd/utils.py
# ---------------------------------------------

# def find_centroid_by_mask(masks):
#     """
#     Find centroids from predicted masks.
#     Args:
#         masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
#             of mask. H and W is the height and width of the mask, respectively
#     Returns:
#         a torch tensor
#     """
#     centroids = []
#     for mask in np.asarray(masks.cpu()):
#         polys = Mask(mask).polygons()
#         pts = [pt for contour in polys.points for pt in contour]
#         if len(pts) >= 3:
#             M = cv2.moments(np.array(pts))
#             centroids.append([M['m10'] / M["m00"], M['m01'] / M["m00"]])
#     return torch.tensor(centroids)

# def find_endpoint_by_mask(masks, return_linemasks=True):
#     """
#     Find the two farest points as the endpoints of the each mask.
#     Args:
#         masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
#             of mask. H and W is the height and width of the mask, respectively
#         return_linemasks (bool): if return the masks with endpoints.
#     Returns:
#         torch tensors
#     """
#     endpoints = []
#     linemask_indices = []
#     for m, mask in enumerate(np.asarray(masks.cpu())):
#         polys = Mask(mask).polygons()
#         pts = [pt for contour in polys.points for pt in contour]
#         if len(pts) >= 2:
#             dist_matrix = cdist(pts, pts, metric='euclidean')
#             i, j = np.where(dist_matrix==dist_matrix.max())[0][:2]
#             endpoints.append([pts[i], pts[j]])
#             linemask_indices.append(m)
#     if return_linemasks:
#         return torch.tensor(endpoints), masks[linemask_indices]
#     return torch.tensor(endpoints)

def find_nearest_link(juncs, lines, line_masks=None,
        max_e2j_dist=30, max_e2e_dist=50, path_thred=0.5, e2e_on=True, return_index=True):
    """
    Find the links between junctions and lines.
    Args:
        juncs (torch.tensor): a tensor of junctions of shape (N, 2), where N is the number
            of junction. Each junction is represented by a point (X, Y).
        lines (torch.tensor): a tensor of lines of shape (N, 2, 2), where N is the number
            of line. Each line is represented by two points.
        line_masks (Optional[torch.tensor]): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        max_e2j_dist (int): the maximun tolerance distance between endpoints and junctions.
        max_e2e_dist (int): the maximun tolerance distance between endpoints and enpoints.
        path_thred (Optional[float]): a float between [0, 1] that filters out links with path confindence under path_thred.
        return_index (bool): if return the indices of connected junction.
    Returns:
        a torch tensor
    """
        
    links = []
    for line in lines:
        # E2J link prediction
        e2j_dist_matrix = cdist(line, juncs, metric='euclidean')
        i, j = e2j_dist_matrix.argsort(1)[:,0]
        if i != j and e2j_dist_matrix[0, i] < max_e2j_dist and e2j_dist_matrix[1, j] < max_e2j_dist:
            if return_index:
                links.append([i, j])
            else:
                links.append(juncs[[i, j]].numpy().tolist())
        
        # E2E link prediction
        # I remove it as it won't be used
    
    if return_index:
        # return juncs, torch.tensor(links)
        return juncs, links
        
    # return torch.tensor(links)
    return links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    print("Creating dataset:", args.output_dir)

    label_files = glob.glob(os.path.join(args.input_dir, "*.json"))

    image_data = [
        # image_id, url, width, height, coco_id, flickr_id
    ]
    objects = [
        # image_id, objects
    ]
    relationships = [
        # image_id, relationships
    ]

    image_data_json = os.path.join(args.output_dir, "image_data.json")
    objects_json = os.path.join(args.output_dir, "objects.json")
    relationships_json = os.path.join(args.output_dir, "relationships.json")

    current_object_len = 0
    current_relationship_len = 0

    for image_id, filename in enumerate(label_files, start=1):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = os.path.splitext(os.path.basename(filename))[0]
        out_img_file = os.path.join(args.output_dir, "images", str(image_id) + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        image_data.append(
            dict(
                image_id = image_id,
                url = None,
                width = img.shape[1],
                height = img.shape[0],
                coco_id = None,
                flickr_id = None,
            )
        )

        objects.append(
            dict(
                image_id = image_id,
                objects = [
                    # object_id, x, y, w, h, name, synsets
                ],
            )
        )

        relationships.append(
            dict(
                image_id = image_id,
                relationships = [
                    # relationship_id, predicate, synsets, subject, object
                ]
            )
        )

        intersection_list = []
        spacing_list = []
        try:
            current_object_len += len(objects[-2]["objects"])
            current_relationship_len += len(relationships[-2]["relationships"])
        except:
            pass
        intersection_object_list = []

        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            # group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            # area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist() # bottom left width height
            [bbox_x, bbox_y, bbox_w, bbox_h] = [int(bbox) for bbox in bbox] # bottom left width height

            if label == "intersection":
                intersection_object_list.append(
                    dict(
                        object_id = current_object_len + len(intersection_object_list),
                        x = bbox_x,
                        y = bbox_y,
                        w = bbox_w,
                        h = bbox_h,
                        name = "intersection",
                        synsets = ["intersection.n.01"],
                    )
                )
                objects[-1]["objects"].append(intersection_object_list[-1])
                intersection_list.append([bbox_x + bbox_w/2, bbox_y + bbox_h/2]) # centroid of intersection

            elif label == "spacing":
                dist_matrix = cdist(points, points, metric='euclidean')
                max_matrix = np.where(dist_matrix==dist_matrix.max())[0]
                if len(max_matrix) < 3:
                    i, j = max_matrix[:2]
                else:
                    i, j = 0, 2
                spacing_list.append([points[i], points[j]]) # Two edges (points) of spacing

        _, links = find_nearest_link(intersection_list, spacing_list, e2e_on=False)

        for link in links:
            relationships[-1]["relationships"].append(
                dict(
                    relationship_id = current_relationship_len + len(relationships[-1]["relationships"]),
                    predicate = "spacing",
                    synsets = ["spacing.n.01"],
                    subject = intersection_object_list[link[0]],
                    object = intersection_object_list[link[1]],
                )
            )
            relationships[-1]["relationships"].append(
                dict(
                    relationship_id = current_relationship_len + len(relationships[-1]["relationships"]),
                    predicate = "spacing",
                    synsets = ["spacing.n.01"],
                    subject = intersection_object_list[link[1]],
                    object = intersection_object_list[link[0]],
                )
            ) # I add 2 times because this relationship is directional. TODO: change to non-directional

    with open(image_data_json, "w") as f:
        json.dump(image_data, f)

    with open(objects_json, "w") as f:
        json.dump(objects, f)

    with open(relationships_json, "w") as f:
        json.dump(relationships, f)


if __name__ == '__main__':
    main()
