import tensorflow as tf

from pycocotools import mask, coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from skimage import measure
from PIL import Image

from pathlib import Path
import numpy as np

import base64
from io import BytesIO
import json

def binary_mask2coco_annot(binary_mask, image_id, category_id, annot_id, iscrowd=0):
    """
    Args:
        binary_mask (np.array): 
        image_id (int):
        category_id (int):
        annot_id (int):
        iscrowd (int):

    Returns:
        dict: annotation dict
    """
    fortran_binary_mask = np.asfortranarray(binary_mask)
    encoded = mask.encode(fortran_binary_mask)
    area = mask.area(encoded)
    bbx = mask.toBbox(encoded)
    contours = measure.find_contours(binary_mask, 0.5)

    annotation = {
        "segmentation": [],
        "area": area.tolist(),
        "iscrowd": iscrowd,
        "image_id": image_id,
        "bbox": bbx.tolist(),
        "category_id": category_id,
        "id": annot_id
    }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    return annotation

def generate_coco_json_from_masks(
    list_binary_masks: np.array, 
    list_image_id: list, 
    extension="",
    info="",
    licenses=""):
    """[summary]

    Args:
        list_binary_masks (np.array): [description]
        list_image_id (list):
        extension (str, optional): [description]. Defaults to "".

    Raises:
        ValueError: [description]
    """    
    # One annotation per image, each annotation has multiple segments (1/contour)
    annotations = []
    for i, img in enumerate(list_binary_masks):
        
        annot = binary_mask2coco_annot(
            binary_mask=img, 
            image_id= list_image_id[i], 
            category_id=1, 
            annot_id=i,  # 
            iscrowd=0
        )
        annotations.append(annot)

    # Initialize output coco json
    coco_json = {
        "info": info,
        "licenses": licenses,
        "annotations": annotations,
        "images": [im for im in coco_dict["images"] if im["id"] not in img_id_wt_annot],
        "categories": coco_dict["categories"]
    }
    
    pass

def generate_coco_json_from_tfRecords(tfrecords_path: str):
    """[summary]

    Args:
        tfrecords_path (str): [description]

    Returns:
        [type]: [description]
    """
    annotations = []
    images = []
    for i, rec in enumerate(tf.data.TFRecordDataset([str(tfrecords_path)])):
        # TFRecords ingestion
        example_bytes = rec.numpy()
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
   
        image = {
            "coco_url": "",
            "date_captured":"",
            "file_name": example.features.feature["image/filename"].bytes_list.value[0].decode(),
            "flickr_url":"",
            "height": int(example.features.feature["image/height"].int64_list.value[0]),
            "id": int(example.features.feature["image/source_id"].bytes_list.value[0]),
            "license":0,
            "width": int(example.features.feature["image/width"].int64_list.value[0]),
        }
        images.append(image)
    
        # Binary Mask 
        # TODO: doesn't work
        tfr_mask = example.features.feature["image/object/mask"].bytes_list.value[0]
        m = Image.open(BytesIO(tfr_mask))
        numpy_binary_mask = np.asarray(m)
        
        annot = binary_mask2coco_annot(
            binary_mask=numpy_binary_mask, 
            image_id=image["id"], 
            category_id=1, 
            annot_id=i,  # 
            iscrowd=0
        )
        
        annotations.append(annot)
        
    return {
        "info": "",
        "licenses": "",
        "annotations": annotations,
        "images": images,
        "categories": [
            {
                "color": "#b00cd4",
                "id": 1,
                "keypoint_colors": [],
                "metadata": {},
                "name":"Dégâts",
                "supercategory":""
            }
        ]
    }

def img2mask(coco, img_id, cat_id):
    anns_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)
    annotations = coco.loadAnns(ids=anns_ids)
    return np.sum([coco.annToMask(a) for a in annotations], axis=0) > 0

def IOU(component1, component2):
    component1 = np.array(component1, dtype=bool)
    component2 = np.array(component2, dtype=bool)

    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR

    return overlap.sum()/float(union.sum())

def compute_IoU_from_coco_json(coco_gt, coco_dt, coco_type='segm'):
    cocoGt = COCO(coco_gt)
    cocoDt = COCO(coco_dt)  # needs to have a `score` field for each annotations
    
    list_img_ids_Gt = cocoGt.getImgIds()
    list_img_ids_Dt = cocoDt.getImgIds()

    if set(list_img_ids_Dt) != set(list_img_ids_Gt):
        raise ValueError(
            "Input COCO JSON files have different images." \
            "Found {} non-overlapping images".format(
                len(set(list_img_ids_Dt) - set(list_img_ids_Gt))
            )
        )
    
    ious = {}
    for img_id in list(set(list_img_ids_Dt)):
        mask_Gt = img2mask(cocoGt, img_id, cat_id=1)
        mask_Dt = img2mask(cocoDt, img_id, cat_id=1)

        ious[img_id] = IOU(mask_Gt, mask_Dt)

    mean_iou = np.mean(
        [iou for _, iou in ious.items()]
    )
    # For additional metrics
    # coco_eval = COCOeval(cocoGt, cocoDt, iouType=coco_type)

    # Only use if score is available for each annotation
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    
    return mean_iou


if __name__ == "__main__":

    tfrecords = Path("./sample_data/tfrecords")
    train_tfr = tfrecords / "train.record"
    test_tfr = tfrecords / "test.record"

    coco = Path("./sample_data/coco_json")
    test_coco = coco / "val_car_damages.json"

    d = generate_coco_json_from_tfRecords(str(train_tfr))
    with open(coco / "train_car_damages_2.json", "w") as f:
        json.dump(d, f)

    mean_iou = compute_IoU_from_coco_json(
        str(coco / "train_car_damages.json"),
        str(coco / "train_car_damages.json")
    )
    
