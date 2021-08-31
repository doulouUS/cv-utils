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

def generate_coco_json_from_manifest(input_manifest, write=False):
    """Transform a manifest file to COCO Json format.
    The manifest file (obtained from SageMaker groundtruth) contains annotations with 
    a single class (TODO here `categories` is hardcoded, to be parameterized).
    However, there can be several instances per class. They are all squashed
    together. The output COCO JSON file is therefore made for semantic segmentation
    with one annotation per image.

    Args:
        input_manifest (str): path to manifest file
        write (bool, optional): whether to write the dict to json. Defaults to False.

    Returns:
        dict: coco json as a python dict
    """
    date_now = datetime.today().strftime("%Y-%m-%d")
    year = datetime.today().strftime("%Y")

    # TODO parameterize
    licenses = [{"name": "", "id": 0, "url": ""}]
    info = {
        "contributor": "AXA GETD - AXA Morocco",
        "date_created": date_now,
        "description": "Car Damage Dataset",
        "url": "",
        "version": 0.1,
        "year": "2021",
    }
    # TODO parameterize
    category = {
        "color":"#b00cd4",
        "id":1,
        "keypoint_colors":[],
        "metadata":{},
        "name":"Dégâts",
        "supercategory":""
    }

    with open(input_manifest, "r") as f:
        lines = f.readlines()

    images = []
    for i, sample in enumerate(lines):
        sample = json.loads(sample)
        annot_data = sample.get('damage').get('annotationsFromAllWorkers')[0].get('annotationData')
        content  = json.loads(annot_data.get('content'))

        sample['height'] = content.get('annotatedResult').get('inputImageProperties').get('height')
        sample['width'] = content.get('annotatedResult').get('inputImageProperties').get('width') 
        try:
            image = {
                "coco_url": "",
                "date_captured": "",
                "flickr_url": "",
                "license": 0,
                "id": i,
                "file_name": Path(sample['source-ref']).name,
                "height": content.get('annotatedResult').get('inputImageProperties').get('height'),
                "width": content.get('annotatedResult').get('inputImageProperties').get('width'),
                "angle_object": content['angle_object'],
                 "brand": content['brand'],
                 "car_type": content['car_type'],
                 "hubcaps": content['hubcaps'],
                 "rim_tire": content['rim_tire']
            }
        except:
            # TODO this is a very specific need from how we handled our data
            # it should probably be abstracted to make the code more general to
            # any cases
            # TODO license and other fields are hardcoded 
            # For curated images, additional data is not in `content` but in the first level of the json
            image = {
                "coco_url": "",
                "date_captured": "",
                "flickr_url": "",
                "license": 0,
                "id": i,
                "file_name": Path(sample['source-ref']).name,
                "height": content.get('annotatedResult').get('inputImageProperties').get('height'),
                "width": content.get('annotatedResult').get('inputImageProperties').get('width'),
                "angle_object": sample['angle_object'],
                 "brand": sample['brand'],
                 "car_type": sample['car_type'],
                 "hubcaps": sample['hubcaps'],
                 "rim_tire": sample['rim_tire']
            }
        images.append(image)

    annotations = []
    for img_id, sample in enumerate(lines):
        sample = json.loads(sample)
        annot_data = sample.get('damage').get('annotationsFromAllWorkers')[0].get('annotationData')
        content  = json.loads(annot_data.get('content'))

        mask_bytes = content.get('annotatedResult').get('labeledImage').get('pngImageData')
        mask = Image.open(io.BytesIO(base64.b64decode(mask_bytes)))
        # Convert to grayscale to squash all different instances into one
        mask_image = mask.convert('L')
        binary_mask = np.asarray(mask_image) > 0

        # One annotation per image, so image_id and annot_id are the same
        annot = binary_mask2coco_annot(
            binary_mask,
            img_id,
            1,
            img_id
        )
        annotations.append(annot)
    coco_json = {
        "info": info,
        "licenses": licenses,
        "images": images,
        "annotations": annotations,
        "categories": category
    }
    if write:
        with open("coco-json-from-manifest.json", "w") as f:
            json.dumps(coco_json, f)
    return coco_json

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

    d = generate_coco_json_from_tfRecords(str(test_tfr))
    with open(coco / "val_car_damages_2.json", "w") as f:
        json.dump(d, f)

    mean_iou = compute_IoU_from_coco_json(
        str(coco / "val_car_damages.json"),
        str(coco / "val_car_damages_2.json")
    )
    print("")
