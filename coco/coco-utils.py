import tensorflow as tf

from pycocotools import mask, coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from skimage import measure
from sklearn.model_selection import train_test_split

from PIL import Image

from pathlib import Path
import numpy as np

import base64
from io import BytesIO
import json
import random

import funcy

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

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

def remove_img_wt_annotations(coco_dict):
    """Remove from a COCO dictionary (COCO JSON loaded as a Python dict) images that
    have no corresponding annotations.


    Args:
        coco_dict (dict): COCO JSON loaded as a Python dict, we call it COCO dict

    Returns:
        dict: COCO dict
    """
    img_id_from_annot = set([ann["image_id"] for ann in coco_dict["annotations"]])
    img_id = set([ann["id"] for ann in coco_dict["images"]])

    img_id_wt_annot = img_id - img_id_from_annot

    return {
        "info": coco_dict["info"],
        "licenses": coco_dict["licenses"],
        "annotations": coco_dict["annotations"],
        "images": [im for im in coco_dict["images"] if im["id"] not in img_id_wt_annot],
        "categories": coco_dict["categories"]
    }

def generate_small_batch(coco_dict, nb_img=50, seed=42):
    """Generate a COCO dict by keeping only `nb_img` images
    and their associated annotations. Handy to create small test datasets.

    Args:
        coco_dict ([type]): [description]
        nb_img (int, optional): [description]. Defaults to 50.
        seed (int, optional): [description]. Defaults to 42.

    Returns:
        dict: COCO dict
    """
    random.seed(42)
    img_id_to_keep = set(
        random.sample(
            [ann["id"] for ann in coco_dict["images"]],
            nb_img
        )
    )
    return {
        "info": coco_dict["info"],
        "licenses": coco_dict["licenses"],
        "annotations": [c for c in coco_dict["annotations"] if c["image_id"] in img_id_to_keep],
        "images": [im for im in coco_dict["images"] if im["id"] in img_id_to_keep],
        "categories": coco_dict["categories"]
    }

def merge_coco_json(json1, json2, write=False, output_file_path="./merged_datasets.json", overwrite_ids=True):
    """
    
        overwrite_ids: bool, whether to overwrite image ids. 
            If true: image ids are reassigned with their order of appearance under the "images" key
            If false: image ids are kept as is, but a check is performed to confirm there is no overlap
    """
    json1, json2 = Path(json1), Path(json2)
    
    with open(str(json1), "r") as j1:
        data1 = json.load(j1)
    with open(str(json2), "r") as j2:
        data2 = json.load(j2)
        
    def reconcile_fields(f1, f2):
        if f1 is None and f2 is None:
            f = ""
        elif f1 is not None and f2 is not None:
            print(f"[WAR] Keeping first dataset field {f1}")
            f = f1
        else:
            f = f1 if f1 is not None else f2
        return f
    
    info1 = data1['info'] if 'info' in data1 else None
    info2 = data2['info'] if 'info' in data2 else None

    licenses1 = data1['licenses'] if 'licenses' in data1 else None
    licenses2 = data2['licenses'] if 'licenses' in data2 else None
         
    images_id_old2new_1 = {im['id']:it for it, im in enumerate(data1['images'])}
    images_id_old2new_2 = {im['id']:it+len(data1['images']) for it, im in enumerate(data2['images'])}
    
    annot_id_old2new_1 = {annot['id']:it for it, annot in enumerate(data1['annotations'])}
    annot_id_old2new_2 = {annot['id']:it+len(data1['annotations']) for it, annot in enumerate(data2['annotations'])}
    
    if not overwrite_ids and len(set(images_id_old2new_1.keys()).intersection(images_id_old2new_2.keys())) != 0:
        raise ValueError("Image IDs overlap, you need to select overwrite_ids=True")
        
    if overwrite_ids:
    
        for i, _ in enumerate(data1['images']):
            data1['images'][i]['id'] = i
            
        for j, _ in enumerate(data2['images']):
            data2['images'][j]['id'] = len(data1['images']) + j
            
        for i, annot in enumerate(data1['annotations']):
            data1['annotations'][i]['image_id'] = images_id_old2new_1[annot['image_id']]
            
        for j, annot in enumerate(data2['annotations']):
            data2['annotations'][j]['image_id'] = images_id_old2new_2[annot['image_id']]
            
        for i, annot in enumerate(data1['annotations']):
            data1['annotations'][i]['id'] = annot_id_old2new_1[annot['id']]
            
        for j, annot in enumerate(data2['annotations']):
            data2['annotations'][j]['id'] = annot_id_old2new_2[annot['id']]
            
            
            
    merged_data = {
        'info': reconcile_fields(info1, info2), # present if coming from Ahmed's code
        'licenses':reconcile_fields(licenses1, licenses2),  # present if coming from Ahmed's code
        'images': [im for im in data1['images']] + [im for im in data2['images']],
        'annotations': [annot for annot in data1['annotations']] + [annot for annot in data2['annotations']],
        'categories': data1['categories']  # unchanged in our use-case
    }
    
    if write:
        with open(output_file_path, "w") as f:
            json.dump(merged_data, f)
        
    return merged_data

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def train_test_split_coco_json(input_json, claims_mapping, save_path="./", filename="car_damages"):
    with open(input_json, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)
        unique_claim_ids = list(set(list(claims_mapping.values())))
        
        #x, y = train_test_split(images, train_size=0.85, random_state=42)
        x, y = train_test_split(unique_claim_ids, train_size=0.85, random_state=42)
        
        with open("train_claim_ids.txt", "w") as f:
            for claim_id in x:
                f.write(claim_id+"\n")
        with open("valid_claim_ids.txt", "w") as f:
            for claim_id in y:
                f.write(claim_id+"\n")
        
        x_images = [k for k, v in claims_mapping.items() for val in x if v == val]
        y_images = [k for k, v in claims_mapping.items() for val in y if v == val]
        train_img_objs = [val for v in x_images for val in images if v == val['file_name']]
        val_img_objs = [val for v in y_images for val in images if v == val['file_name']]
        save_coco(
            os.path.join(
                save_path,
                f"train_{filename}.json"
            ),
            info, licenses, train_img_objs, 
            filter_annotations(annotations, train_img_objs), 
            categories
        )
        save_coco(
            os.path.join(
                save_path,
                f"val_{filename}.json"
            ), 
            info, licenses, val_img_objs, 
            filter_annotations(annotations, val_img_objs), 
            categories)

        print("Saved {} entries in {} and {} in {}".format(len(train_img_objs), './car_damages_train.json', len(val_img_objs), './car_damages_val.json'))

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
