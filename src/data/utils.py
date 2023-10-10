import os
import re
import shutil
import cv2
from glob import glob

from .transformers import Transformers

def load_transform(image_size):
    transform = Transformers(image_size)
    return transform

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_object_class(region):
    # no multiclass object
    for index, val in enumerate(region["region_attributes"].values()):
        if len(val) > 0:
            return index
    else:
        return None

def _parse_json(path, image_dir, object_class_num):
    json_dict = _load_json(path)
    info = json_dict["_via_img_metadata"]
    output = {}
    for _, anot_info in info.items():
        file_name = anot_info["filename"]
        regions = anot_info["regions"]
        file_id = file_name.split(".")[0]
        
        img = None
        for image_path in glob(f"{image_dir}/*.*"):
            if os.path.basename(image_path).split(".")[0] == file_id:
                image = cv2.imread(image_path)
                break
        else:
            # un annotated labels.. should fix bug for json file
            for image_path in glob(f"{image_dir}/*.*"):
                if os.path.basename(image_path).split(".")[0][:5] == file_id[:5]:
                    image = cv2.imread(image_path)
                    break
            
        # input image size
        mask = np.zeros((*image.shape[:2], object_class_num))
        for region in regions:
            mask_mono = np.zeros(image.shape[:2])
            xs = region["shape_attributes"]["all_points_x"]
            ys = region["shape_attributes"]["all_points_y"]
            object_class = get_object_class(region)
            contours = []
            for x, y in zip(xs, ys):
                contours.append([x,y])
            contours = np.array(contours)
#             print(file_name, mask.shape, object_class)
            mask[:,:,object_class] = cv2.fillPoly(mask_mono, pts=[contours], color=255)
        output[file_id] = mask
    return output

def json2anot_file(json_path, data_root, object_class_num):
    target_dir = os.path.join(data_root, "annotations")
    mask_paths = []
    annotations = _parse_json(json_path, os.path.join(data_root, "images"), object_class_num=object_class_num)
    os.makedirs(target_dir, exist_ok=True)
    for file_id, mask in annotations.items():
        mask_path = f"{os.path.join(target_dir, file_id)}.npy"
        # print(f"mask_path:{mask_path}, {mask.astype(np.uint8).shape}")
        np.save(mask_path, mask)
        mask_paths.append(mask_path)
    return mask_paths

    
def reform_mvtec_dir_tree_for_segmentation(
        source_dir="../data/mvtec_original", 
        target_dir="../data"
    ):
    for _source in glob(f"{source_dir}/*"):
        category = os.path.basename(_source)
        for index, (image_path, gt_path) in enumerate(zip(
            sorted(glob(f"{source_dir}/{category}/test/**/*.*")),
            sorted(glob(f"{source_dir}/{category}/ground_truth/**/*.*"))
        )):  
            target = "train"
            if index % 10 == 0:
                target = "valid"
            # gt
            error_name = gt_path.split("/")[-2]
            _name = os.path.basename(gt_path).split("_mask")[0]
            mask_name = f"{_name}.png"
            mask_path = os.path.join(target_dir, f"{category}", target, "annotations", f"anomaly_{error_name}_{mask_name}")
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            shutil.copy(gt_path, mask_path)
            
            # image
            if "good" in image_path:
                des_path = os.path.join(target_dir, f"{category}", target, "images", f"normal_{os.path.basename(image_path)}")
            else:
                error_name = image_path.split("/")[-2]
                des_path = os.path.join(target_dir, f"{category}", target, "images", f"anomaly_{error_name}_{os.path.basename(image_path)}")
            os.makedirs(os.path.dirname(des_path), exist_ok=True)
            shutil.copy(image_path, des_path)