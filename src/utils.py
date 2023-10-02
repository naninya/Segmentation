import random
import numpy as np
import os
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import cv2
import re

from glob import glob
from tqdm import tqdm
from pytorch_lightning import seed_everything

# fix seeds
def seed_everything_custom(seed: int = 91):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False    
    seed_everything(seed)
    print(f"seed{seed}, applied")

def show_result(configs, target="accuracy"):
    path = os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME,"results.pkl")
    os.path.dirname(path)
    with open(path, "rb") as f:
        results = pickle.load(f)
    if isinstance(results[0][target], torch.Tensor):
        df = pd.DataFrame(results)
        df[target] = df[target].map(lambda x:float(x.mean()))
        df = df.sort_values(target, ascending=True)
    else:
        df = pd.DataFrame(sorted(results, key=lambda x:x[target], reverse=True))
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot()
    sns.barplot(y="scheme", x=target, data=df, orient="h",ax=ax)
    
def get_best_scheme(configs, target="accuracy"):
    path = os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME,"results.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)
    scheme_accuracy_dict = {}
    for result in results:
        if isinstance(result[target], torch.Tensor):
            target_value = float(result[target].mean())
            reverse_flag = False
        else:
            target_value = float(result[target])
            reverse_flag = True
        scheme_accuracy_dict[result["scheme"]] = target_value
    return sorted(scheme_accuracy_dict.items(), key=lambda x:x[1], reverse=reverse_flag)[0][0]

def save_result_images(configs):
    path = os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME,"results.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)
    save_image_dir = os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME)
    for result in results:
        for transformed_image, gt, pred, _id, miss in zip(result["transformed_images"], result["gts"], result["preds"], result["ids"], result["miss"]):
            pred = (torch.sigmoid(pred) > 0.5) * 1
            transformed_image = transformed_image.permute(1,2,0).contiguous().numpy().astype(np.uint8)
            gt_3d = np.stack([
                np.zeros_like(gt),
                gt*255,
                np.zeros_like(gt)
            ], -1).astype(np.uint8)
            pred_3dgt_3d = np.stack([
                pred*255,
                np.zeros_like(gt),
                np.zeros_like(gt)
            ], -1).astype(np.uint8)
            
            result_img = cv2.addWeighted(transformed_image, 1, gt_3d, 0.5, 1)
            result_img = cv2.addWeighted(result_img, 1, pred_3dgt_3d, 0.5, 1)
            os.makedirs(os.path.join(save_image_dir, "segmentation_image", result["scheme"]), exist_ok=True)
            file_name = os.path.join(save_image_dir, "segmentation_image", result["scheme"], _id)
            cv2.imwrite(f"{file_name}.png", result_img)

def show_result_images(configs, scheme, show_image_num=30):
    paths = glob(os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME,"segmentation_image", scheme) + "/*.png")
    show_image_num = min(show_image_num, len(paths))
    col_num = int(np.ceil(show_image_num / 4))
    row_num = 4
    fig = plt.figure(figsize=(18, show_image_num))
    for index, path in enumerate(paths):
        ax = fig.add_subplot(col_num, row_num, index+1)
        result_image = cv2.imread(path)
        ax.imshow(result_image)
        ax.set_title(os.path.basename(path))
        if index == show_image_num - 1:
            break

def show_predict_mask(configs, scheme):
    with open(os.path.join(configs.RESULT_ROOT, configs.PROJECT_NAME,"results.pkl"), "rb") as f:
        results = pickle.load(f)
    vis_th = configs.VISUALIZE_TH
    row_num = len(results[0]["labels"])
    fig = plt.figure(figsize=(18, 8 * row_num))
    ax = fig.subplots(row_num, 4)
    for i, result in enumerate(results):
        correct_num = 0
        if result["scheme"] == scheme:
            th = result["thres_hold"]
            for index, (
                transformed_image, 
                gt, 
                label, 
                pred, 
                _id, 
                loss, 
                score
                ) in enumerate(zip(
                    result["transformed_images"],
                    result["gts"],  
                    result["labels"], 
                    result["preds"], 
                    result["ids"], 
                    result["losses"], 
                    result["anomaly_scores"])
                ):
                
                # target image
                image = transformed_image.permute(1,2,0).numpy().astype(np.uint8)
                ax[index][0].imshow(image)
                ax[index][0].set_title(f"ID:{_id}, anomaly:{label}")
                
                # pred mask
                pred_mask = (torch.sigmoid(pred) > 0.5)*1.0
                ax[index][1].imshow(pred_mask)
                ax[index][1].set_title(f"score:{score:.2f}")
                
                # pred mask with th
                pred_mask_th = (torch.sigmoid(pred) > vis_th)*1.0
                ax[index][2].imshow(pred_mask_th)
                ax[index][2].set_title(f"vis_th:{vis_th}")
                
                # gt
                ax[index][3].imshow(gt)
                ax[index][3].set_title(f"Anomaly:{label}, Pred:{score > th}")
            break
