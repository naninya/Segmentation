import os
import torch
import json
import numpy as np
import cv2
from glob import glob
from PIL import Image
from .utils import json2anot_file

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        annotation_path=None, 
        transform=None
    ):
        image_paths = glob(f"{data_root}/images/*.*")
        mask_paths = glob(f"{data_root}/annotations/*.*")
        if all([
            annotation_path is not None,
            len(mask_paths)==0
        ]):
            print("parsing..")
            # json file to annotations file
            mask_paths = json2anot_file(annotation_path, data_root)
        if any([
            mask_paths is None,
            len(mask_paths) == 0
        ]):
            print("there is no mask, force masks all 0")
        # non train target : mask.sum()==0
        image_paths_filtered = []
        mask_paths_filtered = []
        for image_path, mask_path in zip(sorted(image_paths), sorted(mask_paths)):
            if os.path.basename(mask_path).split(".")[-1] == "npy":
                mask = np.load(mask_path)
            else:
                mask = cv2.imread(mask_path)
            if mask.sum() != 0:
                image_paths_filtered.append(image_path)
                mask_paths_filtered.append(mask_path)
        else:
            image_paths_filtered = image_paths
            mask_paths_filtered = mask_paths
        self.image_paths = image_paths_filtered
        self.mask_paths = mask_paths_filtered
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        image_id = os.path.basename(image_path).split(".")[0]
        
        mask_path = self.mask_paths[idx]
        if os.path.basename(mask_path).split(".")[-1] == "npy":
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path)
            mask = mask.mean(axis=-1)
        mask = self._preprocess_mask(mask)
        
        sample = dict(
            image=image, 
            mask=mask, 
            label=mask.sum() > 0,
            image_path=image_path, 
            mask_path=mask_path,
            image_id=image_id
        )
        if self.transform is not None:
            sample = self.transform(**sample)
        sample["image"] = sample["image"].to(torch.float32)
        sample["mask"] = torch.where(sample["mask"] > 0, 1, 0)
        if os.path.basename(mask_path).split(".")[-1] == "npy":
            sample["mask"] = sample["mask"].permute(2, 0, 1)
        else:
            sample["mask"] = sample["mask"].unsqueeze(0)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.uint8)
        mask[mask!=0] = 1
        return mask