import os
import re
import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import numpy as np
from itertools import chain
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader

reg = re.compile(r"(?P<model_architecture>[^_]{1,})_(?P<encoder>[\w-]{1,})*.ckpt")

class SegmentationModel(pl.LightningModule):
    def __init__(
            self, 
            arch, 
            encoder_name, 
            in_channels, 
            out_classes, 
            loss_fn,
            pos_weight,
            **kwargs
        ):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        self.encoder = encoder_name
        self.model_architecture = arch
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = loss_fn
        self.pos_weight = pos_weight

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        
        # logits mask : N , 1, H, W
        # mask : N, 1, H, W
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(pos_weight=self.pos_weight)(logits_mask, mask.squeeze(1))
        
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
#         if stage == "valid":
#             self.log("val_loss", loss)
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
#         loss = [x["loss"] for x in outputs]
#         print(f"STAGE-->{stage}_loss print:{loss}")
        
        loss = torch.stack([x["loss"] for x in outputs]).mean()
    
        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": loss
        }
#         if stage == "valid":
#             loss = torch.stack([x["loss"] for x in outputs]).mean()
#             metrics[f"{stage}_loss"] = loss
            
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _load_weight_model(self, weight_path):
        return self.load_from_checkpoint(
            checkpoint_path=weight_path,
            arch=reg.search(os.path.basename(weight_path)).group("model_architecture"),
            encoder_name=reg.search(os.path.basename(weight_path)).group("encoder"), 
            in_channels=3, 
            out_classes=1,
            loss_fn = self.loss_fn,
            pos_weight = self.pos_weight,
        )

    def _get_result_loss(self, results, gts):
        return self.loss_fn(pos_weight=self.pos_weight, reduction=True)(results, gts)

    def _predict(self, dataloader):
        images = []
        mask_results = []
        gts = []
        labels = []
        ids = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(dataloader):
                pred = self.model(batch["image"])
                images.append(batch["image"])
                mask_results.append(pred)
                gts.append(batch["mask"])
                labels.append(batch["label"])
                ids.append(batch["image_id"])
        images = torch.cat(images)
        mask_results = torch.cat(mask_results).squeeze(1)
        gts = torch.cat(gts)[:len(mask_results)].squeeze(1)
        labels = torch.cat(labels)[:len(mask_results)]
        ids = list(chain(*ids))[:len(mask_results)]
        return images, mask_results, gts, labels, ids, self._get_result_loss(mask_results, gts)[0]

    def anomaly_detection_result(self, result, labels, th_method):
        # labels
        labels = np.array(labels) * 1

        # normalize info save
        infer_norm_max = result.max()
        infer_norm_min = result.min()

        # normalize result
        result = (result - result.min()) / (result.max() - result.min())
        
        # calculate scores
        target = result.reshape(result.shape[0], -1)
        anomaly_scores = np.sort(target, axis=-1)[:,::-1][:,:5].mean(axis=-1)

        # calculate thres hold value 
        if th_method == "no_overdetection":
            th = np.where(labels==0, anomaly_scores, 0).max()
            pred_result = (anomaly_scores > th) * 1.0
        elif th_method == "no_miss":
            th = np.where(labels==1, anomaly_scores, 1).min()
            pred_result = (anomaly_scores >= th) * 1.0
        else:
            th = anomaly_scores.mean()
            pred_result = (anomaly_scores > th) * 1.0

        # result: over detections
        overdetection = (pred_result == 1) & (labels == 0)

        # result: miss
        miss = (pred_result == 0) & (labels == 1)

        # accuracy
        acc = (pred_result == labels).mean()

        return acc, anomaly_scores, miss, overdetection, th, [infer_norm_max, infer_norm_min]
        
    def inference(self, dataloader, weight_path, th_method):
        print(f"inferencing....weight path:{weight_path}", end=".....")
        self.model = self._load_weight_model(weight_path)
        images, result, gts, labels, ids, losses = self._predict(dataloader)
        acc, anomaly_scores, miss, overdetection, mean_th, norm_info = self.anomaly_detection_result(result, labels, th_method)
        print("end")
        return {
            "scheme":os.path.basename(weight_path).split(".")[0],
            "transformed_images":images,
            "preds":result,
            "gts":gts,
            "labels":labels,
            "ids":ids,
            "losses":losses,
            "accuracy":acc,
            "anomaly_scores":anomaly_scores,
            "miss":miss,
            "overdetection":overdetection,
            "thres_hold":mean_th,
            "norm_info":norm_info
        }

 
        