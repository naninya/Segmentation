import os
import torch
from glob import glob

from pytorch_lightning import Trainer as TR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


from data.dataset import SegmentationDataset
from torch.utils.data import DataLoader
from data.utils import load_transform

from model.models import SegmentationModel
from model.losses import BceDiceLoss, DiceLoss
from utils import seed_everything_custom

class Trainer:
    def __init__(
        self,
        configs
    ):
        self.configs = configs
        self._build()

    def _build(self):
        # Define transformer & test
        seed_everything_custom()
        train_dataset = SegmentationDataset(
            data_root = os.path.join(self.configs.DATA_ROOT, "train"),
            annotation_path=None,
            transform=load_transform(self.configs.IMAGE_SIZE).train_transformer,
        )
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.configs.BATCH_SIZE, shuffle=False, drop_last=False)

        
        valid_dataset = SegmentationDataset(
            data_root = os.path.join(self.configs.DATA_ROOT, "valid"),
            annotation_path=None,
            transform=load_transform(self.configs.IMAGE_SIZE).train_transformer,
        )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.configs.BATCH_SIZE, shuffle=False, drop_last=False)
        
        self.enable_progress_bar = self.configs.ENABLE_PROGRESS_BAR
        self.model_root = os.path.join(self.configs.MODEL_ROOT, self.configs.PROJECT_NAME)
        self.result_root = os.path.join(self.configs.RESULT_ROOT, self.configs.PROJECT_NAME)
        os.makedirs(self.model_root, exist_ok=True)
        os.makedirs(self.result_root, exist_ok=True)
        self.loss_fn = self._get_loss_fn()

    def _get_loss_fn(self):
        losses = [
            BceDiceLoss,
            DiceLoss,
        ]
        losses_dict = {l.__name__.lower(): l for l in losses}
        try:
            return losses_dict[self.configs.LOSS]
        except KeyError:
            raise KeyError(
                "Wrong loss name `{}`. Available options are: {}".format(
                    self.configs.LOSS,
                    list(losses_dict.keys()),
                )
            )
        
    def _fit(self, model):        
        early_stop_callback = EarlyStopping(
            monitor='valid_loss',
            min_delta=0.00,
            patience=self.configs.PATIENCE,
            verbose=False,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="valid_loss",
            mode="min",
            dirpath=self.model_root,
            filename=f"{model.model_architecture}_{model.encoder}",
        )
        trainer = TR(
            devices=1,
            accelerator="gpu",
            max_epochs=self.configs.EPOCHS,
            enable_progress_bar=self.enable_progress_bar,
            callbacks=[
                early_stop_callback,
                checkpoint_callback
            ],
            logger=False,
        )
        
        trainer.fit(
            model, 
            train_dataloaders=self.train_dataloader, 
            val_dataloaders=self.valid_dataloader,
        )

    def fit_all_scheme(self):
        for encoder in self.configs.ENCODERS:
            for model_architecture in self.configs.ARCHITECHURES:
                torch.cuda.empty_cache()
                print(f"processing...{model_architecture}, {encoder}")
                try:
                    try:
                        model = SegmentationModel(model_architecture, encoder, in_channels=3, out_classes=1, loss_fn=self.loss_fn, pos_weight=self.configs.POS_WEIGHT)
                        for exist_path in glob(f"{self.model_root}/{model_architecture}_{encoder}*.ckpt"):
                            print(f"weight file exist..delete{exist_path}")
                            os.remove(exist_path)
                    except Exception as e:
                        print(e)
                        print("model load error")
                        continue
                    
                    self._fit(model)
                    print(f"{model_architecture}, {encoder} --> finished")
                except Exception as e:
                    print(e)
                    print(f"{model_architecture}, {encoder} --> Error")