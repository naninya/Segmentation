import os
import torch
import pickle

from torch.utils.data import DataLoader


from data.dataset import SegmentationDataset
from torch.utils.data import DataLoader
from data.utils import load_transform

from model.models import SegmentationModel
from model.losses import BceDiceLoss, DiceLoss
from utils import seed_everything_custom

class Tester:
    def __init__(
        self,
        configs
    ):
        self.configs = configs
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

    def run(self, path):
        seed_everything_custom()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Define transformer & test
        test_dataset = SegmentationDataset(
            data_root = path,
            annotation_path=None,
            transform=load_transform(self.configs.IMAGE_SIZE).valid_transformer,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=self.configs.BATCH_SIZE, shuffle=False, drop_last=False)

        
        results = []
        for encoder in self.configs.ENCODERS:
            for model_architecture in self.configs.ARCHITECHURES:
                torch.cuda.empty_cache()
                print(f"processing...{model_architecture}, {encoder}")
                try:
                    weight_path = f"{self.configs.MODEL_ROOT}/{self.configs.PROJECT_NAME}/{model_architecture}_{encoder}.ckpt"
                    
                    if os.path.isfile(weight_path):
                        model = SegmentationModel(model_architecture, encoder, in_channels=3, out_classes=1, loss_fn=self.loss_fn, pos_weight=self.configs.POS_WEIGHT)
                        result = model.inference(test_dataloader, weight_path, self.configs.THRES_HOLD_METHOD)
                        results.append(result)
                    else:
                        raise Exception("No weight file")
                    print(f"{model_architecture}, {encoder} --> finished")
                
                except Exception as e:
                    print(e)
                    print(f"{model_architecture}, {encoder} --> Error")
        
        file_path = os.path.join(self.configs.RESULT_ROOT, self.configs.PROJECT_NAME, "results.pkl")
        
        with open(file_path, "wb") as f:
            pickle.dump(results, f)
        # return results