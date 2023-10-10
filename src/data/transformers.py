import albumentations
import albumentations.pytorch
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class Transformers:
    def __init__(self, image_size):
        self._build(image_size)

    def _build(self, image_size):
        self.train_transformer = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            # albumentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
        self.valid_transformer = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            # albumentations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
    
    def get_trasformer(self, mode):
        if self.mode == "train":
            return self.train_transformer
        elif self.mode == "valid":
            return self.valid_transformer
        elif self.mode == "test":
            return self.valid_transformer