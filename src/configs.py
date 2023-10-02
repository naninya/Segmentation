import os

class Configs:
    PROJECT_NAME = "bottle"
    # seed value
    SEED = 91
    DATA_ROOT = "../data/bottle"
    RESULT_ROOT = "../results"
    # train batch size
    BATCH_SIZE = 2
    # processing image size
    IMAGE_SIZE = 256
    # training weight -> object
    POS_WEIGHT = 10
    # annotations file path(json)
    ANNOTATION_FILE_PATH = None
    
    ENABLE_PROGRESS_BAR = False
    # architectures
    ARCHITECHURES = [
        "unet",
        "unetplusplus",
        "manet",
        "linknet",
        "fpn",
        "pspnet",
        "deeplabv3",
        "deeplabv3plus",
        "pan",
    ]
    # encoders
    ENCODERS = [
        "resnext50_32x4d",
        "resnext101_32x48d",
        "efficientnet-b7",
        "inceptionresnetv2",
        "resnet34",
        "mobilenet_v2",
        "se_resnext101_32x4d",
        "timm-regnety_160",
        "timm-skresnext50_32x4d", 
    ]
    PATIENCE = 5
    EPOCHS = 200
    
    VISUALIZE_TH = 0.9
    LOSS = "diceloss"
    THRES_HOLD_METHOD = "no_miss"
