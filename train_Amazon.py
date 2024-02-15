
import torch
from AmazonDataset import IbamaInpe25km_Dataset
from pathlib import Path
import numpy as np
from openstl.utils import create_parser, default_parser
from osgeo import gdal
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import geopandas as gpd
from BaseExperiment import BaseExperiment

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    for i in range(input_image.shape[0]):
        input_image[i, :, :][amazon_mask == 2.0] = 2
    return input_image

# root_dir = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/Dataset/DETR_Patches')
root_dir = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/IBAMA/25K')

print(root_dir)
batch_size = 32
num_workers = 8
Debug = False

# Calculate weights
# mask_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif'
# train_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/train.tif'
# mask = load_tif_image(mask_path)
# img_train = load_tif_image(train_path)
# img_train = img_train.reshape((3, -1, img_train.shape[1], img_train.shape[2])).max(axis=0)

# mask[mask == 0.0] = 2.0
# mask[mask == 1] = 0.0
# print(mask.shape)

# img_train = apply_legal_amazon_mask(img_train, mask)

# class0_pixels = np.sum(img_train == 0)
# class1_pixels = np.sum(img_train == 1)
# total_pixels = class0_pixels + class1_pixels

# print(f'Class numbers: 0: {class0_pixels} - 1: {class1_pixels} - Total: {total_pixels} - Shape: {img_train.shape[0] * img_train.shape[1] * img_train.shape[2]}')
# print(f'Class percentages: 0: {class0_pixels / total_pixels} - 1: {class1_pixels / total_pixels}')
# # loss_weights = [int(total_pixels / class0_pixels), int(total_pixels / class1_pixels)]
# # loss_weights = [1 - (class0_pixels / total_pixels), 1 - (class1_pixels / total_pixels)]
# # loss_weights = [0.05, 0.95]
# loss_weights = None # Tentar 0.6 e 0.4
# print(f'Loss weights: {loss_weights}')

prob = 0.5
copy_fn = lambda x, **kwargs: x.copy()
transform = A.Compose(
    [
        A.RandomRotate90(p=prob),
        A.OneOf([A.HorizontalFlip(p=prob), A.VerticalFlip(p=prob)]),
        A.Lambda(image=copy_fn, mask=copy_fn),
        ToTensorV2()
    ], is_check_shapes=False
)

# train_set = CustomDataset(root_dir=root_dir / 'Train', Debug=Debug, transform=transform)
# val_set = CustomDataset(root_dir=root_dir / 'Val', Debug=Debug)
train_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, transform=transform)
val_data = train_set.get_validation_set()
val_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=val_data, mean=train_set.mean, std=train_set.std)

print(len(train_set), len(val_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'pre_seq_length': 2,
    'aft_seq_length': 1,
    'total_length': 3,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-4,
    # 'metrics': ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM'],
    'metrics': ['mse', 'mae'],

    'ex_name': 'custom_exp02', # custom_exp
    'dataname': 'custom',
    'in_shape': [2, 1, 98, 136], # T, C, H, W = self.args.in_shape
    'patience': 10,
    'delta': 0.0001,
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 2,
    'N_T': 2,
    'hid_S': 16, # default: 64
    'hid_T': 128 # default: 256
}

exp = BaseExperiment(dataloader_train, dataloader_val, custom_model_config, custom_training_config)

exp.train()


    
# dataloader_test = torch.utils.data.DataLoader(
#     test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

# test_model(dataloader_test, custom_training_config, custom_model_config)
