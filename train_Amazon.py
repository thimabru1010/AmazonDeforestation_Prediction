
import torch
from AmazonDataset import CustomDataset
from GiovanniDataset import GiovanniDataset
from pathlib import Path
import numpy as np
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser
from osgeo import gdal
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import geopandas as gpd
import GiovConfig as config
from prep_giov_data import prep4dataset

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    for i in range(input_image.shape[0]):
        input_image[i, :, :][amazon_mask == 2.0] = 2
    return input_image

root_dir = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/Dataset/DETR_Patches')

print(root_dir)
batch_size = 32
num_workers = 8
Debug = False
data_prep = 'giov'

if data_prep == 'giov':
    loss_weights = None
    
    train_data, val_data, patches_sample_train, patches_sample_val, frames_idx, county_data, counties_time_grid, \
        precip_time_grid, tpi_array, scores_time_grid, night_time_grid = prep4dataset(config)
    
    counties_time_grid = None # county_defor
    tpi_array = None #topological 
    
    county_data = None
    precip_time_grid = None
    scores_time_grid = None
    night_time_grid = None
    
    train_set = GiovanniDataset(
        train_data, 
        patches_sample_train, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=None
    )
    
    val_set = GiovanniDataset(
        val_data, #previously = test_data
        patches_sample_val, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=None
    )

else:
    # Calculate weights
    mask_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif'
    train_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/train.tif'
    mask = load_tif_image(mask_path)
    img_train = load_tif_image(train_path)
    img_train = img_train.reshape((3, -1, img_train.shape[1], img_train.shape[2])).max(axis=0)

    # mask[mask == 0.0] = 2.0
    # mask[mask == 1] = 0.0
    # print(mask.shape)

    # img_train = apply_legal_amazon_mask(img_train, mask)

    class0_pixels = np.sum(img_train == 0)
    class1_pixels = np.sum(img_train == 1)
    total_pixels = class0_pixels + class1_pixels

    print(f'Class numbers: 0: {class0_pixels} - 1: {class1_pixels} - Total: {total_pixels} - Shape: {img_train.shape[0] * img_train.shape[1] * img_train.shape[2]}')
    print(f'Class percentages: 0: {class0_pixels / total_pixels} - 1: {class1_pixels / total_pixels}')
    # loss_weights = [int(total_pixels / class0_pixels), int(total_pixels / class1_pixels)]
    # loss_weights = [1 - (class0_pixels / total_pixels), 1 - (class1_pixels / total_pixels)]
    # loss_weights = [0.05, 0.95]
    loss_weights = None # Tentar 0.6 e 0.4
    print(f'Loss weights: {loss_weights}')

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

    train_set = CustomDataset(root_dir=root_dir / 'Train', Debug=Debug, transform=transform)
    val_set = CustomDataset(root_dir=root_dir / 'Val', Debug=Debug)

print(len(train_set), len(val_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
# dataloader_test = torch.utils.data.DataLoader(
#     test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'pre_seq_length': 4,
    'aft_seq_length': 1,
    'total_length': 5,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-3,
    'final_div_factor': 	10000.0,   
    'metrics': ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM'],

    'ex_name': 'custom_exp06_PopPast', # custom_exp
    'dataname': 'custom',
    'in_shape': [4, 1, 64, 64], # T, C, H, W = self.args.in_shape
    'loss_weights': loss_weights,
    'weight_decay': 1e-4
}
#     'early_stop_epoch': 10,
#     'warmup_epoch': 0, #default = 0
#     'sched': 'step',
#     'decay_epoch': 10,
#     'decay_rate': 0.8,
#     'weight_decay': 1e-2,
#     'resume_from': None,
#     'auto_resume': False,
#     'test_time': False,
#     'seed': 5,
#     'loss': 'focal'
# }

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 64, # default: 64
    'hid_T': 256 # default: 256
}
    
#TODO: Save the config to be used in test script
args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, None))

#TODO: Botar weights na Cross Entropy para balancear os labels
print('>'*35 + ' training ' + '<'*35)
exp.train()

# print('>'*35 + ' testing  ' + '<'*35)
# exp.test(classify=True)