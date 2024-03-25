
import torch
from AmazonDataset import IbamaInpe25km_Dataset, IbamaDETER1km_Dataset
from pathlib import Path
import numpy as np
from openstl.utils import create_parser, default_parser
from osgeo import gdal
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import geopandas as gpd
from BaseExperiment import BaseExperiment, test_model
from preprocess import reconstruct_time_patches
import os

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    for i in range(input_image.shape[0]):
        input_image[i, :, :][amazon_mask == 2.0] = 2
    return input_image

batch_size = 16
num_workers = 8
Debug = False
pixel_size = '1K'

patch_size = 128
overlap = 0.15
window_size = 6
min_def = 0.005
normalize = True

root_dir = Path(f'/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/IBAMA_INPE/{pixel_size}')
print(root_dir)

prob = 0.5
copy_fn = lambda x, **kwargs: x.copy()
transform = A.Compose(
    [
        # TODO: Make Random Rotate work
        #A.RandomRotate90(p=prob),
        A.OneOf([A.HorizontalFlip(p=prob), A.VerticalFlip(p=prob)]),
        A.Lambda(image=copy_fn, mask=copy_fn),
        ToTensorV2()
    ], is_check_shapes=False
)

if pixel_size == '25K':
    train_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, transform=transform)
    val_data = train_set.get_validation_set()
    val_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=val_data,\
        means=[train_set.mean, train_set.mean_for, train_set.mean_clouds], stds=[train_set.std, train_set.std_for, train_set.std_clouds])
    width = 136
    height = 98
elif pixel_size == '1K':
    train_set = IbamaDETER1km_Dataset(root_dir=root_dir, normalize=normalize, Debug=Debug, transform=transform,\
        patch_size=patch_size, overlap=overlap, min_def=min_def, window_size=window_size)
    val_data, mask_val_data = train_set.get_validation_set()
    val_set = IbamaDETER1km_Dataset(root_dir=root_dir, normalize=normalize, Debug=Debug, mode='val', patch_size=patch_size,\
        val_data=val_data, mask_val_data=mask_val_data, means=[train_set.mean], stds=[train_set.std])
    width = patch_size
    height = patch_size

print(len(train_set), len(val_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'pre_seq_length': 4,
    'aft_seq_length': 2,
    'total_length': 6,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 1e-4,
    # 'metrics': ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM'],
    'metrics': ['mse', 'mae'],

    'ex_name': 'custom_exp2', # custom_exp
    'dataname': 'custom',
    'in_shape': [4, 1, height, width], # T, C, H, W = self.args.in_shape
    'patience': 10,
    'delta': 0.0001,
    'amazon_mask': True,
    'pixel_size': pixel_size,
    'patch_size': patch_size,
    'window_size': window_size,
    'overlap': overlap,
    'loss': 'focal',
    'aux_metrics': ['f1_score0', 'f1_score1', 'CM'],
    'normalize': normalize
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 3,
    'N_T': 4,
    'hid_S': 32, # default: 64
    'hid_T': 128, # default: 256,
    'classification': True,
    'num_classes': 2
}

exp = BaseExperiment(dataloader_train, dataloader_val, custom_model_config, custom_training_config)

mean_std = np.stack((train_set.mean, train_set.std))
np.save(os.path.join('work_dirs', custom_training_config['ex_name'], 'mean_std.npy'), mean_std)

exp.train()

#TODO: pass test patches to the experiment
if pixel_size == '25K':
    test_data, _ = train_set.get_test_set()
    test_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data, means=[train_set.mean, train_set.mean_for, train_set.mean_clouds], stds=[train_set.std, train_set.std_for, train_set.std_clouds])
elif pixel_size == '1K':
    test_data, mask_test_data = train_set.get_test_set()
    print(len(test_data), len(mask_test_data))
    print(test_data.shape, mask_test_data.shape)
    test_set = IbamaDETER1km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data,\
        mask_val_data=mask_test_data, means=[train_set.mean], stds=[train_set.std])

dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, pin_memory=True)

preds = test_model(dataloader_test, custom_training_config, custom_model_config)
# preds = np.load('work_dirs/custom_exp1/preds.npy')

work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
print('Reconstructing patches....')
print(preds.shape)
preds_clssf = np.argmax(preds, axis=1)
print(preds_clssf.shape)
preds_reconstructed = reconstruct_time_patches(preds_clssf, patch_size=patch_size, time_idx=43, original_img_shape=(2333, 3005), len_patches=1656)
print('Preds reconstructed')
np.save(os.path.join(work_dir_path, 'preds_reconstructed.npy'), preds_reconstructed)
del preds_reconstructed

print('Reconstructing def preds....')
preds_def = preds[:, 1]

def_preds_reconstructed = reconstruct_time_patches(preds_def, patch_size=patch_size, time_idx=43, original_img_shape=(2333, 3005), len_patches=1656)
print('Def Preds reconstructed')
np.save(os.path.join(work_dir_path, 'def_preds_reconstructed.npy'), def_preds_reconstructed)
del def_preds_reconstructed
