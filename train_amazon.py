
import torch
from AmazonDataset import IbamaInpe25km_Dataset, IbamaDETER1km_Dataset
from pathlib import Path
import numpy as np
# from openstl.utils import create_parser, default_parser
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import geopandas as gpd
from BaseExperiment import BaseExperiment, test_model
from preprocess import reconstruct_time_patches, load_tif_image, load_npy_image
import os
import argparse

parser = argparse.ArgumentParser(description='Amazon Deforestation Prediction')
parser.add_argument('--root_dir', type=str,\
    default='/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/IBAMA_INPE', help='Root directory for the dataset')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for the training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate to be used in the training')
parser.add_argument('--patch_size', type=int, default=128, help='Patch size for the dataset')
parser.add_argument('--overlap', type=float, default=0.15, help='Overlap for the patches')
# parser.add_argument('--window_size', type=int, default=6, help='Window size for the patch series. This includes the predict horizon')
parser.add_argument('--past_window', type=int, default=4, help='Past window for the patch series')
parser.add_argument('--predict_horizon', type=int, default=2, help='Predict horizon for the patch series')
parser.add_argument('--min_def', type=float, default=0.005, help='Minimum deforestation value for the dataset')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
parser.add_argument('--debug', action='store_true', help='Enable or disable debug mode')
parser.add_argument('--normalization', type=str, default='mean_std',  choices=['mean_std', 'minmax'], help='Select Which type of nornalization to be used. Standardization or MinMax Scaler')
parser.add_argument('--pixel_size', type=str, default='1K', help='Pixel size for the images')
parser.add_argument('--N_S', type=int, default=3, help='Number of Encoder Layers')
parser.add_argument('--N_T', type=int, default=3, help='Number of Temporal Layers')
parser.add_argument('--hid_S', type=int, default=32, help='Number of hidden Encoder Layers')
parser.add_argument('--hid_T', type=int, default=128, help='Number of hidden Temporal Layers')
parser.add_argument('--exp_name', type=str, default='exp01', help='Experiment name')
parser.add_argument('--scheduler_step_size', type=int, default=100, help='Number of epochs to change learning rate')
parser.add_argument('--scheduler_decay_factor', type=float, default=0.1, help='Multiplicative factor of learning rate decay.')
parser.add_argument('-optm', '--optmizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optmizer name to be used in the training')
parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momemtum. Only applicable if optmizer is sgd')
parser.add_argument('--focal_gamma', type=float, default=4.5, help='Focal loss gamma hyperparameter')
parser.add_argument('--focal_alpha', type=float, default=None, help='Focal loss alpha hyperparameter')
parser.add_argument('--translator', type=str, default='gsta', help='Translator model type')
parser.add_argument('--dilation_size', type=int, default=-1, help='Filter size to apply to dilation')
parser.add_argument('--prob_aug', type=float, default=0.5, help='probability of apply augmentations')
parser.add_argument('--pool_size', type=int, default=1, help='Pooling size to apply to original image')
parser.add_argument('--binary', type=str, default='both',  choices=['input', 'output', 'both'], help='Select which part of data to be transformed into categories')
parser.add_argument('--aggregate_output', action='store_true', help='Enable or disable aggregate output with Maximum')
args = parser.parse_args()

batch_size = args.batch_size
num_workers = args.num_workers
Debug = args.debug
pixel_size = args.pixel_size

patch_size = args.patch_size
overlap = args.overlap
window_size = args.past_window + args.predict_horizon
min_def = args.min_def
normalization = args.normalization

# root_dir = Path(f'/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/IBAMA_INPE/{pixel_size}')
root_dir = Path(args.root_dir) / pixel_size
print(root_dir)

prob = args.prob_aug
copy_fn = lambda x, **kwargs: x.copy()
transform = A.Compose(
    [
        # TODO: Make Random Rotate work
        A.RandomRotate90(p=prob),
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
    train_set = IbamaDETER1km_Dataset(root_dir=root_dir, normalization=normalization, Debug=Debug, transform=transform,\
        patch_size=patch_size, overlap=overlap, min_def=min_def, window_size=window_size, predict_horizon=args.predict_horizon,\
            dilation_size=args.dilation_size, pool_size=args.pool_size, binary=args.binary,\
                aggregate_output=args.aggregate_output)
    val_data, mask_val_data = train_set.get_validation_set()
    val_set = IbamaDETER1km_Dataset(root_dir=root_dir, normalization=normalization, Debug=Debug, mode='val', patch_size=patch_size,\
        val_data=val_data, mask_val_data=mask_val_data, window_size=window_size, predict_horizon=args.predict_horizon,\
            means=[train_set.mean], stds=[train_set.std], dilation_size=args.dilation_size, binary=args.binary,\
                aggregate_output=args.aggregate_output)
    width = patch_size
    height = patch_size

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
# Exp17 training with categories 2 in input
custom_training_config = {
    'pre_seq_length': args.past_window,
    'aft_seq_length': args.predict_horizon,
    'total_length': window_size,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': args.epochs,
    'lr': args.learning_rate,
    # 'metrics': ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM'],
    'metrics': ['mse', 'mae'],

    'ex_name': args.exp_name, # custom_exp
    'in_shape': [args.past_window, 1, height, width], # T, C, H, W = self.args.in_shape
    'patience': 10,
    'delta': 0.0001,
    'amazon_mask': True,
    'pixel_size': pixel_size,
    'patch_size': patch_size,
    'window_size': window_size,
    'overlap': overlap,
    'loss': 'focal',
    'aux_metrics': ['CM'],
    'normalization': normalization,
    'scheduler_step_size': args.scheduler_step_size,
    'scheduler_decay_factor': args.scheduler_decay_factor,
    'optmizer': args.optmizer,
    'sgd_momentum': args.sgd_momentum,
    'focal_gamma': args.focal_gamma,
    'focal_alpha': args.focal_alpha,
    'dilation_size': args.dilation_size,
    'aggregate_output': args.aggregate_output
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': args.translator, # 'gsta', 'timesformer'
    'N_S': args.N_S,
    'N_T': args.N_T,
    'hid_S': args.hid_S, # default: 64
    'hid_T': args.hid_T, # default: 256,
    'classification': True,
    'num_classes': 1
}

exp = BaseExperiment(dataloader_train, dataloader_val, custom_model_config, custom_training_config)

mean_std = np.stack((train_set.mean, train_set.std))
np.save(os.path.join('work_dirs', custom_training_config['ex_name'], 'mean_std.npy'), mean_std)
print('Mean: ', train_set.mean, 'Std: ', train_set.std)
exp.train()

#TODO: pass test patches to the experiment
if pixel_size == '25K':
    test_data, _ = train_set.get_test_set()
    test_set = IbamaInpe25km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data, means=[train_set.mean, train_set.mean_for, train_set.mean_clouds], stds=[train_set.std, train_set.std_for, train_set.std_clouds])
elif pixel_size == '1K':
    test_data, mask_test_data = train_set.get_test_set()
    # print(len(test_data), len(mask_test_data))
    print(test_data.shape)
    # test_set = IbamaDETER1km_Dataset(root_dir=root_dir, Debug=Debug, mode='val', val_data=test_data,\
    #     mask_val_data=mask_test_data, means=[train_set.mean], stds=[train_set.std])
    test_set = IbamaDETER1km_Dataset(root_dir=root_dir, normalization=normalization, Debug=Debug, mode='val', patch_size=patch_size,\
        val_data=val_data, mask_val_data=mask_val_data, window_size=window_size, predict_horizon=args.predict_horizon,\
            means=[train_set.mean], stds=[train_set.std], dilation_size=args.dilation_size, binary=args.binary,\
                aggregate_output=args.aggregate_output)

dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, pin_memory=True)

preds = test_model(dataloader_test, custom_training_config, custom_model_config)
# preds = np.load('work_dirs/custom_exp1/preds.npy')

work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
print('Reconstructing patches....')
print(preds.shape)
preds_clssf = np.argmax(preds, axis=1)
print(preds_clssf.shape)
preds_reconstructed = reconstruct_time_patches(preds_clssf, patch_size=patch_size, original_img_shape=(2333, 3005), len_patches=1656)
print('Preds reconstructed')
np.save(os.path.join(work_dir_path, 'preds_reconstructed.npy'), preds_reconstructed)
del preds_reconstructed

print('Reconstructing def preds....')
preds_def = preds[:, 0]

def_preds_reconstructed = reconstruct_time_patches(preds_def, patch_size=patch_size, original_img_shape=(2333, 3005), len_patches=1656)
print('Def Preds reconstructed')
np.save(os.path.join(work_dir_path, 'def_preds_reconstructed.npy'), def_preds_reconstructed)
del def_preds_reconstructed
