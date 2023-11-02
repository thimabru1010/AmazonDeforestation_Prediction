
import torch
from AmazonDataset import CustomDataset
from pathlib import Path
import numpy as np
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

    
root_dir = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/Dataset/DETR_Patches')

print(root_dir)
batch_size = 32
Debug = False

train_set = CustomDataset(root_dir=root_dir / 'Train', Debug=Debug)
val_set = CustomDataset(root_dir=root_dir / 'Val', Debug=Debug)
test_set = CustomDataset(root_dir=root_dir / 'Test', Debug=Debug, Test=True)

print(len(train_set), len(val_set), len(test_set))

dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

#! Test dataset class
# for i, (data, labels) in enumerate(train_set):
#     print(i, data.shape, labels.shape)
#     print(np.histogram(data.numpy(), bins=np.arange(0, 4, step=1)), np.histogram(labels.numpy(), bins=np.arange(0, 4, step=1)))
#     print()
    
custom_training_config = {
    'pre_seq_length': 4,
    'aft_seq_length': 1,
    'total_length': 5,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 100,
    'lr': 0.001,   
    'metrics': ['acc'],

    'ex_name': 'custom_exp3', # custom_exp
    'dataname': 'custom',
    'in_shape': [4, 3, 64, 64], # T, C, H, W = self.args.in_shape
}

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
    'hid_S': 64,
    'hid_T': 256
}
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

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test), nclasses=3)

#TODO: Botar weights na Cross Entropy para balancear os labels
# print('>'*35 + ' training ' + '<'*35)
# exp.train()

print('>'*35 + ' testing  ' + '<'*35)
exp.test(classify=True)
