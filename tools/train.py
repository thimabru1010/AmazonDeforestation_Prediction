# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

# from .. import openstl
from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    batch_size = 8
    # custom_training_config = {
    # 'pre_seq_length': None,
    # 'aft_seq_length': None,
    # 'total_length': None,
    # 'batch_size': batch_size,
    # 'val_batch_size': batch_size,
    # 'epoch': 100,
    # 'lr': 1e-3,   
    # 'metrics': ['acc', 'Recall', 'Precision', 'f1_score', 'CM'],

    # 'ex_name': 'mmnist_simvp_gsta', # custom_exp
    # 'dataname': 'mmnist',
    # 'in_shape': [10, 1, 64, 64], # T, C, H, W = self.args.in_shape
    # 'loss_weights': None,
    # 'early_stop_epoch': 10,
    # 'warmup_epoch': 0, #default = 0
    # 'sched': 'step',
    # 'decay_epoch': 10,
    # 'decay_rate': 0.8,
    # 'weight_decay': 1e-2,
    # 'resume_from': None,
    # 'auto_resume': False,
    # 'test_time': False,
    # 'seed': 5,
    # 'loss': 'focal'
    # }


    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
        
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'batch_size', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        print(config)
        config['loss'] = None
        config['batch_size'] = 8
        config['val_batch_size'] = 8
        config['num_workers'] = 8
        config['metrics'] = ['mse', 'mae', 'acc', 'Recall', 'Precision', 'f1_score', 'CM']
        config['lr'] = 1e-3
        config['sched'] = 'step'
        config['decay_epoch'] = 5
        config['decay_rate'] = 0.8
        config['weight_decay'] = 1e-2
        # custom_training_config = { 'in_shape': [5, 1, 64, 64]}
        # # update the training config
        # config.update(custom_training_config)
        
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)
