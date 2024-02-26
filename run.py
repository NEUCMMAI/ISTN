'''
Author: Kang Yang
Date: 2023-02-16 11:36:26
LastEditors: Kang Yang
LastEditTime: 2023-07-31 15:21:54
FilePath: /project-b/run.py
Description: Project B run file
Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
'''

import importlib
from logging import getLogger
import os

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from trainers.transfer_trainer import TransferTrainer
from utils import TransferConfig, get_args, sync_config


def run_model(model=None, model_type='general_recommender', main_config = None , source_dataset=None, target_dataset=None, saved=True):
    
    main_config = main_config

    model_flag = 0

    try:
        model_class =  getattr(importlib.import_module('models'), model)
        model_flag = 1
    except AttributeError:
        model_class =  getattr(importlib.import_module('recbole.model.'+ model_type), model)
        
    source_config = 'configs/'+ model +'/source-'+ source_dataset +'.yaml'
    target_config = 'configs/'+ model +'/target-'+ target_dataset +'.yaml'

    source_config = Config(model=model_class, dataset=source_dataset, config_file_list=[source_config], config_dict=main_config.file_config_dict)
    target_config = Config(model=model_class, dataset=target_dataset, config_file_list=[target_config], config_dict=main_config.file_config_dict)
    
    # source_config['gpu_id'] = main_config['gpu_id']
    # target_config['gpu_id'] = main_config['gpu_id']
    
    # source_config['use_gpu'] = main_config['use_gpu']
    # target_config['use_gpu'] = main_config['use_gpu']
    
    # source_config['device'] = main_config['device']
    # target_config['device'] = main_config['device']
    
    # sync_config(main_config, source_config)
    # sync_config(main_config, target_config)
    
    init_seed(main_config['seed'], main_config['reproducibility'])

    init_logger(main_config)
    logger = getLogger()

    if not main_config['only_output_test']:
        logger.info(main_config)
    
    trainer = TransferTrainer(model_class, main_config)

    if not main_config['source_weight_path'] and not main_config['random_weight']:
        source_dataset = create_dataset(source_config)
        if not main_config['only_output_test']:
            logger.info(source_dataset)

        source_train_data, source_valid_data, source_test_data = data_preparation(source_config, source_dataset)
        
        trainer.source_init(source_config, source_train_data, source_valid_data, source_dataset, model_flag = model_flag)
        source_best_valid_score, source_best_valid_result = trainer.source_fit(
            source_train_data, source_valid_data, saved=saved, show_progress=main_config['show_progress']
        )
    if not main_config['only_source']:
        target_dataset = create_dataset(target_config)
        if not main_config['only_output_test']:
            logger.info(target_dataset)
        target_train_data, target_valid_data, target_test_data = data_preparation(target_config, target_dataset)
        
        if not main_config['source_weight_path'] and not main_config['random_weight']:
            trainer.target_init(target_config, target_train_data, target_valid_data, target_dataset)

        elif main_config['random_weight']:
            trainer.target_init(target_config, target_train_data, target_valid_data, target_dataset)
        else:
            state = torch.load(main_config['source_weight_path'], map_location='cuda:0')
            trainer.target_init(target_config, target_train_data, target_valid_data, target_dataset, state)
        
        if not main_config['only_eval_target']:
            traget_best_valid_score, traget_best_valid_result = trainer.traget_fit(
                target_train_data, target_valid_data, saved=saved, show_progress=main_config['show_progress']
            )
    else:
        # target_test_data = source_test_data
        pass

    test_result = trainer.evaluate(target_test_data, show_progress=main_config['show_progress'])

    logger.info('{}-{}->{}-test result: {}'.format(main_config['model'], main_config['source_dataset'], main_config['target_dataset'], test_result))
    
if __name__ == '__main__':
    args = get_args()
    config_file_list = ['configs/' + args.model +'/main.yaml']
    main_config = TransferConfig(config_file_list=config_file_list, config_dict=None)
    use_gpu = main_config['use_gpu']
    gpu_id = main_config['gpu_id']
    if use_gpu:
        with torch.cuda.device(int(gpu_id)):
            run_model(model=args.model, main_config = main_config, source_dataset=args.source_dataset, target_dataset=args.target_dataset, model_type='sequential_recommender')
    else:
        run_model(model=args.model, main_config = main_config, source_dataset=args.source_dataset, target_dataset=args.target_dataset, model_type='sequential_recommender')
