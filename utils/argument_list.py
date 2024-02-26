'''
Author: Kang Yang
Date: 2023-02-24 14:59:51
LastEditors: Kang Yang
LastEditTime: 2023-02-24 14:59:58
FilePath: /project-b/utils/argument_list.py
Description: 
Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
'''
# yapf: disable

general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'reproducibility',
    'state',
    'data_path',
    'show_progress',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'training_neg_sample_num',
    'training_neg_sample_distribution',
    'eval_step', 'stopping_step',
    'checkpoint_dir'
]

evaluation_arguments = [
    'eval_setting',
    'group_by_user',
    'split_ratio', 'leave_one_num',
    'real_time_process',
    'metrics', 'topk', 'valid_metric',
    'eval_batch_size'
]
