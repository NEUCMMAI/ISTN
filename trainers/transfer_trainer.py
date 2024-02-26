'''
Author: Kang Yang
Date: 2023-02-24 14:55:00
LastEditors: Kang Yang
LastEditTime: 2023-08-02 20:35:01
FilePath: /project-b/trainers/transfer_trainer.py
Description: 
Copyright (c) 2023 by Kang Yang, All Rights Reserved. 
'''
import os
from logging import getLogger

import torch
from recbole.utils import get_trainer


class TransferTrainer(object):
    def __init__(self, model_class, main_config):
        self.model_class = model_class
        self.main_config = main_config
        self.logger = getLogger()
        
    def source_init(self, source_config, source_train_data, source_vaild_data, source_dataset, model_flag):
        self.source_config = source_config
        self.source_train_data = source_train_data
        self.source_vaild_data = source_vaild_data
        self.source_dataset = source_dataset
        self.model_flag = model_flag
        
        if model_flag == 1:
            self.source_model = self.model_class(self.source_config, self.source_train_data, self.source_vaild_data, self.source_dataset)
        else:
            self.source_model = self.model_class(self.source_config, self.source_dataset)
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.main_config['gpu_id'])
        self.source_model = self.source_model.to(self.main_config['device'])
        if not self.main_config['only_output_test']:
            self.logger.info(self.source_model)
        self.s_trainer = get_trainer(self.main_config['MODEL_TYPE'], self.main_config['model'])(self.source_config, self.source_model)
        
    def target_init(self, target_config, target_train_data, target_vaild_data, target_dataset, state=None):
        self.target_config = target_config
        self.target_train_data = target_train_data
        self.target_vaild_data = target_vaild_data
        self.target_dataset = target_dataset
        try:
            self.target_model = self.model_class(self.target_config, self.target_train_data, self.target_vaild_data, self.target_dataset, only_eval_target=self.main_config['only_eval_target'])
        except:
            self.target_model = self.model_class(self.target_config, self.target_dataset)
        if not self.main_config['only_output_test']:
            self.logger.info(self.target_model)
        if self.main_config['source_weight_path']:
            if self.main_config['excluded_weights'] is not None:
                for param_name in self.main_config['excluded_weights']:
                    # print(state["state_dict"].keys())
                    try:
                        del state["state_dict"][param_name]
                    except:
                        print('delete {} fail'.format(param_name))
                if not self.main_config['only_eval_target']:
                    for name, param in self.target_model.named_parameters():
                        if name not in self.main_config['excluded_weights']:
                            param.requires_grad=False
                self.target_model.load_state_dict(state["state_dict"], strict=False)
            else:
                self.target_model.load_state_dict(state["state_dict"])
            self.target_model = self.target_model.to(self.main_config['device'])
        else:
            self.target_model = self.target_model.to(self.main_config['device'])
        
        
        self.t_trainer = get_trainer(self.main_config['MODEL_TYPE'], self.main_config['model'])(self.target_config, self.target_model)
        
    def source_fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        best_valid_score, best_valid_result = self.s_trainer.fit(
            train_data, valid_data, saved=saved, show_progress=show_progress
        )
        return best_valid_score, best_valid_result
    
    def traget_fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        best_valid_score, best_valid_result = self.t_trainer.fit(
            train_data, valid_data, saved=saved, show_progress=show_progress
        )
        return best_valid_score, best_valid_result
    
    @torch.no_grad()
    def evaluate(self, eval_data, show_progress=False):
        if not self.main_config['only_source']:
            result = self.t_trainer.evaluate(eval_data, load_best_model=False, show_progress=show_progress)
        else:
            result = self.s_trainer.evaluate(eval_data, load_best_model=False, show_progress=show_progress)
        return result
