import random
import unittest

import pytorch_lightning as pl
import torch
import numpy as np

import sys
  
# setting path
sys.path.append('../')
from discrete_bottleneck.datamodule.grid_datamodule import GridDataModule
from discrete_bottleneck.datamodule.grid_datasets import ConstantGridDataset
from discrete_bottleneck.datamodule.grid_datasets import VariableElementCountGridDataset
from pytorch_lightning import seed_everything


class TestDataset(unittest.TestCase):
    
#     import hydra
#     from omegaconf import OmegaConf
#     configs_path = "configs"
#     config_name = "config.yaml"
#     with hydra.initialize(config_path=configs_path):
#         config = hydra.compose(config_name=config_name,
#                                overrides=[f"data_dir=../",
#                                           f"work_dir=../",
#                                           f"datamodule=simple_grid"])
#         datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)
#         datamodule.setup()
    
    def setUp(self):
        
        self.seeds = [1, 123, 543, 358]
        self.num_samples_train = 50
        self.num_samples_val = 30
        self.num_samples_test = 20
        self.num_rows = 3
        self.num_cols = 4
        self.seq_len = self.num_rows*self.num_cols # BOS and EOS will be added in collate_fn, not present in the data itself.
        self.num_object_classes = 3
        self.num_objects_to_place = 3
        
        self.batch_size = 32
        self.num_workers = 24
        
        self.dict_keys = [str(seed) for seed in self.seeds]
        self.params = {}.fromkeys(self.dict_keys)
        
        for i, key in enumerate(self.dict_keys):
            self.params[key] = {}
            self.params[key]['dataset_parameters'] = {'train':
                                                      {
                                                          'dataset':
                                                          {
                                                              '_target_': ConstantGridDataset
                                                             ,'split':"train"
                                                             ,'seed': self.seeds[i]
                                                             ,'num_samples': self.num_samples_train
                                                             ,'num_rows': self.num_rows
                                                             ,'num_cols': self.num_cols
                                                             ,'num_object_classes': self.num_object_classes
                                                             ,'num_objects_to_place': self.num_objects_to_place
                                                          },
                                                          'dataloader':
                                                          {
                                                              'batch_size': self.batch_size
                                                              ,'num_workers': self.num_workers
                                                          }
                                                      }
                                                      

                                                      ,'val':
                                                      {
                                                          'dataset':
                                                          {
                                                              '_target_': ConstantGridDataset
                                                             ,'split':"val"
                                                             ,'seed': self.seeds[i]
                                                             ,'num_samples': self.num_samples_val
                                                             ,'num_rows': self.num_rows
                                                             ,'num_cols': self.num_cols
                                                             ,'num_object_classes': self.num_object_classes
                                                             ,'num_objects_to_place': self.num_objects_to_place
                                                          },
                                                          'dataloader':
                                                          {
                                                              'batch_size': self.batch_size
                                                              ,'num_workers': self.num_workers
                                                          }

                                                      }
                                                      
                                                      ,'test':
                                                      {
                                                          'dataset':
                                                          {
                                                              '_target_': ConstantGridDataset
                                                             ,'split':"test"
                                                             ,'seed': self.seeds[i]
                                                             ,'num_samples': self.num_samples_test
                                                             ,'num_rows': self.num_rows
                                                             ,'num_cols': self.num_cols
                                                             ,'num_object_classes': self.num_object_classes
                                                             ,'num_objects_to_place': self.num_objects_to_place
                                                          },
                                                          'dataloader':
                                                          {
                                                              'batch_size': self.batch_size
                                                              ,'num_workers': self.num_workers
                                                          }
                                                      }
                                                      
                                                    
                                                 }

        self.pl_datamodules = []
        
        for seed in self.seeds:
            
            seed_everything(seed, workers=True)
            pl_datamodule = GridDataModule(seed, self.num_workers, **self.params[str(seed)])
            pl_datamodule.prepare_data()
            pl_datamodule.setup()
            self.pl_datamodules.append(pl_datamodule)
            

    def test_dataset_number_of_samples(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_datamodule = self.pl_datamodules[i]
                # Check the number of samples are correct
                train_split_dataset = pl_datamodule.data_train
                val_split_dataset = pl_datamodule.data_val
                test_split_dataset = pl_datamodule.data_test
                
                self.assertEqual(train_split_dataset.__len__(), self.num_samples_train)
                self.assertEqual(val_split_dataset.__len__(), self.num_samples_val)
                self.assertEqual(test_split_dataset.__len__(), self.num_samples_test)

    
    def test_dataset_sequence_length(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_datamodule = self.pl_datamodules[i]
                # Check the number of samples are correct
                train_split_dataset = pl_datamodule.data_train
                val_split_dataset = pl_datamodule.data_val
                test_split_dataset = pl_datamodule.data_test
                
                for sample in train_split_dataset:
                    self.assertEqual(len(sample['text'].split()), self.seq_len)
                    
                for sample in val_split_dataset:
                    self.assertEqual(len(sample['text'].split()), self.seq_len)
                    
                for sample in test_split_dataset:
                    self.assertEqual(len(sample['text'].split()), self.seq_len)
                    
    
    def test_dataset_num_obj(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_datamodule = self.pl_datamodules[i]

                # Note that 
                train_split_dataset = pl_datamodule.data_train
                val_split_dataset = pl_datamodule.data_val
                test_split_dataset = pl_datamodule.data_test
                
                for sample in train_split_dataset:
                    grid_sequence = sample['text'].split()
                    target_seq_len = self.params[str(seed)]['dataset_parameters']['train']['dataset']['num_objects_to_place']
                    # removing all empty (i.e. index='0') elements so we can count the number of objects.
                    grid_sequence = list(filter(lambda object_id: object_id != '0', grid_sequence))
                    num_objects = len(grid_sequence)
                    self.assertEqual(num_objects, target_seq_len)
                    
                for sample in val_split_dataset:
                    grid_sequence = sample['text'].split()
                    target_seq_len = self.params[str(seed)]['dataset_parameters']['val']['dataset']['num_objects_to_place']
                    grid_sequence = list(filter(lambda object_id: object_id != '0', grid_sequence))
                    num_objects = len(grid_sequence)
                    self.assertEqual(num_objects, target_seq_len)
                    
                for sample in test_split_dataset:
                    grid_sequence = sample['text'].split()
                    target_seq_len = self.params[str(seed)]['dataset_parameters']['test']['dataset']['num_objects_to_place']
                    grid_sequence = list(filter(lambda object_id: object_id != '0', grid_sequence))
                    num_objects = len(grid_sequence)
                    self.assertEqual(num_objects, target_seq_len)

    
    def test_dataset_reproducibility(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_datamodule_original = self.pl_datamodules[i]
                train_split_dataset_original = pl_datamodule_original.data_train
                val_split_dataset_original = pl_datamodule_original.data_val
                test_split_dataset_original = pl_datamodule_original.data_test
                
                # This is run at the beginning of train.py, so it should be here when testing as well.
                seed_everything(seed, workers=True)
                pl_datamodule_temp = GridDataModule(seed, self.num_workers, **self.params[str(seed)])
                pl_datamodule_temp.prepare_data()
                pl_datamodule_temp.setup()
                
                train_split_dataset_temp = pl_datamodule_temp.data_train
                val_split_dataset_temp = pl_datamodule_temp.data_val
                test_split_dataset_temp = pl_datamodule_temp.data_test
                
                for j, _ in enumerate(train_split_dataset_original):
                    self.assertEqual(train_split_dataset_original[j]['text'] == train_split_dataset_temp[j]['text'], True)                
                
                for j, _ in enumerate(val_split_dataset_original):
                    self.assertEqual(val_split_dataset_original[j]['text'] == val_split_dataset_temp[j]['text'], True)
                      
                for j, _ in enumerate(test_split_dataset_original):
                    self.assertEqual(test_split_dataset_original[j]['text'] == test_split_dataset_temp[j]['text'], True)



                    
                    
if __name__ == "__main__":
    unittest.main()
