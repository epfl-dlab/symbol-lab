import random
import unittest

import pytorch_lightning as pl
import torch
import numpy as np

import sys
  
# setting path
sys.path.append('../')
from discrete_bottleneck.datamodule.datasets_pl import DataModule

from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import discrete_bottleneck.utils.general as utils

log = utils.get_logger(__name__)




class TestSeq2SeqPerformance(unittest.TestCase):
    
    def setUp(self):
        
        self.seeds = [1, 123, 543, 358]
        
        self.num_samples_train = 500
        self.num_samples_val = 300
        self.num_samples_test = 200
        self.row = 3
        self.col = 4
        self.seq_len = self.row*self.col + 2 # Because we have BOS and EOS in all sequences
        self.num_objects = 3
        self.vocab = ['empty', 'circle', 'square', 'triangle'] # keep 'empty' as the first, so stoi for it would be 2
        
        self.dataset_name = 'grid'
        self.batch_size = 32
        
        self.dict_keys = [str(seed) for seed in self.seeds]
        self.params = {}.fromkeys(self.dict_keys)
        
        for key in self.dict_keys:
            self.params[key] = {}
            self.params[key]['dataset'] = {'train_split':
                                        {
                                        'n_samples': self.num_samples_train
                                        ,'batch_size': self.batch_size
                                        ,'row': self.row
                                        ,'col': self.col
                                        ,'num_objects': self.num_objects
                                        ,'vocab': self.vocab
                                        }
                                      ,'valid_split':
                                        {
                                        'n_samples': self.num_samples_val
                                        ,'batch_size': self.batch_size
                                        ,'row': self.row
                                        ,'col': self.col
                                        ,'num_objects': self.num_objects
                                        ,'vocab': self.vocab           
                                        }
                                        ,'test_split':
                                        {
                                        'n_samples': self.num_samples_test
                                        ,'batch_size': self.batch_size
                                        ,'row': self.row
                                        ,'col': self.col
                                        ,'num_objects': self.num_objects
                                        ,'vocab': self.vocab  
                                        }
                                     }

        self.pl_datamodules = []
        
        for seed in self.seeds:
            pl_datamodule = DataModule(self.dataset_name, self.batch_size, seed, **self.params[str(seed)])
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
                    self.assertEqual(len(sample), self.seq_len)
                    
                for sample in val_split_dataset:
                    self.assertEqual(len(sample), self.seq_len)
                    
                for sample in test_split_dataset:
                    self.assertEqual(len(sample), self.seq_len)
                    
    
    def test_dataset_num_obj(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_datamodule = self.pl_datamodules[i]

                train_split_dataset = pl_datamodule.data_train
                val_split_dataset = pl_datamodule.data_val
                test_split_dataset = pl_datamodule.data_test
                
                vocab = train_split_dataset.vocab
                # vocab.stoi['empty'] = 2
                for sample in train_split_dataset:
                    # removing BOS, EOS, and empty tokens
                    sample = [tok for tok in sample if tok>2]
                    self.assertEqual(len(sample), self.num_objects)
                    
                for sample in val_split_dataset:
                    # removing BOS, EOS, and empty tokens
                    sample = [tok for tok in sample if tok>2]
                    self.assertEqual(len(sample), self.num_objects)
                    
                for sample in test_split_dataset:
                    # removing BOS, EOS, and empty tokens
                    sample = [tok for tok in sample if tok>2]
                    self.assertEqual(len(sample), self.num_objects)



def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score useful for hyperparameter optimization.
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Initialize the LIT model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Initialize the LIT data module
    log.info(f"Instantiating data module <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, tokenizer=model.tokenizer)

    # Initialize LIT callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init LIT loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from configs to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training"):
        if config.get("debug") or config.trainer.get("fast_dev_run"):
            log.info("Option to perform testing was selected in debug mode!")
            if config.get("debug_ckpt_path"):
                log.info("Starting testing with given debug checkpoint!")
                trainer.test(ckpt_path=config.get("debug_ckpt_path"))
            else:
                if config.trainer.get("fast_dev_run"):
                    log.info("No checkpoint was passed, nor created! Testing is skipped")

                log.info("Trying to start testing with dummy checkpoint!")
                trainer.test()
        else:
            log.info("Starting testing!")
            trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Used in hyperparameter optimization; returns the metric score
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]



                    
                    
if __name__ == "__main__":
    unittest.main()


    
    