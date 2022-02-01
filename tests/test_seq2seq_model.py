import random
import unittest

import pytorch_lightning as pl
import torch
import numpy as np

import sys
  
# setting path
sys.path.append('../')
from discrete_bottleneck.models.seq2seq_pl import Seq2SeqPL
from discrete_bottleneck.models.modules.lstm_encoder import Encoder
from discrete_bottleneck.models.modules.lstm_decoder import Decoder
from discrete_bottleneck.models.seq2seq_model import Seq2Seq
from pytorch_lightning import seed_everything


class TestSeq2SeqModel(unittest.TestCase):

    def setUp(self):
        
        self.seeds = [1]
#         self.seeds = [1, 123, 543, 358]
        
        self.dict_keys = [str(seed) for seed in self.seeds]
        self.params = {}.fromkeys(self.dict_keys)
        
        for key in self.dict_keys:
            self.params[key] = {}
            self.params[key]['input_dim'] = 4
            self.params[key]['output_dim'] = 4
            self.params[key]['enc_emb_dim'] = 32
            self.params[key]['dec_emb_dim'] = 32
            self.params[key]['hid_dim'] = 64
            self.params[key]['n_layers'] = 2
            self.params[key]['enc_dropout'] = 0.5
            self.params[key]['dec_dropout'] = 0.5
            

        self.pl_seq2seq = []
        
        for seed in self.seeds:
            seed_everything(seed, workers=True)
            pl_seq2seq = Seq2SeqPL(**self.params[str(seed)])
            
            self.pl_seq2seq.append(pl_seq2seq)
            
    
    def test_model_init_reproducibility(self):
        for i, seed in enumerate(self.seeds):
            with self.subTest(i=i):
                pl_seq2seq_original = self.pl_seq2seq[i]
                
                
                seed_everything(seed, workers=True)
                pl_seq2seq_temp = Seq2SeqPL(**self.params[str(seed)])
                pl_seq2seq_temp_generator = pl_seq2seq_temp.model.named_parameters()
                for name, param in pl_seq2seq_original.model.named_parameters():
                    
                    name_, param_ = next(pl_seq2seq_temp_generator)
                    self.assertEqual(torch.equal(param.data, param_.data), True)
                    
                    
                    
if __name__ == "__main__":
    unittest.main()

