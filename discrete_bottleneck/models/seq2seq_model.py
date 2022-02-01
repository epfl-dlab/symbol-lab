import torch
from torch import nn
from .modules.lstm_encoder import Encoder
from .modules.lstm_decoder import Decoder
from .utils import init_weights
import random
import pytorch_lightning as pl

class Seq2Seq(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # This line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        self.encoder = Encoder(**self.hparams['encoder_params'])
        self.decoder = Decoder(**self.hparams['decoder_params'])
        
        layers = nn.ModuleList([self.encoder, self.decoder])
        
        self.apply(init_weights)
        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        # seq_len = row*col
        # src = [seq_len+1, batch_size]
        # trg = [seq_len+1, batch_size]

        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=src.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder are the <sos> tokens for all samples in the batch
        input = src[0,:]
        for t in range(0, trg_len):
            
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token

            input = trg[t] if teacher_force else top1
        
        return outputs
    
    def loss_function(self, prediction, target):
        
        batch_size = prediction.shape[0]
        
        return self.loss(prediction, target)/batch_size
    
    
    
    def sample(self, batch):
        
        '''
        batch: [batch_size, seq_len+2]
        '''
        
        trg = batch[1:, :]                # Removing the BOS tokens from trg sequences. # src: [seq_len+1, batch_size] (BOS+seq_len)
        src = batch[:batch.shape[0]-1, :] # Removing the EOS tokens from src sequences. # trg: [seq_len+1, batch_size] (seq_len+EOS)

        # [seq_len+1, batch_size, vocab_size]
        output = self(src, trg, teacher_forcing_ratio=0.0)
        
        # [seq_len+1, batch_size]
        predictions = output.argmax(2, keepdim=False).squeeze()
        
        message = ''
        for i in range(batch.shape[1]):
            message += f'\n source grid, predicted grid: {batch[1:, i].cpu().numpy()}, {predictions[:,i].cpu().numpy()}'

        print(message)
    


