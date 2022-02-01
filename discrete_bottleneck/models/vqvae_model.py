from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

import torch
from torch import nn
from abc import abstractmethod
from torch.nn import functional as F

import random

from .modules.lstm_encoder import Encoder
from .modules.lstm_decoder import Decoder
from .modules.base_vae import BaseVAE
from .modules.vector_quantizer import VectorQuantizer

    
class VQVAE(BaseVAE):

    def __init__(self,
                 # embedding_dim: int,
                 # num_embeddings: int,
                 # beta: float = 0.25,
                 **kwargs) -> None:
        super().__init__()
        
        # Remember that BaseVAE was defined as a subclass of pytorch-lightning module
        # This line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        

        self.embedding_dim = self.hparams['d_embedding']
        self.num_embeddings = self.hparams['num_embeddings']
        self.beta = self.hparams['beta']
        
        
        # Build Encoder
        self.encoder = Encoder(**self.hparams['encoder_params'])
        self.vq_layer = VectorQuantizer(embedding_dim=self.embedding_dim
                                        , num_embeddings=self.num_embeddings
                                        , beta=self.beta)
        # Build Decoder
        self.decoder = Decoder(**self.hparams['decoder_params'])
        
        # I'm not sure if we absolutely need this line or not (given the self. notation used above)
        # Using nn.ModuleList is helpful for pytorch-lightning to register modules on the correct 
        # device.
        self.modules = nn.ModuleList([self.encoder, self.vq_layer, self.decoder])
        
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        
    def encode(self, input_tensor: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_tensor)
        return result

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        h = kwargs['hidden']
        c = kwargs['cell']
        
        result = self.decoder(z, h, c)
        
        # result = (prediction, new_hidden, new_cell)
        # prediction: [batch_size, output_dim]
        # new_hidden, new_cell:  [n_layers, batch_size, hid_dim] 

        return result

    
    def forward(self, src, trg, **kwargs) -> List[Tensor]:
        
        '''
        src: [seq_len+1, batch_size] (BOS+seq_len)
        trg: [seq_len+1, batch_size] (seq_len+EOS)
        '''


        batch_size = trg.shape[1]
        trg_len = trg.shape[0] # seq_len + eos
        trg_vocab_size = self.decoder.output_dim # n_obj+empty+2 (bos,eos)
        
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        teacher_forcing_ratio = kwargs['teacher_forcing_ratio']
        
        # Encoding
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encode(src)

        # Quantize
        quantized_hidden, vq_loss_hidden = self.vq_layer(hidden)
        quantized_cell, vq_loss_cell = self.vq_layer(cell)
        
        hidden = quantized_hidden
        cell = quantized_cell
        
        
        # Decoding
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=src.device)
        
        # first input to the decoder are the <bos> tokens for all samples in the batch
        _input = src[0,:]
        
        for t in range(0, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decode(_input, hidden=hidden, cell=cell)
            # output, hidden, cell = self.decode(_input, hidden=hidden, cell=cell)
            
            # output: [batch_size, output_dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            # top1: [batch_size]
            top1 = output.argmax(1) 

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            _input = trg[t] if teacher_force else top1
        
        
#         encoding = self.encode(input)[0]
#         quantized_inputs, vq_loss = self.vq_layer(encoding)
#         return [self.decode(quantized_inputs), input, vq_loss]
#         return [outputs, input, vq_loss_hidden+vq_loss_cell]

        return [outputs, trg, vq_loss_hidden+vq_loss_cell]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        
        # [seq_len+1 (eos), batch_size, output_dim (or vocab_size)] output_dim is
        # the vocab_size, because the output is the result of softmax
        recons = args[0]   
        target = args[1]    # [seq_len+1 (eos), batch_size]
        vq_loss = args[2]  # scalar 
        
        batch_size = recons.shape[1]   
        output_dim = recons.shape[-1]

        recons = recons.view(-1, output_dim)
        target = target.contiguous().view(-1)

        recons_loss = self.loss(recons, target)/batch_size
        vq_loss = vq_loss/batch_size

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        return self.forward(x)[0]