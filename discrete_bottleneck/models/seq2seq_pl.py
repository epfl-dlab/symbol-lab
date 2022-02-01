from typing import Any, Dict, List

import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import discrete_bottleneck.utils as utils
import discrete_bottleneck.utils.tokenizers

from .seq2seq_model import Seq2Seq
import hydra


class Seq2SeqPl(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Setup for all computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = Seq2Seq(**self.hparams.nn_params)
        self.tokenizer = hydra.utils.instantiate(self.hparams.tokenizer)
        self.collator = hydra.utils.instantiate(self.hparams.collator)

    #         print(f'The model has {count_parameters(self.model):,} trainable parameters')

    def setup_collate_fn(self, datamodule):
        if not utils.tokenizers.is_tokenizer_trained(self.tokenizer):
            # Get the training data
            datamodule.prepare_data()
            datamodule.setup("fit")

            # Train tokenizer and update the collator's reference
            self.tokenizer.train_tokenizer(datamodule.data_train)
            self.collator.set_tokenizer(self.tokenizer)

            # Set the datamodule's collate_fn
            datamodule.set_collate_fn(self.collator.collate_fn)

    def forward(self, src, trg, teacher_forcing_ratio=0.5, **kwargs):

        output = self.model(
            src,
            trg,
            teacher_forcing_ratio=0.5,
            **kwargs,
        )

        return output

    def training_step(self, batch, batch_idx=None):

        batch = torch.transpose(batch, 0, 1)

        # Removing the BOS tokens from target sequences. # target: [seq_len+1, batch_size] (seq_len+EOS)
        # target = [trg_len, batch_size]
        target = batch[1:, :]

        # Removing the EOS tokens from source sequences. # source: [seq_len+1, batch_size] (BOS+seq_len)
        source = batch[: batch.shape[0] - 1, :]

        # model_output = [trg_len, batch_size, output_dim]
        model_output = self(source, target)

        output_dim = model_output.shape[-1]

        model_output = model_output.view(-1, output_dim)

        target = target.contiguous().view(-1)

        loss = self.model.loss_function(model_output, target)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):

        batch = torch.transpose(batch, 0, 1)

        # Removing the BOS tokens from target sequences. # source: [seq_len+1, batch_size] (BOS+seq_len)
        # target = [trg_len, batch_size]
        target = batch[1:, :]

        # Removing the EOS tokens from source sequences. # target: [seq_len+1, batch_size] (seq_len+EOS)
        source = batch[: batch.shape[0] - 1, :]

        # model_output = [trg_len, batch_size, output_dim]
        model_output = self(source, target)

        output_dim = model_output.shape[-1]

        model_output = model_output.view(-1, output_dim)

        target = target.contiguous().view(-1)

        loss = self.model.loss_function(model_output, target)

        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True)

        #         if batch.shape
        self.model.sample(batch)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):

        batch = torch.transpose(batch, 0, 1)

        # Removing the BOS tokens from target sequences. # source: [seq_len+1, batch_size] (BOS+seq_len)
        # target = [trg_len, batch_size]
        target = batch[1:, :]

        # Removing the EOS tokens from source sequences. # target: [seq_len+1, batch_size] (seq_len+EOS)
        source = batch[: batch.shape[0] - 1, :]

        # model_output = [trg_len, batch_size, output_dim]
        model_output = self(source, target)

        output_dim = model_output.shape[-1]

        model_output = model_output.view(-1, output_dim)

        target = target.contiguous().view(-1)

        loss = self.model.loss_function(model_output, target)

        self.log("test_loss", loss.item(), on_epoch=True, prog_bar=True)

        return {"test_loss": loss}

    def test_step_end(self, outputs: List[Any]):
        # Process the data in the format expected by the metrics

        return

    @rank_zero_only
    def _write_testing_output(self, outputs):
        #         with jsonlines.open("testing_output.jsonl", "w") as writer:
        #             for process_output in outputs:
        #                 for step_output in process_output:
        #                     items = [
        #                         {
        #                             "predictions": step_output["predictions"][i],
        #                             "targets": step_output["targets"][i],
        #                             "inputs": step_output["inputs"][i],
        #                         }
        #                         for i in range(len(step_output["predictions"]))
        #                     ]

        #                     writer.write_all(items)

        return

    #     def test_epoch_end(self, outputs):
    #         """Outputs is a list of either test_step outputs outputs"""
    #         # Log metrics aggregated across steps and processes (in ddp)
    #         self.log("test-precision", self.ts_precision.compute())
    #         self.log("test-recall", self.ts_recall.compute())
    #         self.log("test-f1", self.ts_f1.compute())

    #         if self.hparams.save_testing_data:
    #             # TODO: Test for typos in names
    #             # TODO: Maybe re-implement with each process writing to test_output_{rank}.jsonl files
    #             if torch.distributed.is_initialized():
    #                 torch.distributed.barrier()
    #                 gather = [None] * torch.distributed.get_world_size()
    #                 torch.distributed.all_gather_object(gather, outputs)
    #                 # Gather is a list of `num_gpu` elements, each being the outputs object passed to the test_epoch_end
    #                 outputs = gather
    #             else:
    #                 outputs = [outputs]

    #             self._write_testing_output(outputs)

    #         return {
    #             "test-acc": self.ts_precision.compute(),
    #             "test-recall": self.ts_precision.compute(),
    #             "test-f1": self.ts_precision.compute(),
    #         }

    def configure_optimizers(self):

        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization
        no_decay = ["bias"]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.nn_params.weight_decay,
                #                 "betas": self.hparams.nn_params.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.nn_params.adam_eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                #                 "betas": self.hparams.nn_params.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.nn_params.adam_eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.nn_params.lr,
            weight_decay=self.hparams.nn_params.weight_decay,
        )

        #         optimizer = torch.optim.Adam(
        #             optimizer_grouped_parameters,
        #             lr=self.hparams.nn_params.lr,
        # #             weight_decay=self.hparams.nn_params.weight_decay,
        #         )

        if self.hparams.nn_params.schedule_name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.nn_params.warmup_updates,
                num_training_steps=self.hparams.nn_params.total_num_updates,
            )
        elif self.hparams.nn_params.schedule_name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.nn_params.warmup_updates,
                num_training_steps=self.hparams.nn_params.total_num_updates,
                lr_end=self.hparams.nn_params.lr_end,
            )

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.nn_params.schedule_name}"
            # Used by a LearningRateMonitor callback
        }

        #         lr_dict = None

        return [optimizer], [lr_dict]


#         return [optimizer]
