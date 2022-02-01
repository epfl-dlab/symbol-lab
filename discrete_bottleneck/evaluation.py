from typing import List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

import discrete_bottleneck.utils.general as utils

log = utils.get_logger(__name__)


def evaluate(config: DictConfig) -> Optional[float]:
    """Contains the code for evaluation from file.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        List[tuple[str, float]]: Metric name, metric score pairs
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    getter = hydra.utils.instantiate(config.getter)

    output_data = getter.get_output_data()

    preds = [getter.get_predicted(sample) for sample in output_data]
    target = [getter.get_target(sample) for sample in output_data]

    metrics = {}
    for metric_key, metric_conf in config.metrics.items():
        log.info(f"Instantiating metric <{metric_conf._target_}>, named {metric_key}")
        metric = hydra.utils.instantiate(metric_conf)
        metric(preds, target)
        metric_score = metric.compute()
        metrics[metric_key] = metric_score
        log.info(f"{metric_key}, {metric_score:.4f}")

    loggers: List[LightningLoggerBase] = []

    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                logger.log_metrics(metrics, 0)
                loggers.append(logger)

    for lg in loggers:
        lg.finalize(0)

        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()

    return metrics
