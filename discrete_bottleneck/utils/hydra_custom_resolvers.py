from omegaconf import OmegaConf
import math


def add_args(*args):
    return sum(float(x) for x in args)


def add_args_int(*args):
    return int(sum(float(x) for x in args))


def multiply_args(*args):
    return math.prod(float(x) for x in args)


def multiply_args_int(*args):
    return int(math.prod(float(x) for x in args))


OmegaConf.register_new_resolver("add", add_args)
OmegaConf.register_new_resolver("mult", multiply_args)

OmegaConf.register_new_resolver("add_int", add_args_int)
OmegaConf.register_new_resolver("mult_int", multiply_args_int)
