# -*- coding: utf-8 -*-

from ..utils import Registry

DATALOADER_REGISTRY = Registry("DATALOADER")


def build_dataloader(cfg, name=None):
    loader_name = cfg.dataloader.train.dataloader if name is None else name
    loader = DATALOADER_REGISTRY.get(loader_name)(cfg)
    return loader
