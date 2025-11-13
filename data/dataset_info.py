# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .t2i_selftok_dataset import Selftok_T2IIterableDataset

DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'selftok_t2i': Selftok_T2IIterableDataset,
}

DATASET_INFO = {
    'selftok_t2i': {
        'imagenet_long_caption': {
            'data_dir': '/data/data/Unified_Parquet/ImageNet-Long-Caption', # path of the parquet files
            'token_dir': '/data/data/Unified_Tokens_long_270k', # path of the token
            'num_files': 207, # number of data units to be sharded across all ranks and workers # 207
        },
    },
}
