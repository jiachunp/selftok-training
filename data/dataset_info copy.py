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
        'imagenet_long_caption_overfitting': {
            'data_dir': '/data/data/Unified_Parquet/ImageNet-Long-Caption_overfitting', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 10, # number of data units to be sharded across all ranks and workers # 207
        },
        'blip3o_small': {
            'data_dir': '/data/data/Unified_Parquet/BLIP3o-60k', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 66, # number of data units to be sharded across all ranks and workers # 215
        },
        'blip3o_zju': {
            'data_dir': '/data/data/Unified_Parquet/ZJU_QI', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 18, # number of data units to be sharded across all ranks and workers # 215
        },
        'imagenet_long_caption': {
            'data_dir': '/data/data/Unified_Parquet/ImageNet-Long-Caption', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 621, # number of data units to be sharded across all ranks and workers # 207
        },
        'imagenet_short_caption': {
            'data_dir': '/data/data/Unified_Parquet/ImageNet-Short-Caption', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 621, # number of data units to be sharded across all ranks and workers # 207
        },
        'blip3o_short_caption': {
            'data_dir': '/data/data/Unified_Parquet/BLIP3o-Pretrain-Short-Caption', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 3666, # number of data units to be sharded across all ranks and workers # 1833
        },
        'blip3o_long_caption': {
            'data_dir': '/data/data/Unified_Parquet/BLIP3o-Pretrain-Long-Caption', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 2891, # number of data units to be sharded across all ranks and workers # 2891
        },
        'FLUX-Reason-6M-Aesthetics-Part01': {
            'data_dir': '/data/data/Unified_Parquet/FLUX-Reason-6M-Aesthetics-Part01', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 415, # number of data units to be sharded across all ranks and workers # 215
        },
        'FLUX-Reason-6M-Aesthetics-Part02': {
            'data_dir': '/data/data/Unified_Parquet/FLUX-Reason-6M-Aesthetics-Part02', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 454, # number of data units to be sharded across all ranks and workers # 215
        },
        'FLUX-Reason-6M-Imaginative': {
            'data_dir': '/data/data/Unified_Parquet/FLUX-Reason-6M-Imaginative', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 168, # number of data units to be sharded across all ranks and workers # 215
        },
        'FLUX-Reason-6M-Text': {
            'data_dir': '/data/data/Unified_Parquet/FLUX-Reason-6M-Text', # path of the parquet files
            'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
            'num_files': 143, # number of data units to be sharded across all ranks and workers # 215
        }
    },
}


# DATASET_INFO = {
#     'selftok_t2i': {
#         'selftok_t2i': {
#             'data_dir': '/home/jovyan/zfd/SelfTok-o-main/dataset/overfitting/parquet', # path of the parquet files
#             'token_dir': '/home/jovyan/zfd/SelfTok-o-main/dataset/overfitting/image_token', # path of the token
#             'num_files': 207, # number of data units to be sharded across all ranks and workers
#             'num_total_samples': 1281167, # number of total samples in the dataset
#         },
#     },
# }

# DATASET_INFO = {
#     'selftok_t2i': {
#         'selftok_t2i': {
#             'data_dir': '/data/data/imagenet/imagenet_caption_only', # path of the parquet files
#             'token_dir': '/data/data/imagenet/imagenet512_codes', # path of the token
#             'num_files': 207, # number of data units to be sharded across all ranks and workers
#             'num_total_samples': 1281167, # number of total samples in the dataset
#         },
#     },
# }


# DATASET_INFO = {
#     'selftok_t2i': {
#         'selftok_t2i': {
#             'data_dir': '/data/data/BLIP3o/BLIP3o-Pretrain-Long-Caption-Parqute', # path of the parquet files
#             'token_dir': '/data/data/BLIP3o/BLIP3o-Pretrain-Long-Caption-W1Tokens', # path of the token
#             'num_files': 2891, # number of data units to be sharded across all ranks and workers
#         },
#     },
# }


# DATASET_INFO = {
#     'selftok_t2i': {
#         'selftok_t2i': {
#             'data_dir': '/data/data/BLIP3o/BLIP3o-Pretrain-Long-Caption-Parqute', # path of the parquet files
#             'token_dir': '/data/data/BLIP3o/BLIP3o-Pretrain-Long-Caption-W1Tokens-Render', # path of the token
#             'num_files': 2891, # number of data units to be sharded across all ranks and workers
#         },
#     },
# }


# DATASET_INFO = {
#     'selftok_t2i': {
#         'blip3o_long_caption': {
#             'data_dir': '/data/data/Unified_Parquet/BLIP3o-Pretrain-Long-Caption', # path of the parquet files
#             'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
#             'num_files': 2891, # number of data units to be sharded across all ranks and workers # 2891
#         },
#         'blip3o_short_caption': {
#             'data_dir': '/data/data/Unified_Parquet/BLIP3o-Pretrain-Short-Caption', # path of the parquet files
#             'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
#             'num_files': 1833, # number of data units to be sharded across all ranks and workers # 1833
#         },
#         'imagenet_long_caption': {
#             'data_dir': '/data/data/Unified_Parquet/ImageNet-Long-Caption', # path of the parquet files
#             'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
#             'num_files': 207, # number of data units to be sharded across all ranks and workers # 215
#         },
#         'imagenet_short_caption': {
#             'data_dir': '/data/data/Unified_Parquet/ImageNet-Short-Caption', # path of the parquet files
#             'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
#             'num_files': 207, # number of data units to be sharded across all ranks and workers # 215
#         },
#     },
# }


# DATASET_INFO = {
#     'selftok_t2i': {
#         'blip3o_long_caption': {
#             'data_dir': '/data/data/Unified_Parquet/BLIP3o-Pretrain-Long-Caption', # path of the parquet files
#             'token_dir': '/data/data/Unified_W1Tokens_Render', # path of the token
#             'num_files': 2891, # number of data units to be sharded across all ranks and workers # 2891
#         },
#     },
# }
