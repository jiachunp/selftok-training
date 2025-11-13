# -*- coding: utf-8 -*-
from .text_tokenizer import get_text_tokenizer

__all__ = [
    # 兼容vlip
    "tokenize",
    # below are new api
    "get_text_tokenizer",
]
