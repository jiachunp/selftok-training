# -*- coding: utf-8 -*-

import torch
import numpy as np
# import youtokentome as yttm
from transformers import BertTokenizer, LlamaTokenizer


def get_text_tokenizer(path, tokenizer_type, **kwargs):
    if tokenizer_type == "mimo_LLaMA":
        tokenizer = TokenizerLLaMA(path)
    else:
        print("tokenizer_type:{} NotImplemented".format(tokenizer_type))
        raise NotImplementedError
    print("text tokenizer --> ready")
    return tokenizer




class TokenizerLLaMA:
    def __init__(self, model_name_or_path="LLaMA_tokenizer"):

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=2048,
            padding_side="right",
            truncation=True,
            use_fast=False,
        )
        special_token_list = ["[IMG]", "[/IMG]", "<image>"]
        special_tokens_dict = dict(
            pad_token="[PAD]",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            additional_special_tokens=special_token_list,
        )
        # base_tokenizer_size = self.tokenizer.vocab_size
        new_special_num = self.tokenizer.add_special_tokens(special_tokens_dict)
        print("the special tokenizer is:", new_special_num)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self.unk_token_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.ignore_ids = [self.eos_token_id, self.bos_token_id, self.unk_token_id, self.pad_token_id]
        self.vocab_size = len(self.tokenizer)
        print(f"The Special Tokens: {self.tokenizer.special_tokens_map}")
        print(f"Vocab Size: {len(self.tokenizer)}")

    def decode_text(self, tokens):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        eos_idx = 0
        for eos_idx in range(len(tokens)):
            if tokens[eos_idx] == self.eos_token_id:
                break
        tokens = tokens[: eos_idx + 1]

        tokens = [token for token in tokens if token not in self.ignore_ids]
        return self.tokenizer.decode(tokens)

    def encode_caption(self, text, text_seq_length):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > text_seq_length:
            tokens = tokens[:text_seq_length]
        elif len(tokens) < text_seq_length:
            empty_positions = text_seq_length - len(tokens)
            pad_tokens = [self.pad_token_id] * empty_positions
            tokens = np.hstack((tokens, pad_tokens))
        return tokens

    def encoder_text_all(self, caption_tokens, image_placeholder, all_seq_length):
        image_placeholder_tokens = self.tokenizer.encode(image_placeholder, add_special_tokens=False)
        all_tokens = np.hstack(([self.bos_token_id], caption_tokens, image_placeholder_tokens, [self.eos_token_id]))
        if len(all_tokens) != all_seq_length:
            print("len error!!!", len(all_tokens))
        return torch.tensor(all_tokens).long()


