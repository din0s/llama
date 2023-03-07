# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import Dict, List, Union
import torch
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def __call__(self, *args, **kwargs) -> Dict[str, torch.IntTensor]:
        return self.tokenize(*args, **kwargs)

    def tokenize(self, s: str, truncation: bool = False, max_length: int = -1) -> Dict[str, torch.IntTensor]:
        enc = self.encode(s, bos=True, eos=False)
        if truncation:
            end = min(max_length, len(enc))
            enc = enc[:end]
        return { "input_ids": torch.IntTensor(enc) }

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: Union[List[int], torch.IntTensor]) -> str:
        if isinstance(t, torch.IntTensor):
            t = t.tolist()
        return self.sp_model.decode(t)
