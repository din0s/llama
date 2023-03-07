# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.0, #0.95,
        repetition_penalty_range: int = 1024,
        repetition_penalty_slope: float = 0.7,
        repetition_penalty: float = (1.0 / 0.85),
        token_callback=None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        prev_text = ''
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                next_token_scores = logits
                next_token_scores = apply_temperature(next_token_scores, temperature)
                next_token_scores = apply_top_pk(next_token_scores, top_p, top_k)
                next_token_scores = apply_advanced_repetition_penalty(
                    tokens[:, :cur_pos],
                    next_token_scores,
                    repetition_penalty_range,
                    repetition_penalty_slope,
                    repetition_penalty,
                )
                next_token_scores = torch.nn.functional.softmax(
                    next_token_scores, dim=-1
                )
                next_token = torch.multinomial(
                    next_token_scores, num_samples=1
                ).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if next_token == self.tokenizer.eos_id:
                break
            tokens[:, cur_pos] = next_token
            if token_callback is not None:
                assert len(prompts) == 1
                text, = self.decode(tokens)
                #assert text.startswith(prev_text)
                if not text.startswith(prev_text):
                    # Some kind of bogus token generation; abort early.
                    break
                next_word = text[len(prev_text):]
                prev_text = text
                token_callback(next_word)
            prev_pos = cur_pos

        return self.decode(tokens)

    def decode(self, tokens):
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = [token for token in t if token != -1]
            # # cut to max gen len
            # t = t[: len(prompt_tokens[i]) + max_gen_len]
            while self.tokenizer.eos_id in t:
                pos = t.index(self.tokenizer.eos_id)
                t[pos:pos+1] = self.tokenizer.encode('\n<|endoftext|>\n', bos=False, eos=False)
            decoded.append(self.tokenizer.decode(t))
        return decoded


def apply_temperature(scores, tempt):
    scores = scores / tempt
    return scores


def apply_top_pk(scores, top_p, top_k, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_p <= 0 and top_k <= 0:
        return scores

    if top_k > 0:
        sorted_logits, sorted_indices = torch.topk(scores, top_k)
    else:  # top_p > 0
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)

    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def apply_advanced_repetition_penalty(
    input_ids, scores, penalty_range, penalty_slope, penalty
):
    penalty_range = int(penalty_range)
    clipped_penalty_range = min(input_ids.shape[-1], penalty_range)

    if penalty != 1.0:
        if penalty_range > 0:
            if clipped_penalty_range < input_ids.shape[1]:
                input_ids = input_ids[..., -clipped_penalty_range:]

            if penalty_slope != 0:
                _penalty = (
                    torch.arange(
                        penalty_range, dtype=scores.dtype, device=scores.device
                    )
                    / (penalty_range - 1)
                ) * 2.0 - 1
                _penalty = (penalty_slope * _penalty) / (
                    1 + torch.abs(_penalty) * (penalty_slope - 1)
                )
                _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
                penalty = _penalty[..., -clipped_penalty_range:]

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score <= 0, score * penalty, score / penalty)
        scores.scatter_(1, input_ids, score)

    return scores
