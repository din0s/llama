# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import torch
import fire
import time
import json

from pathlib import Path

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
from llama import ModelArgs, Transformer, Tokenizer, LLaMA, default_quantize

#from fairscale.nn.model_parallel.initialize import initialize_model_parallel

#def setup_model_parallel(seed: int) -> Tuple[int, int]:
#    local_rank = int(os.environ.get("LOCAL_RANK", -1))
#    world_size = int(os.environ.get("WORLD_SIZE", -1))
#
#    torch.distributed.init_process_group("nccl")
#    initialize_model_parallel(world_size)
#    torch.cuda.set_device(local_rank)
#
#    # seed must be the same in all processes
#    torch.manual_seed(seed)
#    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    #local_rank: int,
    #world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    quantize: bool,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.HalfTensor)
    print("Allocating transformer on host")
    ctx_tok = default_quantize.set(quantize)
    model = Transformer(model_args)
    default_quantize.reset(ctx_tok)
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }

    # ?
    torch.set_default_tensor_type(torch.FloatTensor)

    # load the state dict incrementally, to avoid memory problems
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
            del checkpoint[parameter_name]
        del checkpoint

    model.cuda()

    generator = LLaMA(model, tokenizer)
    print(
        f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
    )
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    # top_p: float = 0.95,
    top_p: float = 0.0,
    top_k: int = 40,
    #repetition_penalty_range: int = 1024,
    #repetition_penalty_slope: float = 0,
    repetition_penalty: float = (1 / 0.85),
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    max_batch_size: int = 8,
    seed: int = 1,
    count: int = 1,
    use_int8: bool = False,
    only_new: bool = True,
):
    #local_rank, world_size = setup_model_parallel(seed)
    #if local_rank > 0:
    #    sys.stdout = open(os.devnull, "w")

    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(json.dumps(dict(
        seed=seed,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        #repetition_penalty_range=repetition_penalty_range,
        #repetition_penalty_slope=repetition_penalty_slope,
        repetition_penalty=repetition_penalty,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
        use_int8=use_int8,
    )))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    generator = load(
        ckpt_dir,
        tokenizer_path,
        #local_rank,
        #world_size,
        max_seq_len,
        max_batch_size,
        use_int8,
    )

    prompts = []
    while True:
        f_in = input("Enter a file name (or ENTER to stop): ").strip()
        if f_in == "":
            break
        with open(f_in, "r") as f:
            prompts += [f.read().rstrip()]

    i = 0
    while i < count or count <= 0:
        i += 1
        for idx in range(0, len(prompts), max_batch_size):
            print(f"\n============== sample {i} =================\n")
            texts = generator.generate(
                prompts[idx:idx+max_batch_size],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                #repetition_penalty_range=repetition_penalty_range,
                #repetition_penalty_slope=repetition_penalty_slope,
                repetition_penalty=repetition_penalty,
                only_new=only_new,
            )

            for pidx, text in enumerate(texts):
                print(f"-------------- output {pidx+1} -----------------")
                print(text.split("\n")[0].strip())
                print()


if __name__ == "__main__":
    fire.Fire(main)
