# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os

import fire

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
from llama import load

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
    samples: int = 1,
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

    generator, _ = load(
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
    while i < samples or samples <= 0:
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
