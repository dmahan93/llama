# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Tokenizer, LLaMA
from llama.hf_import import LLaMAForCausalLM, LLAMAConfig
from transformers import DebertaV2Tokenizer, convert_slow_tokenizer, PreTrainedTokenizerFast


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[local_rank] if len(checkpoints) == 0 else ckpt_dir + "/single.pth"
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=2, **params)
    sptokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = sptokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    config = LLAMAConfig(
        sptokenizer.n_words, 2048, model_args.dim, model_args.n_layers, model_args.n_heads, multiple_of=model_args.multiple_of, norm_eps=model_args.norm_eps,
        bos_token_id=sptokenizer.bos_id, eos_token_id=sptokenizer.eos_id
    )
    model = LLaMAForCausalLM(config)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.transformer.load_state_dict(checkpoint, strict=False)
    model.save_pretrained("LLAMAHF")


def main(ckpt_dir: str = "./7B", tokenizer_path: str = "./tokenizer.model", temperature: float = 1.0, top_p: float = 0.95):
    generator = load(ckpt_dir, tokenizer_path, 0, 1)



if __name__ == "__main__":
    fire.Fire(main)
