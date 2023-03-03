import torch
import numpy as np
import os

mappings = {
    "tok_embeddings": 1,
    "output": 0,
    "wq.weight": 0,
    "wk.weight": 0,
    "wv.weight": 0,
    "wo.weight": 1,
    "w1.weight": 0,
    "w2.weight": 1,
    "w3.weight": 0,
    "norm.weight": -1
}
discard = "inner_attention"
if __name__ == '__main__':
    original_dir = "13B"
    model_shards = list()
    output_shards = dict()
    for i in range(2):
        model_shards.append(torch.load(f"{original_dir}/consolidated.{i:02}.pth"))
    for key in list(model_shards[0].keys()):
        if discard in key:
            continue
        for map in list(mappings.keys()):
            if map in key:
                if mappings[map] == -1:
                    output_shards[key] = model_shards[0][key]
                elif mappings[map] == 0:
                    output_shards[key] = torch.cat(list([model_shards[i][key] for i in range(len(model_shards))]), dim=0)
                elif mappings[map] == 1:
                    output_shards[key] = torch.cat(list([model_shards[i][key] for i in range(len(model_shards))]), dim=1)
        print(key, model_shards[0][key].shape)
    print("-------------------")
    for key in list(output_shards.keys()):
        print(key, model_shards[0][key].shape)
    torch.save(output_shards, f"{original_dir}/single.pth")