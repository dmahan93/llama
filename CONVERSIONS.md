# Resharding

[reshard_model.py](reshard_model.py) contains the basics.
Right now it only reshards to a single model. If you need something different you should be able
to split it after the cat in the same dimension.

- Change line 19 to point to the model directory
- Change line 22 to the number of shards (2 for 13B, 4 for 30B, 8 for 65B)

It will then output a single model named single.pth in the same directory as the shards.

# HF conversion
[create_hf_model.py](create_hf_model.py)

- Change line 57 to point to your directory
- Change line 54 to point to your desired output directory

You can then use it like in the [hf_example.py](hf_example.py) script

# Questions? Support?

lmao, good luck