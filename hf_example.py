import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline, DebertaV2Tokenizer, PreTrainedTokenizer
from llama.tokenizer import Tokenizer
from llama.hf_import import LLaMAConfig, LLaMAForCausalLM
LLaMAConfig.register_for_auto_class()
LLaMAForCausalLM.register_for_auto_class("AutoModelForCausalLM")


if __name__ == '__main__':
    tokenizer = Tokenizer("./tokenizer.model")
    model = LLaMAForCausalLM.from_pretrained(
        r"./LLAMAHF_13B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0
    )  # type: TextGenerationPipeline
    text = generator("mr Llama what do you think you're doing with 13B parameters?", max_new_tokens=28)[0]['generated_text']
    print(tokenizer.decode(tokenizer(text)['input_ids']), text, tokenizer(text)['input_ids'])