# This code is entirely taken from https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/01_main-chapter-code.
# Credit goes to Sebastian Raschka
# Any changes made here are for self learning

import torch
import json
import os
import re
import urllib
from tqdm import tqdm
from dataloader import *
from previous_chapters import (
    GPTModel,
    generate,
    text_to_token_ids,
    token_ids_to_text
)
import tiktoken

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Load an SFT model
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth", map_location=torch.device("cpu")))
model.eval()
model = model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")

# Gather all the responses for test dataset
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing


