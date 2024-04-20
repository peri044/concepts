# attention

This repo is inspired from https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention and is mostly for self learning the following concepts in detail and with code

* attention.py - contains SelfAttention, CausalAttention
* multi_head_attention.py - MultiHead attention, GroupedQueryAttention, EfficientMHA

## Multi-head attention
There are three types of MHA implementes provided in this repo. 
1) Uses a stack of CausalAttention layers and concatenates them to provide final context vector
2) Combines qkv into a single linear layer and later splits them. This uses torch.scaled_dot_product_attention which is much faster. 
3) GroupedQueryAttention which is an efficient MHA (with less parameters). This is exactly
same as the one used in Llama3 and is taken from <a href="https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/models/llama/modeling_llama.py#L355-L371">here</a>

## How to run

### Self attention and Causal attention
```
python attention.py --self_attn
python attention.py --causal_attn
```

### Multi head attention
```
python multi_head_attention.py --mha
python multi_head_attention.py --gqa 
python multi_head_attention.py --efficient_mha
```

* --mha : MultiHeadAttention Wrapper (combines multiple CausalAttention's)
* --gqa : GroupedQueryAttention (used in Llama3)
* --efficient_mha : Efficient MHA (which uses torch.nn.scaled_dot_product_attention)


On NVIDIA RTX 3080Ti, here's the results when you run `python multi_head_attention.py`

```py
# Self attention
num_tokens: 6, input d_in: 4, head_dim: 8
Input sequence shape:  torch.Size([1, 6, 4])
Self attention output:  torch.Size([1, 6, 8])
Self attention Average time:  0.23223400115966797  ms

# Causal attention
num_tokens: 6, input d_in: 4, head_dim: 8, dropout: 0.2
Input sequence shape:  torch.Size([1, 6, 4])
Causal attention output:  torch.Size([1, 6, 8])
Causal attention Average time:  0.38367748260498047  ms

# MHA Wrapper
num_tokens: 6, input d_in: 4, d_kq: 256, num_heads: 32, dropout: 0.2
Input sequence shape:  torch.Size([1, 6, 4])
MHA wrapper output:  torch.Size([1, 6, 256])
MultiHeadAttention Wrapper Average time:  11.990056037902832  ms

# Efficient MHA
num_tokens: 6, input d_in: 4, d_kq: 256, num_heads: 32, dropout: 0.2
Input sequence shape:  torch.Size([1, 6, 4])
Efficient MHA output:  torch.Size([1, 6, 256])
Efficient MHA Average time:  0.16221046447753906  ms

# GroupedQueryAttention
num_tokens: 6, input d_in: 4, d_kq: 256, num_heads: 32, dropout: 0.2
Input sequence shape:  torch.Size([1, 6, 4])
Grouped query attention output:  torch.Size([1, 6, 256])
GroupedQuery Attention Average time:  0.6246852874755859  ms
```

## Resources:

* https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
* https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
* https://github.com/ELS-RD/kernl/tree/main/tutorial - amazing resources on online softmax, flash attention
* Flash attention : https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf


