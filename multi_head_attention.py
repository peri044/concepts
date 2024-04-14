import torch 
import torch.nn as nn
from attention import MultiHeadAttentionWrapper

class EfficientMHA(torch.nn.Module):
    def __init__(self, d_in, d_kq, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_kq % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.d_in = d_in
        self.d_kq = d_kq
        self.head_dim = d_kq // num_heads
        self.qkv = nn.Linear(d_in, 3*d_kq, bias=qkv_bias)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        assert embed_dim == self.d_in, "Input embedding size should match with d_in"

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        use_dropout = 0. if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_kq = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_kq)

        return context_vec

num_tokens = 6; d_in = 4; d_kq = 256; dropout=0.2; num_heads = 32

# This assumes queries, keys, values have save dimension (d_kq)
mha_efficient = EfficientMHA(
    d_in=d_in,
    d_kq=d_kq,
    dropout=0.2,
    num_heads=32,
    qkv_bias=False
).cuda()

input_seq = torch.randn((1, num_tokens, d_in)).cuda()
# Output shape of mha = 1 x num_tokens x (num_heads*d_v)
print("Efficient MHA output: ", mha_efficient(input_seq).shape, "num_heads: ", num_heads)

d_kq = d_kq // num_heads; d_v = d_kq // num_heads
mha = MultiHeadAttentionWrapper(d_in, d_kq, d_v, dropout, num_heads)
# Time both the attentions
import time
import numpy as np
timings = []
for _ in range(50):
    start_time = time.time()
    mha(input_seq)
    timings.append(time.time()-start_time)
print("MHA Wrapper Average time: ", np.mean(timings)*1000, " ms")

timings = []
for _ in range(50):
    start_time = time.time()
    mha_efficient(input_seq)
    timings.append(time.time()-start_time)
print("MHA Efficient Average time: ", np.mean(timings)*1000, " ms")


