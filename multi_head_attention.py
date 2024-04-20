import torch 
import torch.nn as nn
import argparse
from attention import CausalAttention, measure_time

class EfficientMHA(torch.nn.Module):
    def __init__(self, d_in, d_kq, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_kq % num_heads == 0, "d_kq is indivisible by num_heads"

        self.num_heads = num_heads
        self.d_in = d_in
        self.d_kq = d_kq
        self.head_dim = d_kq // num_heads
        self.qkv = nn.Linear(d_in, 3*self.d_kq, bias=qkv_bias)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        assert embed_dim == self.d_in, "Input embedding size should match with d_in"

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * self.d_kq)
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
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, head_dim, dropout, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, head_dim, dropout).eval().cuda() for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_kq, num_heads, num_kv_heads, dropout):
        super().__init__()
        self.d_kq = d_kq
        self.head_dim = d_kq // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads 
        # d_in is the input embedding size of each token
        # head_dim is the query/key/value embedding size
        self.W_query = nn.Parameter(torch.rand(d_in, self.num_heads * self.head_dim))
        self.W_key   = nn.Parameter(torch.rand(d_in, self.num_kv_heads * self.head_dim))
        self.W_value = nn.Parameter(torch.rand(d_in, self.num_kv_heads * self.head_dim))
        self.dropout = nn.Dropout(dropout)
    
    def repeat_kv(self, hidden_states, n_repeats):
        # states is of shape (1, num_kv_heads, seq_len, head_dim)
        # We transform it to (1, num_heads, seq_len, head_dim)
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_repeats == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_repeats, seq_len, head_dim)
        hidden_states = hidden_states.reshape(batch, num_kv_heads * n_repeats, seq_len, head_dim)
        return hidden_states
    
    def forward(self, x):
        # input_dim is same as d_in
        batch_size, num_tokens, input_dim = x.shape
        
        keys = torch.matmul(x, self.W_key) # 1 x num_tokens x self.num_kv_heads * head_dim
        queries = torch.matmul(x, self.W_query) # 1 x num_tokens x self.num_heads * head_dim
        values = torch.matmul(x, self.W_value) # 1 x num_tokens x self.num_kv_heads * head_dim
        
        # Reshape keys/values to (1 x num_kv_heads x seq_length x head_dim)
        # Reshape queries to (1 x num_heads x seq_length x head_dim)
        queries = queries.view(1, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat keys and values to meet num_heads on 2nd dimension.
        # The main idea is we are using same keys/values for a group of queries
        keys = self.repeat_kv(keys, self.num_heads // self.num_kv_heads)
        values = self.repeat_kv(values, self.num_heads // self.num_kv_heads)

        # unnormalized attention weights
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))  # 1 x num_tokens x num_tokens
        block_size = num_tokens
        # Create an upper triangular matrix with ones. 
        # The reasoning is each token can only rely on the past tokens and itself and not the future tokens
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).cuda()
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(
            masked / self.num_heads**0.5, dim=-1 # scale the dot product and apply softmax
        ) # shape is 1 x num_tokens x num_tokens
        # Apply dropout 
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.matmul(attn_weights, values) # 1 x num_heads x num_tokens x head_dim
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_kq)
        return context_vec

def main(args):
    if args.efficient_mha:
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
        print(f"num_tokens: {num_tokens}, input d_in: {d_in}, d_kq: {d_kq}, num_heads: {num_heads}, dropout: {dropout}")
        print("Input sequence shape: ", input_seq.shape)
        # Output shape of mha = 1 x num_tokens x (num_heads*d_v)
        print("Efficient MHA output: ", mha_efficient(input_seq).shape)
        measure_time(mha_efficient, input_seq, "Efficient MHA")
    elif args.gqa:
        num_tokens = 6; d_in = 4; d_kq = 256; num_heads = 32; num_kv_heads = 8; dropout=0.2
        gqa_att = GroupedQueryAttention(d_in, d_kq, num_heads, num_kv_heads, dropout).eval().cuda()

        input_seq = torch.randn((1, num_tokens, d_in)).cuda()
        context_output = gqa_att(input_seq) # Output shape = 1 x num_tokens x head_dim
        print(f"num_tokens: {num_tokens}, input d_in: {d_in}, d_kq: {d_kq}, num_heads: {num_heads}, dropout: {dropout}")
        print("Input sequence shape: ", input_seq.shape)
        print("Grouped query attention output: ", context_output.shape)
        measure_time(gqa_att, input_seq, "GroupedQuery Attention")
    elif args.mha:
        num_tokens = 6; d_in = 4; d_kq = 256; dropout=0.2; num_heads = 32
        head_dim = d_kq // num_heads
        mha = MultiHeadAttentionWrapper(d_in, head_dim, dropout, num_heads)
        input_seq = torch.randn((1, num_tokens, d_in)).cuda()
        print(f"num_tokens: {num_tokens}, input d_in: {d_in}, d_kq: {d_kq}, num_heads: {num_heads}, dropout: {dropout}")
        print("Input sequence shape: ", input_seq.shape)
        # Output shape of mha = 1 x num_tokens x (num_heads*d_v)
        print("MHA wrapper output: ", mha(input_seq).shape)
        measure_time(mha, input_seq, "MultiHeadAttention Wrapper")
    else:
        raise ValueError("Invalid attention flag provided. Options include --efficient_mha, --gqa")
    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run different types of attention. Options include MultiHeadAttention (MHA), EfficientMHA, GroupedQueryAttention"
    )
    arg_parser.add_argument(
        "--efficient_mha",
        action="store_true",
        help="Boolean flag to run EfficientMHA",
    )
    arg_parser.add_argument(
        "--gqa",
        action="store_true",
        help="Boolean flag to run GroupedQueryAttention",
    )
    arg_parser.add_argument(
        "--mha",
        action="store_true",
        help="Boolean flag to run MHAWrapper",
    )

    args = arg_parser.parse_args()
    main(args)
