
import torch 
import torch.nn as nn
import argparse
import numpy as np
import time

__all__ = ["SelfAttention", "CausalAttention"]

def measure_time(model, input_seq, model_name):
    timings = []
    for _ in range(50):
        start_time = time.time()
        model(input_seq)
        timings.append(time.time()-start_time)
    print(f"{model_name} Average time: ", np.mean(timings)*1000, " ms")

class SelfAttention(nn.Module):
    def __init__(self, d_in, head_dim):
        super().__init__()
        self.head_dim = head_dim
        # d_in is the input embedding size of each token
        # head_dim is the key/query matrix output size
        # head_dim is the value matrix output size
        self.W_query = nn.Parameter(torch.rand(d_in, head_dim))
        self.W_key   = nn.Parameter(torch.rand(d_in, head_dim))
        self.W_value = nn.Parameter(torch.rand(d_in, head_dim))
    
    def forward(self, x):
        keys = torch.matmul(x, self.W_key) # 1 x num_tokens x head_dim
        queries = torch.matmul(x, self.W_query) # 1 x num_tokens x head_dim
        values = torch.matmul(x, self.W_value) # 1 x num_tokens x head_dim
         
        keys_transpose = keys.transpose(2, 1) # 1 x d_kq x num_tokens
        # unnormalized attention weights
        attn_scores = torch.matmul(queries, keys_transpose)  # dot product of queries and keys 
        attn_weights = torch.softmax(
            attn_scores / self.head_dim**0.5, dim=-1 # scale the dot product and apply softmax
        ) # shape is 1 x num_tokens x num_tokens

        context_vec = torch.matmul(attn_weights, values) # 1 x num_tokens x d_v
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in, head_dim, dropout):
        super().__init__()
        self.head_dim = head_dim
        # d_in is the input embedding size of each token
        # head_dim is the key/query matrix output size
        # head_dim is the value matrix output size
        self.W_query = nn.Parameter(torch.rand(d_in, head_dim))
        self.W_key   = nn.Parameter(torch.rand(d_in, head_dim))
        self.W_value = nn.Parameter(torch.rand(d_in, head_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        keys = torch.matmul(x, self.W_key) # 1 x num_tokens x d_kq
        queries = torch.matmul(x, self.W_query) # 1 x num_tokens x d_kq
        values = torch.matmul(x, self.W_value) # 1 x num_tokens x d_v
         
        keys_transpose = keys.transpose(2, 1) # 1 x d_kq x num_tokens
        # unnormalized attention weights
        attn_scores = torch.matmul(queries, keys_transpose)  # 1 x num_tokens x num_tokens
        block_size = attn_scores.shape[1]
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).cuda()
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(
            masked / self.head_dim**0.5, dim=-1 # scale the dot product and apply softmax
        ) # shape is 1 x num_tokens x num_tokens
        # Apply dropout 
        attn_weights = self.dropout(attn_weights)
        context_vec = torch.matmul(attn_weights, values) # 1 x num_tokens x d_v
        return context_vec

def main(args):
    if args.self_attn:
        num_tokens = 6; d_in = 4; head_dim = 8
        self_att = SelfAttention(d_in, head_dim).eval().cuda()

        input_seq = torch.randn((1, num_tokens, d_in)).cuda()
        context_output = self_att(input_seq) # Output shape = 1 x num_tokens x d_v
        print(f"num_tokens: {num_tokens}, input d_in: {d_in}, head_dim: {head_dim}")
        print("Input sequence shape: ", input_seq.shape)
        print("Self attention output: ", context_output.shape)
        measure_time(self_att, input_seq, "Self attention")
    
    elif args.causal_attn:
        num_tokens = 6; d_in = 4; head_dim = 8; dropout=0.2
        causal_att = CausalAttention(d_in, head_dim, dropout).eval().cuda()

        input_seq = torch.randn((1, num_tokens, d_in)).cuda()
        context_output = causal_att(input_seq) # Output shape = 1 x num_tokens x d_v
        print(f"num_tokens: {num_tokens}, input d_in: {d_in}, head_dim: {head_dim}, dropout: {dropout}")
        print("Input sequence shape: ", input_seq.shape)
        print("Causal attention output: ", context_output.shape)
        measure_time(causal_att, input_seq, "Causal attention")
    else:
        raise ValueError(f"Invalid attention type provided. Select among --self_attn, --causal_attn")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run different types of attention. Options include SelfAttention, CausalAttention"
    )
    arg_parser.add_argument(
        "--self_attn",
        action="store_true",
        help="Boolean flag to run SelfAttention",
    )
    arg_parser.add_argument(
        "--causal_attn",
        action="store_true",
        help="Boolean flag to run CausalAttention",
    )
    args = arg_parser.parse_args()
    main(args)
