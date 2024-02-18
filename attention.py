import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v):
        super().__init__()
        self.d_kq = d_kq
        # d_in is the input embedding size of each token
        # d_kq is the key/query matrix output size
        # d_v is the value matrix output size
        self.W_query = nn.Parameter(torch.rand(d_in, d_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_v))
    
    def forward(self, x):
        keys = torch.matmul(x, self.W_key) # 1 x num_tokens x d_kq
        queries = torch.matmul(x, self.W_query) # 1 x num_tokens x d_kq
        values = torch.matmul(x, self.W_value) # 1 x num_tokens x d_v
         
        keys_transpose = keys.transpose(2, 1) # 1 x d_kq x num_tokens
        # unnormalized attention weights
        attn_scores = torch.matmul(queries, keys_transpose)  # dot product of queries and keys 
        attn_weights = torch.softmax(
            attn_scores / self.d_kq**0.5, dim=-1 # scale the dot product and apply softmax
        ) # shape is 1 x num_tokens x num_tokens

        context_vec = torch.matmul(attn_weights, values) # 1 x num_tokens x d_v
        return context_vec

num_tokens = 6; d_in = 4; d_kq = 8; d_v = 8
self_att = SelfAttention(d_in, d_kq, d_v).eval().cuda()

input_seq = torch.randn((1, num_tokens, d_in)).cuda()
context_output = self_att(input_seq) # Output shape = 1 x num_tokens x d_v
print("Self attention output: ", context_output.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_in, d_kq, d_v).eval().cuda() 
                                    for _ in range(num_heads)])
        
    def forward(self, x):
        # concatenate across d_v dimension
        head_outputs = [head(x) for head in self.heads]
        return torch.cat(head_outputs, dim=-1) 

num_heads = 32
mha = MultiHeadAttention(d_in, d_kq, d_v, num_heads)
mha_output = mha(input_seq)
# Output shape of mha = 1 x num_tokens x (num_heads*d_v)
print("Multi head attention output: ", mha_output.shape, "num_heads: ", num_heads) 

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_kq, d_v):
        super().__init__()
        self.d_kq = d_kq
        # d_in is the input embedding size of each token
        # d_kq is the key/query matrix output size
        # d_v is the value matrix output size
        self.W_query = nn.Parameter(torch.rand(d_in, d_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_v))
    
    def forward(self, x):
        keys = torch.matmul(x, self.W_key) # 1 x num_tokens x d_kq
        queries = torch.matmul(x, self.W_query) # 1 x num_tokens x d_kq
        values = torch.matmul(x, self.W_value) # 1 x num_tokens x d_v
         
        keys_transpose = keys.transpose(2, 1) # 1 x d_kq x num_tokens
        # unnormalized attention weights
        attn_scores = torch.matmul(queries, keys_transpose)  # dot product of queries and keys 
        block_size = attn_scores.shape[1]
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).cuda()
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(
            masked / self.d_kq**0.5, dim=-1 # scale the dot product and apply softmax
        ) # shape is 1 x num_tokens x num_tokens

        context_vec = torch.matmul(attn_weights, values) # 1 x num_tokens x d_v
        return context_vec

num_tokens = 6; d_in = 4; d_kq = 8; d_v = 8
self_att = CausalAttention(d_in, d_kq, d_v).eval().cuda()

input_seq = torch.randn((1, num_tokens, d_in)).cuda()
context_output = self_att(input_seq) # Output shape = 1 x num_tokens x d_v
print("Causal attention output: ", context_output.shape)