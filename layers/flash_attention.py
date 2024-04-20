import torch
import numpy as np
import time

def tiled_matmul(queries, keys, Br, Bc):
    num_tokens = queries.shape[0]
    d = queries.shape[1]
    output = torch.zeros((num_tokens, num_tokens)).cuda()

    for block_start_Bc in range(0, num_tokens, Bc):
        block_end_Bc = block_start_Bc + Bc
        Kj = keys[block_start_Bc:block_end_Bc, :]  # shape Bc x d = 8 x 8
        for block_start_Br in range(0, num_tokens, Br):
            block_end_Br = block_start_Br + Br
            Qi = queries[block_start_Br:block_end_Br, :]  # shape Br x d = 4 x 8

            # QKt at the tile level
            Sij = Qi @ Kj.T  # shape Br x Bc = 4 x 8
            output[block_start_Br:block_end_Br, block_start_Bc:block_end_Bc] += Sij
    
    return output

num_tokens = 6; d_kq = 8; d_v = 8
# tile size for matmul, no op bigger than this size can be stored in SRAM
Br = 4
Bc = 4
queries = torch.randn((6, 8)).cuda()
keys = torch.randn((6, 8)).cuda()
values = torch.randn((6, 8)).cuda()
tiled = tiled_matmul(queries, keys, Br, Bc)

# variables outside the for loop represent the global memory
# they are the only ones bigger than what the SRAM can store
O = torch.zeros((num_tokens, d_v)).cuda()

# For the 2 variables below, they may be removed in a serially executed code (in particular the outter for loop)
# They are needed in parallelized execution where each thread block need to sync its findings with the others
# l will store the denominator of the softmax for each row
l = torch.zeros((num_tokens, 1)).cuda()
# m will store the row max (computed progressively, block after block)
m = torch.full((num_tokens, 1), -torch.inf).cuda()

for block_start_Bc in range(0, num_tokens, Bc):
    block_end_Bc = block_start_Bc + Bc
    # load a block from matmul input tensor
    Kj = keys[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = values[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    for block_start_Br in range(0, num_tokens, Br):
        block_end_Br = block_start_Br + Br

        # line 8, load stuff from globabl memory, aka the work of the other thread blocks
        mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
        li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
        Qi = queries[block_start_Br:block_end_Br, :]  # shape Br x d

        # line 9, QKt at the tile level
        Sij = Qi @ Kj.T  # shape Br x Bc

        # line 10, find max of each row of the current loaded block (and only this block)
        mij_hat = torch.max(Sij, dim=1).values[:, None]
        # line 10, compute the softmax numerator like if we only had the data from this block (and nothing before or after)
        pij_hat = torch.exp(Sij - mij_hat)
        # line 10, compute the softmax denominator like if we only had the data from this block (and nothing before or after)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        # line 11, find max of each row regarding the current block and all the previous ones we have already visited
        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
        # line 11, adjusting factor (see online softmax computation above) leveraging the rule of exponentiation
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # line 12, first part before the "+" is the adjustment of the past blocks
        # second part after the "+" is the incorporation of the information from the current block and the matmul for this block
        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        # Note that we replace (=) and not update (+=) the global variables like we would do in tilted matmul
        # line 13, save statistics
        m[block_start_Br:block_end_Br, :] = mi_new  # row max
        l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator
        # save attention block to global memory
        O[block_start_Br:block_end_Br, :] = Oi