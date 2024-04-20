import torch


def regular_matmul(queries, keys):
    B = keys.T
    M, K = queries.shape
    K, N = B.shape
    output = torch.zeros((M, N)).cuda()
    for i in range(M):
        for j in range(N):
            for k in range(K):
                output[i][j] += queries[i][k] * B[k][j]
    
    return output

def tiled_matmul(M, N, K):
    # for simplification tile shapes are all multiple of matrix shapes
    # otherwise we would need to check matrix bounds and mask out of bounds values by 0s in tiles
    block_M, block_N, block_K = M // 3, N // 3, K // 2 # 5, 3, 6 respectively

    output = torch.zeros((M, N))
    total_load = 0
    total_write = 0

    for index_M in range(0, M, block_M):
        start_M = index_M
        end_M = index_M + block_M

        for index_N in range(0, N, block_N):
            start_N = index_N
            end_N = index_N + block_N
            accumulator = torch.zeros((block_M, block_N))
            for index_K in range(0, K, block_K):
                start_K = index_K
                end_K = index_K + block_K

                tile_A = A[start_M:end_M, start_K:end_K] # 5 x 6
                total_load += tile_A.numel() # 30
                tile_B = B[start_K:end_K, start_N:end_N] # 6 x 3
                total_load += tile_B.numel()
                # @ means matmul in numpy and pytorch
                accumulator += tile_A @ tile_B # 5 x 3
            output[start_M:end_M, start_N:end_N] = accumulator
            total_write += accumulator.numel() # 5 x 3

    assert torch.allclose(output, A @ B)
    print("total elements load from global memory:", total_load)
    print("total elements write to global memory:", total_write)


M, N, K = 15, 9, 12

A = torch.rand((M, K))
B = torch.rand((K, N))
tiled_matmul(M, N, K)