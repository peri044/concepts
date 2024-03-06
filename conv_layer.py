import numpy as np
import torch 
import torch.nn.functional as F

def conv(input, filter, padding=1, stride=1):
    # Get all tensor sizes
    N, C, H, W = input.size()
    fK, fC, fH, fW = filter.size()

    # Num input channels should be same as filter channels
    assert fC == C 
    
    # Calculate output shape and define output array
    oH = (H-fH + 2*padding) // stride + 1
    oW = (W-fW + 2*padding) // stride + 1
    output = torch.zeros(N, fK, oH, oW).cuda()
    print("Output shape calculated: ", (N, fK, oH, oW))
    # Pad inputs, given input 1x3x224x224, pad=1, pad_input = 1x3x226x226
    pad_hor = torch.zeros(1, 3, 1, W).cuda()
    input_pad_hor = torch.cat([pad_hor, input, pad_hor], dim=2)
    pad_ver = torch.zeros(1, 3, H+2*padding, 1).cuda()
    padded_input = torch.cat([pad_ver, input_pad_hor, pad_ver], dim=3)

    for n in range(N): # Iterate over batch
        for k in range(fK): # Over num_output channels
            for h in range(oH): # Over output height
                for w in range(oW): # Over output width
                    for c in range(C): # Over input channels
                        for fh in range(fH): # Over filter height
                            for fw in range(fW): # Over filter width
                                output[n][k][h][w] += padded_input[n][c][h+fh][w+fw]*filter[k][c][fh][fw]

    return output

input = torch.randn(1, 3, 32, 32).cuda()
filter = torch.randn(4, 3, 3, 3).cuda()
pad, stride = 1, 1
output = conv(input, filter, pad, stride)
print("Output of convolution: ", output)

output_pyt = F.conv2d(input, filter, padding=pad, stride=stride)
print("Output of pytorch convolution: ", output_pyt)
print("Difference b/w mine and Pytorch convolution: ", torch.mean(output_pyt-output))

