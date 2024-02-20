import torch 

torch.manual_seed(456)

def online_softmax(input):
    # torch softmax as a reference
    expected_softmax = torch.softmax(input, dim=1)

    # 1st read, torch max output both indexes and values, we only want the values
    # we transpose it to get a vertical tensor
    row_max = torch.max(input, dim=1).values[:, None]
    print("input row max\n", row_max)
    # 2nd read
    input_safe = input - row_max
    print("Below we reduce values amplitude, that's the safe part of safe softmax")
    print("original 1st row input:\n", input[0, :], "safe softmax input 1st row:\n", input_safe[0, :])

    softmax_numerator = torch.exp(input_safe)
    # 3rd read
    normalizer_term = torch.sum(softmax_numerator, dim=1)[:, None]
    # 4th read
    naive_softmax = softmax_numerator / normalizer_term

    assert torch.allclose(naive_softmax, expected_softmax)

row_count, col_count = 4, 16
input = torch.rand((row_count, col_count))
online_softmax(input)