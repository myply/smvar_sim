import torch.nn as nn
import torch
class Slice(nn.Module):
    # Focus wh information into c-space
    def __init__(self):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
