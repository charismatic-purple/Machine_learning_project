import torch
from src.nn_models import MLPBaseline, MLPRegularized, MLPWide

x = torch.randn(4, 64)  # batch size 4, input_dim 64

print(MLPBaseline(64)(x).shape)      # should be (4, 3)
print(MLPRegularized(64)(x).shape)   # should be (4, 3)
print(MLPWide(64)(x).shape)          # should be (4, 3)