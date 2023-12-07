# Created by scalers.ai for Dell
"""
A conversion dictionary for converting torch dtype strings
to actual torch dtypes.

This file should be used as a module, and the TORCH_DTYPES
variable should be imported in the following way:
from torch import TORCH_DTYPES
"""
import torch

TORCH_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "auto": "auto",
}
