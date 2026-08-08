"""
More information on LayerNorm-free transformers can be found on the paper 
`Stronger Normalization-Free Transformers`:

- Paper: https://arxiv.org/abs/2512.10938
- github: https://github.com/zlab-princeton/Derf
"""
from .dynamic_tanh import DynamicTanh
from .linear_clip import LinearClip
