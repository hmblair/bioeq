from bioeq.mol._cpp import Polymer
import torch

x = torch.randn(10, 3)
y = torch.randint(0, 5, (10,))
poly = Polymer(x, y)

breakpoint()
