import torch

x = torch.randn(3, 4)
print(x)

# Compute scale based on actual data range (symmetric, INT8)
max_val = x.abs().max()
scale = max_val / 127
zero_point = 0

y = torch.quantize_per_tensor(x, scale=scale.item(), zero_point=zero_point, dtype=torch.qint8)
print(y)

z = y.dequantize()
print(z)
