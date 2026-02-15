import torch

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

height = 512
width = 512
num_inference_steps = 50
compression_sizes = [64, 56, 48, 32, 16, 8, 4, 2, 1]
num_samples = 64

# 20 evenly spaced timestep indices
t_indices = torch.linspace(0, num_inference_steps - 1, 20).long()

experiment_configs = [
    ("sd3", False),
    ("flux", False),
    ("auraflow", False),
]
