import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

N = 10000
H = W = 32 * 4

# Sample gaussian noise: (N, 1, H, W)
x = torch.randn(N, 1, H, W)

sizes = list(range(2, H, 2))  # 2, 4, 6, ..., 30
print(sizes)

fig, axes = plt.subplots(6, 10, figsize=(15, 9), sharey=False)
axes = axes.flatten()

for ax, s in zip(axes, sizes):
    downsampled = F.interpolate(x, size=(s, s), mode='area')
    # Per-pixel std across N samples: shape (s, s)
    # This shows which pixel positions are more/less variable due to the interpolation pattern
    pixel_std = downsampled[:, 0, :, :].std(dim=0)
    im = ax.imshow(pixel_std.numpy(), vmin=0, vmax=1, cmap='viridis')
    ax.set_title(f"{s}x{s}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "Distribution of intra-image spatial std after bilinear downsampling\n"
    "of 32x32 Gaussian noise (align_corners=True, N=10000)",
    fontsize=13,
)
plt.tight_layout()
# plt.savefig("downsample_std_analysis.png", dpi=150)
plt.show()
print("Saved to downsample_std_analysis.png")
