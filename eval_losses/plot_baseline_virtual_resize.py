import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

with open("eval_losses/eval_losses_results.json") as f:
    data = json.load(f)

trajectory = data["baseline_virtual_resize/ema"]
sizes = list(trajectory.keys())
loss_types = list(next(iter(trajectory.values())).values().__iter__().__next__().__class__.__mro__)  # just get from data

# Determine loss types from first size/timestep
first_size_data = next(iter(trajectory.values()))
loss_types = list(next(iter(first_size_data.values())).keys())

colors = cm.tab10(np.linspace(0, 1, len(sizes)))

fig, axes = plt.subplots(1, len(loss_types), figsize=(6 * len(loss_types), 5))
if len(loss_types) == 1:
    axes = [axes]

for ax, loss_type in zip(axes, loss_types):
    for color, size in zip(colors, sizes):
        size_data = trajectory[size]
        timesteps = sorted(float(t) for t in size_data.keys())
        values = [size_data[f"{t:.4f}"][loss_type] for t in timesteps]
        ax.plot(timesteps, values, label=size, color=color, marker="o", markersize=3)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Loss")
    ax.set_title(f"baseline_virtual_resize/ema — {loss_type}")
    ax.legend(title="Size", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("baseline_virtual_resize_losses.png", dpi=150)
# print("Saved to baseline_virtual_resize_losses.png")

plt.show()