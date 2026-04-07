"""
Compute the optimal image-size schedule for SD3 denoising under a fixed compute budget.

Reads resize_experiment_results_scaled_sigma.pkl and uses dynamic programming to find,
for each of the 20 timesteps, which image size minimises the total upsampled-latent MSE
while staying within a given compute budget.

Compute cost model (single forward pass at image size `size`):
  n     = size // 16            # spatial dim of latent after SD3's patch_size=2
  seq   = n**2 + n_txt          # total sequence length (image tokens + text tokens)
  ff    = 24 * seq * d**2       # feed-forward FLOPs (proportional)
  attn  = 4 * seq**2 * d        # attention FLOPs (proportional)
  cost  = ff + attn             # total per-step cost

All costs are normalised so that the cost of a full-resolution (512 px) step equals 1.0,
and the budget is expressed as a percentage of N_STEPS x cost(512).
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RESULTS_PATH = "results/resize_experiment_dit_imagenet1k_scaled_sigma.pkl"
METRIC = "upsampled_latent"
USE_SNR = True
N_TXT = 60    # SD3: maximally T5(77) + CLIP(77) context tokens
D = 1536      # SD3-medium hidden dim (only relative scale matters)

BUDGETS_PCT = list(range(10, 101, 10))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def step_cost(size: int) -> float:
    """Unnormalised compute cost for one transformer forward pass at `size` pixels."""
    n = size // 16  # spatial dim after patch_size-2 embedding
    seq = n ** 2 + N_TXT
    ff_cost = 24 * seq * D ** 2
    attn_cost = 4 * seq ** 2 * D
    return float(ff_cost + attn_cost)


def build_cost_table(sizes: list[int]) -> dict[int, float]:
    """Return cost-per-step for each size, normalised to cost(512) = 1.0."""
    raw = {s: step_cost(s) for s in sizes}
    norm = raw[512]
    return {s: c / norm for s, c in raw.items()}


# ---------------------------------------------------------------------------
# MSE table
# ---------------------------------------------------------------------------
def build_mse_table(
    results: dict,
    sizes: list[int],
    t_values: list[int],
) -> np.ndarray:
    """
    Return MSE[size_idx, t_idx] averaged over samples, weighted by SNR gap.

    Weight each timestep's MSE by the signal-to-noise ratio so that high-noise
    steps (where SNR is low and errors matter less for final image quality)
    contribute less to the objective.

    SNR(t) = (1 - sigma(t))^2 / sigma(t)^2

    The correct ELBO weight for each step i is the SNR gap:
        w_i = SNR(t_{i-1}) - SNR(t_i)
    where t_{i-1} is the previous (less noisy) timestep.  This is the finite-
    difference form of the standard continuous-time ELBO weight for the x0-
    prediction parameterisation.
    """
    data = results[METRIC]
    mse = np.zeros((len(sizes), len(t_values)))
    for si, size in enumerate(sizes):
        for ti, t in enumerate(t_values):
            mse[si, ti] = float(np.mean(data[size][t]))

    # SD3 uses a linear sigma schedule: sigma(t_idx) = t_idx / (num_steps - 1)
    num_steps = max(t_values) + 1
    t_arr = np.array(t_values, dtype=float)
    sigma = t_arr / (num_steps - 1)

    def snr(s):
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return (1 - s) ** 2 / s ** 2

    t_prev_arr = np.maximum(t_arr - 1, 0)
    sigma_prev = t_prev_arr / (num_steps - 1)
    delta_snr = snr(sigma_prev) - snr(sigma)   # always > 0 since SNR decreases with sigma

    # Normalise so weights average to 1 (keeps MSE magnitudes interpretable)
    delta_snr = delta_snr / delta_snr.mean()
    mse = mse * delta_snr[np.newaxis, :]

    return mse


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------
def dp_optimal_schedule(
    mse: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> tuple[list[int], float]:
    """
    Find the sequence of size indices (one per timestep) that minimises sum(MSE)
    subject to sum(cost) <= budget, using dynamic programming.

    Args:
        mse:    shape (n_sizes, n_steps) — mean MSE[size_idx, step_idx]
        costs:  shape (n_sizes,)         — normalised cost per step per size
        budget: total allowed cost (in same units as costs)

    Returns:
        (size_indices, total_mse) — optimal size index per step and objective value
    """
    n_sizes, n_steps = mse.shape

    # Discretise budget into integer "tokens" for exact DP.
    # We use resolution = 0.01 (i.e., 1% of a 512-step cost).
    resolution = 0.01
    B = int(round(budget / resolution))        # budget in tokens

    # dp[b] = minimum total MSE achievable using exactly b budget tokens
    #         for the first `step` steps.
    INF = float("inf")
    dp = np.full(B + 1, INF)
    dp[0] = 0.0
    # choice[step, b] = size_idx chosen at `step` when arriving at budget b
    choice = np.full((n_steps, B + 1), -1, dtype=np.int32)

    for step in range(n_steps):
        new_dp = np.full(B + 1, INF)
        for si in range(n_sizes):
            c = int(round(costs[si] / resolution))
            if c > B:
                continue
            # For each budget state b_prev where c tokens are available
            b_prev_max = B - c
            valid = dp[: b_prev_max + 1] < INF
            if not valid.any():
                continue
            b_prevs = np.where(valid)[0]
            b_nexts = b_prevs + c
            new_vals = dp[b_prevs] + mse[si, step]
            improve = new_vals < new_dp[b_nexts]
            new_dp[b_nexts[improve]] = new_vals[improve]
            choice[step, b_nexts[improve]] = si

        dp = new_dp

    # Find the best feasible budget state
    best_b = int(np.argmin(dp))
    if dp[best_b] == INF:
        raise ValueError("No feasible schedule found — budget too small.")

    # Backtrack
    size_indices = []
    b = best_b
    for step in range(n_steps - 1, -1, -1):
        si = choice[step, b]
        size_indices.append(si)
        c = int(round(costs[si] / resolution))
        b -= c
    size_indices.reverse()

    return size_indices, float(dp[best_b])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    sizes = sorted(results[METRIC].keys())
    t_values_all = sorted(next(iter(results[METRIC].values())).keys())

    # Sigma per step (linear schedule)
    num_steps_full = max(t_values_all) + 1
    t_arr_all = np.array(t_values_all, dtype=float)
    sigma_all = t_arr_all / (num_steps_full - 1)

    # Pin only the zero-noise step (sigma=0) to 512px and exclude from DP.
    boundary_mask = sigma_all == 0.0
    inner_indices = [i for i, b in enumerate(boundary_mask) if not b]
    t_values_inner = [t_values_all[i] for i in inner_indices]
    n_steps_inner = len(t_values_inner)

    cost_table = build_cost_table(sizes)
    costs_arr = np.array([cost_table[s] for s in sizes])
    mse_arr = build_mse_table(results, sizes, t_values_inner)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / (len(BUDGETS_PCT) - 1)) for i in range(len(BUDGETS_PCT))]

    fig, ax = plt.subplots(figsize=(10, 5))

    # budget_pct -> list of sizes (one per step, including pinned boundary steps)
    schedules: dict[int, list[int]] = {}

    for color, budget_pct in zip(colors, BUDGETS_PCT):
        budget_abs = n_steps_inner * budget_pct / 100.0
        try:
            size_indices, _ = dp_optimal_schedule(mse_arr, costs_arr, budget_abs)
        except ValueError:
            continue
        optimal_sizes = [sizes[si] for si in size_indices]

        # Build full sigma / size arrays including pinned boundary steps
        full_sigmas, full_sizes = [], []
        inner_iter = iter(optimal_sizes)
        for i, _ in enumerate(t_values_all):
            full_sigmas.append(sigma_all[i])
            full_sizes.append(512 if boundary_mask[i] else next(inner_iter))

        schedules[budget_pct] = full_sizes
        ax.plot(full_sigmas, full_sizes, color=color, linewidth=1.8, label=f"{budget_pct}%")

    ax.set_xlabel("sigma (noise level)")
    ax.set_ylabel("image size (px)")
    ax.set_title("Optimal size schedule vs. compute budget\n(metric=upsampled_latent, use_snr=True)")
    ax.invert_xaxis()  # sigma=0 (clean) on the right, sigma=1 (noise) on the left
    ax.set_yticks(sizes[::2])
    ax.legend(title="budget", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = "optimal_schedule_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

    json_path = "optimal_schedules.json"
    with open(json_path, "w") as f:
        json.dump({"t_values": t_values_all, "schedules": {str(k): v for k, v in schedules.items()}}, f, indent=2)
    print(f"Schedules saved to {json_path}")


if __name__ == "__main__":
    main()
