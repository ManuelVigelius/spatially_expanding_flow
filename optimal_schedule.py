"""
Compute the optimal image-size schedule for SD3 denoising under a fixed compute budget.

Reads resize_experiment_results_scaled_sigma.pkl (or the unscaled variant) and uses
dynamic programming to find, for each of the 20 timesteps, which image size minimises
the total upsampled-latent MSE while staying within a given compute budget.

Compute cost model (single forward pass at image size `size`):
  n     = size // 16            # spatial dim of latent after SD3's patch_size=2
  seq   = n**2 + n_txt          # total sequence length (image tokens + text tokens)
  ff    = 24 * seq * d**2       # feed-forward FLOPs (proportional)
  attn  = 4 * seq**2 * d        # attention FLOPs (proportional)
  cost  = ff + attn             # total per-step cost

All costs are normalised so that the cost of a full-resolution (512 px) step equals 1.0,
and the budget is expressed as a percentage of N_STEPS × cost(512).

Usage:
  python optimal_schedule.py [--budget 50] [--metric upsampled_latent]
                             [--results resize_experiment_results_scaled_sigma.pkl]
                             [--use-snr] [--n-txt 154] [--d 1536]
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
N_TXT_DEFAULT = 60   # SD3: maximally T5(77) + CLIP(77) context tokens
D_DEFAULT = 1536      # SD3-medium hidden dim (only relative scale matters)


def step_cost(size: int, n_txt: int = N_TXT_DEFAULT, d: int = D_DEFAULT) -> float:
    """Unnormalised compute cost for one transformer forward pass at `size` pixels."""
    n = size // 16  # spatial dim after patch_size-2 embedding
    seq = n ** 2 + n_txt
    ff_cost = 24 * seq * d ** 2
    attn_cost = 4 * seq ** 2 * d
    return float(ff_cost + attn_cost)


def build_cost_table(
    sizes: list[int], n_txt: int = N_TXT_DEFAULT, d: int = D_DEFAULT
) -> dict[int, float]:
    """Return cost-per-step for each size, normalised to cost(512) = 1.0."""
    raw = {s: step_cost(s, n_txt, d) for s in sizes}
    norm = raw[512]
    return {s: c / norm for s, c in raw.items()}


# ---------------------------------------------------------------------------
# MSE table
# ---------------------------------------------------------------------------
def build_mse_table(
    results: dict,
    metric: str,
    sizes: list[int],
    t_values: list[int],
    use_snr: bool,
) -> np.ndarray:
    """
    Return MSE[size_idx, t_idx] averaged over samples.

    If use_snr=True, weight each timestep's MSE by the signal-to-noise ratio
    so that high-noise steps (where SNR is low and errors matter less for
    final image quality) contribute less to the objective.

    SNR(t) = (1 - sigma(t))^2 / sigma(t)^2

    The correct ELBO weight for each step i is the SNR gap:
        w_i = SNR(t_{i-1}) - SNR(t_i)
    where t_{i-1} is the previous (less noisy) timestep.  This is the finite-
    difference form of the standard continuous-time ELBO weight for the x0-
    prediction parameterisation.  Using SNR(t_i) directly (as is common but
    wrong) misses the step-spacing dependence.
    """
    data = results[metric]
    mse = np.zeros((len(sizes), len(t_values)))
    for si, size in enumerate(sizes):
        for ti, t in enumerate(t_values):
            mse[si, ti] = float(np.mean(data[size][t]))

    if use_snr:
        # SD3 uses a linear sigma schedule: sigma(t_idx) = t_idx / (num_steps - 1)
        # t_values are the sampled indices (e.g. 20 out of 50 steps).
        # For the boundary t_0 we use sigma = 0 → SNR = +inf, so the gap for the
        # first step is SNR(t_{-1}=0) - SNR(t_0).  We cap SNR at a finite value
        # corresponding to sigma_min = 1/(num_steps-1) to avoid division by zero.
        num_steps = max(t_values) + 1          # e.g. 50 for the 0..49 index range
        t_arr = np.array(t_values, dtype=float)
        sigma = t_arr / (num_steps - 1)        # linear schedule, exact

        def snr(s):
            s = np.clip(s, 1e-6, 1 - 1e-6)
            return (1 - s) ** 2 / s ** 2

        # Previous sigma: for step i, t_{i-1} is the immediately preceding index
        # in the full scheduler (t_values[i] - 1), not the previous sampled step.
        # This avoids the boundary blow-up from anchoring at sigma=0.
        # For the first sampled step (t_values[0]) the previous index is
        # t_values[0] - 1, which has a well-defined finite sigma.
        t_prev_arr = np.maximum(t_arr - 1, 0)   # step just before each sampled t
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
# Plot
# ---------------------------------------------------------------------------
def _plot_schedules(
    results, sizes, t_values_all, boundary_mask, sigma_all,
    metric, use_snr, n_txt, d,
) -> None:
    t_values_inner = [t for t, b in zip(t_values_all, boundary_mask) if not b]

    costs_arr = np.array([build_cost_table(sizes, n_txt, d)[s] for s in sizes])
    mse_arr = build_mse_table(results, metric, sizes, t_values_inner, use_snr)
    n_steps_inner = len(t_values_inner)

    budgets_pct = list(range(10, 101, 10))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / (len(budgets_pct) - 1)) for i in range(len(budgets_pct))]

    fig, ax = plt.subplots(figsize=(10, 5))

    for color, budget_pct in zip(colors, budgets_pct):
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

        ax.plot(full_sigmas, full_sizes, color=color, linewidth=1.8, label=f"{budget_pct}%")

    ax.set_xlabel("sigma (noise level)")
    ax.set_ylabel("image size (px)")
    ax.set_title(f"Optimal size schedule vs. compute budget\n(metric={metric}, use_snr={use_snr})")
    ax.invert_xaxis()  # sigma=0 (clean) on the right, sigma=1 (noise) on the left
    ax.set_yticks(sizes[::2])
    ax.legend(title="budget", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = "optimal_schedule_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="resize_experiment_results_scaled_sigma.pkl",
        help="Path to the experiment results pickle file.",
    )
    parser.add_argument(
        "--metric",
        default="upsampled_latent",
        choices=["velocity", "latent", "upsampled_latent"],
        help="Which MSE metric to optimise.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=50.0,
        help=(
            "Compute budget as a percentage of the cost of running all steps at 512px. "
            "E.g. 50 means half the full-resolution budget."
        ),
    )
    parser.add_argument(
        "--use-snr",
        action="store_true",
        help="Weight MSE by signal-to-noise ratio (penalise errors at low-noise steps more).",
    )
    parser.add_argument(
        "--n-txt",
        type=int,
        default=N_TXT_DEFAULT,
        help="Number of text tokens in the SD3 transformer sequence.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=D_DEFAULT,
        help="Hidden dimension of the transformer (only relative cost matters).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot optimal size schedules across a range of compute budgets.",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results, "rb") as f:
        results = pickle.load(f)

    sizes = sorted(results["velocity"].keys())
    t_values_all = sorted(next(iter(results["velocity"].values())).keys())

    # Sigma per step (linear schedule)
    num_steps_full = max(t_values_all) + 1
    t_arr_all = np.array(t_values_all, dtype=float)
    sigma_all = t_arr_all / (num_steps_full - 1)

    # Pin only the zero-noise step (sigma=0) to 512px and exclude from DP.
    # At sigma=0 the SNR diverges to infinity, so this step must always run
    # at full resolution.  The pure-noise step (sigma=1) has weight → 0 and
    # no singularity, so the DP is free to choose any resolution there.
    boundary_mask = sigma_all == 0.0
    inner_indices = [i for i, b in enumerate(boundary_mask) if not b]
    t_values = [t_values_all[i] for i in inner_indices]
    n_steps = len(t_values)

    # Build tables (only inner steps)
    cost_table = build_cost_table(sizes, args.n_txt, args.d)
    costs_arr = np.array([cost_table[s] for s in sizes])
    mse_arr = build_mse_table(results, args.metric, sizes, t_values, args.use_snr)

    # Full budget = inner steps only (boundary steps are free at 512px)
    full_budget = n_steps * 1.0
    budget_abs = full_budget * args.budget / 100.0

    print(f"Metric         : {args.metric}")
    print(f"Use SNR weight : {args.use_snr}")
    print(f"Budget         : {args.budget:.1f}% of full-res ({budget_abs:.2f} normalised units)")
    print(f"Inner steps    : {n_steps} (boundary steps pinned to 512px, excluded from DP)")
    print()

    # Run DP
    size_indices, total_mse = dp_optimal_schedule(mse_arr, costs_arr, budget_abs)

    # Reconstruct full schedule including pinned boundary steps
    optimal_sizes_inner = [sizes[si] for si in size_indices]
    full_schedule = []  # (t_idx, size) for all steps
    inner_iter = iter(optimal_sizes_inner)
    for i, t in enumerate(t_values_all):
        if boundary_mask[i]:
            full_schedule.append((t, 512))
        else:
            full_schedule.append((t, next(inner_iter)))

    actual_cost = sum(cost_table[s] for _, s in full_schedule)
    actual_cost_pct = 100.0 * actual_cost / (len(t_values_all) * 1.0)

    # Baseline and best-possible over inner steps only
    baseline_mse = sum(mse_arr[sizes.index(512), ti] for ti in range(n_steps))
    best_possible_mse = sum(mse_arr[:, ti].min() for ti in range(n_steps))

    # SNR weights for display
    t_arr = np.array(t_values, dtype=float)
    sigma_arr = t_arr / (num_steps_full - 1)

    def snr(s):
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return (1 - s) ** 2 / s ** 2

    t_prev_arr = np.maximum(t_arr - 1, 0)
    sigma_prev_arr = t_prev_arr / (num_steps_full - 1)
    delta_snr = snr(sigma_prev_arr) - snr(sigma_arr)
    delta_snr_norm = delta_snr / delta_snr.mean()

    print(f"{'Step':>4}  {'t_idx':>5}  {'Size':>6}  {'Cost%':>6}  {'sigma':>7}  {'SNR_w':>7}  {'MSE':>10}")
    print("-" * 62)
    inner_step = 0
    for i, (t, s) in enumerate(full_schedule):
        c_pct = 100.0 * cost_table[s]
        sigma_val = sigma_all[i]
        if boundary_mask[i]:
            print(f"{i:>4}  {t:>5}  {s:>4}px  {c_pct:>5.1f}%  {sigma_val:>7.4f}  {'(pin)':>7}  {'---':>10}")
        else:
            mse_val = mse_arr[sizes.index(s), inner_step]
            snr_w = delta_snr_norm[inner_step]
            print(f"{i:>4}  {t:>5}  {s:>4}px  {c_pct:>5.1f}%  {sigma_val:>7.4f}  {snr_w:>7.4f}  {mse_val:>10.5f}")
            inner_step += 1
    print("-" * 62)
    print(f"\nTotal MSE (optimal)  : {total_mse:.5f}")
    print(f"Total MSE (512px)    : {baseline_mse:.5f}")
    print(f"Total MSE (best poss): {best_possible_mse:.5f}")
    print(f"Actual compute used  : {actual_cost_pct:.1f}% of full-res budget (incl. pinned steps)")
    print(
        f"MSE vs full-res      : {100.0 * total_mse / baseline_mse:.1f}% "
        f"(+{100.0*(total_mse - baseline_mse)/baseline_mse:.1f}%)"
    )

    if args.plot:
        _plot_schedules(
            results, sizes, t_values_all, boundary_mask, sigma_all,
            args.metric, args.use_snr, args.n_txt, args.d,
        )


if __name__ == "__main__":
    main()
