import torch

def farey_sequence(n: int) -> list[tuple[int, int]]:
    """Farey sequence F_n as (numerator, denominator) pairs, including 0/1 and 1/1."""
    fracs = [(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        fracs.append((c, d))
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    return fracs


def build_atomic_intervals(n: int) -> tuple[list[tuple[int, int, int, int]], torch.Tensor]:
    """
    Atomic intervals from F_n.  Each consecutive Farey pair (p1/q1, p2/q2)
    spans width 1/(q1*q2) by the Farey neighbour property.

    Returns:
        intervals : list of (p1, q1, p2, q2)
        widths    : float32 tensor of shape (J,)
    """
    fracs = farey_sequence(n)
    intervals, widths = [], []
    for (p1, q1), (p2, q2) in zip(fracs, fracs[1:]):
        intervals.append((p1, q1, p2, q2))
        widths.append(1.0 / (q1 * q2))
    return intervals, torch.tensor(widths, dtype=torch.float32)


def build_masks(n: int, intervals: list, J: int) -> list[torch.Tensor]:
    """
    For each resolution k = 1..n, a binary mask of shape (k, J).
    mask[i, j] = 1  iff  atomic interval j falls in bin [i/k, (i+1)/k).

    Bin assignment: because k ≤ n, every bin boundary i/k is itself a
    Farey fraction, so atomic intervals never straddle bin boundaries.
    The containing bin is simply  i = floor(p1 * k / q1)  (integer division).
    """
    masks = []
    for k in range(1, n + 1):
        mask = torch.zeros(k, J)
        for j, (p1, q1, *_) in enumerate(intervals):
            mask[(p1 * k) // q1, j] = 1.0
        masks.append(mask)
    return masks


def build_bin_indices(n: int, intervals: list) -> torch.Tensor:
    """
    Precompute bin assignments for all resolutions.

    Returns:
        bin_idx : LongTensor of shape (n, J) where bin_idx[k-1, j] is the
                  bin index of atomic interval j at resolution k.
    """
    J = len(intervals)
    bin_idx = torch.empty(n, J, dtype=torch.long)
    for j, (p1, q1, *_) in enumerate(intervals):
        for k in range(1, n + 1):
            bin_idx[k - 1, j] = (p1 * k) // q1
    return bin_idx


def sample_noise_fields(n: int, d: int, b: int) -> list[torch.Tensor]:
    """
    Sample a consistent family of noise fields at resolutions 1..n.

    The underlying model is a continuous Gaussian field on [0,1] whose
    Farey-interval decomposition gives mutually consistent discretisations.

    Each atomic interval j contributes  a_j ~ N(0, w_j)  to every bin
    that contains it.  No sqrt(k) rescaling is applied, so the variance
    of a bin at resolution k is  1/k  (sum of widths it contains).

    Args:
        n : maximum resolution
        d : field dimensionality (stacked independent fields)
        b : batch size

    Returns:
        List of length n.  Entry k-1 has shape (b, d, k).
    """
    intervals, widths = build_atomic_intervals(n)
    J = len(intervals)
    bin_idx = build_bin_indices(n, intervals)  # (n, J)

    # a_j ~ N(0, w_j)  =>  a = eps * sqrt(w),   shape (b, d, J)
    atomic = torch.randn(b, d, J) * widths.sqrt()

    fields = []
    for k in range(1, n + 1):
        idx = bin_idx[k - 1]  # (J,)
        field = torch.zeros(b, d, k)
        field.scatter_add_(2, idx.expand(b, d, -1), atomic)
        fields.append(field)
    return fields


def sample_noise_fields_2d(n: int, d: int, b: int, chunk: int = 512) -> list[torch.Tensor]:
    """
    Sample a consistent family of 2D noise fields at resolutions 1..n.

    Uses streaming over chunks of 2D atomic interval pairs (jx, jy) so that
    memory stays at O(chunk * b * d + sum of k²) rather than O(J² * b * d).
    Each 2D atomic rectangle has variance  w_jx * w_jy  and is assigned to
    bin (ix, iy) at resolution k via the 1D bin indices.

    Args:
        n     : maximum resolution
        d     : field dimensionality (channels)
        b     : batch size
        chunk : number of 2D atomic pairs to process at once

    Returns:
        List of length n.  Entry k-1 has shape (b, d, k, k).
    """
    intervals, widths = build_atomic_intervals(n)
    J = len(intervals)
    bin_idx = build_bin_indices(n, intervals)  # (n, J)

    # Pre-allocate output fields
    fields = [torch.zeros(b, d, k, k) for k in range(1, n + 1)]

    # Stream over chunks of (jx, jy) pairs
    for jx_start in range(0, J, chunk):
        jx_end = min(jx_start + chunk, J)
        chunk_x = jx_end - jx_start
        w_x = widths[jx_start:jx_end]                        # (chunk_x,)
        bin_x = bin_idx[:, jx_start:jx_end]                   # (n, chunk_x)

        for jy_start in range(0, J, chunk):
            jy_end = min(jy_start + chunk, J)
            chunk_y = jy_end - jy_start
            w_y = widths[jy_start:jy_end]                     # (chunk_y,)
            bin_y = bin_idx[:, jy_start:jy_end]                # (n, chunk_y)

            # 2D atomic variances: w_jx * w_jy, shape (chunk_x, chunk_y)
            w_2d = w_x[:, None] * w_y[None, :]                # (cx, cy)

            # Sample independent noise for this chunk: (b, d, cx, cy)
            noise = torch.randn(b, d, chunk_x, chunk_y) * w_2d.sqrt()

            # Scatter into each resolution
            for k in range(1, n + 1):
                bx = bin_x[k - 1]   # (chunk_x,)  values in [0, k)
                by = bin_y[k - 1]   # (chunk_y,)  values in [0, k)

                # Flat 2D bin index: ix * k + iy, shape (cx, cy)
                flat_idx = bx[:, None] * k + by[None, :]      # (cx, cy)

                # Flatten spatial dims and scatter_add into (b, d, k*k)
                fields[k - 1].view(b, d, -1).scatter_add_(
                    2,
                    flat_idx.reshape(1, 1, -1).expand(b, d, -1),
                    noise.reshape(b, d, -1),
                )

    return fields


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    n, d, b = 16, 4, 6

    print(f"Sampling noise fields  n={n}  d={d}  b={b}")
    fields = sample_noise_fields(n, d, b)

    # Sanity: check shapes
    for k, f in enumerate(fields, 1):
        assert f.shape == (b, d, k), f"Wrong shape at k={k}: {f.shape}"
    print("Shape check passed.\n")

    # Consistency test for power-of-2 resolutions.
    #
    # Without the sqrt(k) rescaling the coarser level is the *sum* of its
    # two children, not their average:
    #
    #   fields[k][bin i]  =  fields[2k][bin 2i]  +  fields[2k][bin 2i+1]
    #
    # This is the defining property of the Farey field construction.

    print("Power-of-2 sum-consistency test  (fields[k] == sum of pairs in fields[2k]):")
    all_ok = True
    m = 1
    while 2 * (2 ** m) <= n:
        k      = 2 ** m
        coarse = fields[k - 1]           # (b, d, k)
        fine   = fields[2 * k - 1]       # (b, d, 2k)
        reconstructed = fine[..., 0::2] + fine[..., 1::2]
        ok = torch.allclose(reconstructed, coarse, atol=1e-5)
        mark = "✓" if ok else "✗"
        print(f"  {mark}  fields[{k:2d}] == sum_pairs(fields[{2*k:2d}])", end="")
        if not ok:
            all_ok = False
            print(f"   max_diff={( reconstructed - coarse).abs().max():.2e}", end="")
        print()
        m += 1

    print()
    print("All tests passed." if all_ok else "SOME TESTS FAILED.")

    # 2D consistency test
    print("\n2D noise field test (n=8):")
    fields_2d = sample_noise_fields_2d(8, d, b)
    for k, f in enumerate(fields_2d, 1):
        assert f.shape == (b, d, k, k), f"Wrong 2D shape at k={k}: {f.shape}"
    print("  Shape check passed.")

    # Check sum-consistency in 2D: coarse == sum of 2×2 blocks in fine
    print("  Power-of-2 sum-consistency (2D):")
    m = 1
    while 2 * (2 ** m) <= 8:
        k = 2 ** m
        coarse = fields_2d[k - 1]           # (b, d, k, k)
        fine   = fields_2d[2 * k - 1]       # (b, d, 2k, 2k)
        reconstructed = (fine[..., 0::2, 0::2] + fine[..., 0::2, 1::2] +
                         fine[..., 1::2, 0::2] + fine[..., 1::2, 1::2])
        ok = torch.allclose(reconstructed, coarse, atol=1e-5)
        mark = "✓" if ok else "✗"
        print(f"    {mark}  fields_2d[{k}] == sum_2x2(fields_2d[{2*k}])")
        m += 1

    # Report raw noise vector sizes for typical input resolutions
    print("\nRaw atomic noise vector sizes (J = |F_n| - 1) for typical input sizes:")
    for s in [64, 128, 256]:
        J = len(build_atomic_intervals(s)[0])
        print(f"  n={s:3d}  =>  J={J:,d}  (1D, √J={J**.5:.1f})    J²={J**2:,d}  (2D, √J²={J:,d})")