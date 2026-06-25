"""Benchmark: summed-area-table (SAT) kernel vs. direct per-window kernel.

Shows that SAT wall-clock is ~flat in radius while the direct kernel grows
~radius^3. MI is excluded (compute_mi=False) so we isolate the MSE/CORR cost
that SAT actually accelerates.

    python benchmark_sat.py
"""

import time

import numpy as np

import r2d2_rust


def bench(shape=(91, 109, 91), radii=(2, 3, 5, 8), seed=0):
    rng = np.random.default_rng(seed)
    reg = np.ascontiguousarray(rng.normal(size=shape))
    tmplt = np.ascontiguousarray(reg + 0.3 * rng.normal(size=shape))
    mask = np.zeros(shape)
    mask[5:-5, 5:-5, 5:-5] = 1.0

    print(f"shape={shape}, masked voxels={int(mask.sum()):,}\n")
    print(f"{'radius':>6} | {'direct (s)':>11} | {'SAT (s)':>9} | {'speedup':>8}")
    print("-" * 44)
    for r in radii:
        # compute_mi=False isolates the metrics SAT accelerates.
        t0 = time.perf_counter()
        r2d2_rust.compute_r2d2(reg, tmplt, mask, r, compute_mi=False, use_sat=False)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2d2_rust.compute_r2d2(reg, tmplt, mask, r, compute_mi=False, use_sat=True)
        t_sat = time.perf_counter() - t0

        print(f"{r:>6} | {t_direct:>11.3f} | {t_sat:>9.3f} | {t_direct / t_sat:>7.1f}x")


if __name__ == "__main__":
    bench()
