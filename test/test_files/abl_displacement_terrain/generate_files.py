#!/usr/bin/env python3
"""Generate input files for the displacement-height regression tests.

All grid files use the TerrainDrag flat-grid format:
    nx
    ny
    x[0..nx-1]
    y[0..ny-1]
    value[i*ny + j]   (x-major, matching ioutils::read_flat_grid_file)

Usage: python3 generate_files.py <flat|terrain> <output_dir>
"""
import math
import sys

LX = LY = 2048.0
N = 17
HILL_H, HILL_R = 150.0, 300.0  # Gaussian hill height / half-width (m)
TREE_H = 20.0  # canopy height (m); d = 2/3 h, z0 = 0.1 h
XC = YC = 0.5 * LX


def write_grid(fname, func):
    xs = [i * LX / (N - 1) for i in range(N)]
    ys = [j * LY / (N - 1) for j in range(N)]
    with open(fname, "w") as f:
        f.write(f"{N}\n{N}\n")
        f.writelines(f"{x:.6f}\n" for x in xs)
        f.writelines(f"{y:.6f}\n" for y in ys)
        for x in xs:
            for y in ys:
                f.write(f"{func(x, y):.6f}\n")


def forest(x, y):
    """Forest patch covering the middle third of the domain in x."""
    return 1.0 if LX / 3.0 <= x <= 2.0 * LX / 3.0 else 0.0


def write_rans(fname, ustar=0.5, z0=0.1, d=0.0, kappa=0.41, cmu=0.556):
    """1-D RANS profile: z u v w tke (required by KransAxell)."""
    with open(fname, "w") as f:
        for k in range(41):
            z = 50.0 * k
            zd = max(z - d, math.e * z0)
            u = ustar / kappa * math.log(zd / z0)
            tke = (ustar / cmu) ** 2
            f.write(f"{z:.3f} {u:.6f} 0.0 0.0 {tke:.6f}\n")


def main():
    case, out = sys.argv[1], sys.argv[2]
    if case == "flat":
        height = lambda x, y: 0.0
    else:
        height = lambda x, y: HILL_H * math.exp(
            -((x - XC) ** 2 + (y - YC) ** 2) / (2.0 * HILL_R**2))
    write_grid(f"{out}/terrain.amrwind", height)
    write_grid(f"{out}/terrain.displacement",
               lambda x, y: 2.0 / 3.0 * TREE_H * forest(x, y))
    write_grid(f"{out}/terrain.roughness",
               lambda x, y: 0.1 + (0.1 * TREE_H - 0.1) * forest(x, y))
    write_rans(f"{out}/rans_1d.info")


if __name__ == "__main__":
    main()
