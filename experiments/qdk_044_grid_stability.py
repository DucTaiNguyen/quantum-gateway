import json
from pathlib import Path
from itertools import combinations
import numpy as np

OUT = Path("results/qdk_044")
OUT.mkdir(parents=True, exist_ok=True)

I = np.eye(2, dtype=complex)

X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

P = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
}

labels = ["I","X","Y","Z"]

OBS = []

for a in labels:
    for b in labels:
        name = a+b
        if name != "II":
            OBS.append((name, np.kron(P[a],P[b])))

I4 = np.eye(4, dtype=complex)


def kraus_phase(p):
    return [
        np.sqrt(1-p/2) * I,
        np.sqrt(p/2) * Z
    ]


def phase_channel(rho,p):

    out = np.zeros_like(rho)

    for K in kraus_phase(p):

        K4 = np.kron(K,I)

        out += K4 @ rho @ K4.conj().T

    tmp = out.copy()

    out = np.zeros_like(rho)

    for K in kraus_phase(p):

        K4 = np.kron(I,K)

        out += K4 @ tmp @ K4.conj().T

    return out


def nonlinear_phase(rho,p):
    return phase_channel(rho,p*p)


def rank(A):

    s = np.linalg.svd(
        A,
        compute_uv=False
    )

    if len(s) == 0:
        return 0

    threshold = (
        max(A.shape)
        * s[0]
        * np.finfo(float).eps
    )

    return int(np.sum(s > threshold))


def build_matrix(channel, grid):

    columns = []

    for _, B in OBS:

        values = []

        for p in grid:

            rp = channel(B,p)

            for _, O in OBS:

                values.append(
                    float(
                        np.real(
                            np.trace(O @ rp)
                        )
                    )
                )

        columns.append(values)

    return np.asarray(columns).T


def rows_for_subset(subset, n_grid):

    rows = []

    for j in range(n_grid):

        base = j * len(OBS)

        for i in subset:
            rows.append(base+i)

    return rows


def find_global_minimum(A, n_grid):

    for k in range(1,16):

        successful = []

        for subset in combinations(range(15),k):

            rows = rows_for_subset(
                subset,
                n_grid
            )

            if rank(A[rows,:]) == 15:
                successful.append(subset)

        if successful:
            return k, successful

    return None, []


GRIDS = {
    "5": np.linspace(0,1,5),
    "9": np.linspace(0,1,9),
    "11": np.linspace(0,1,11),
    "21": np.linspace(0,1,21),
    "41": np.linspace(0,1,41),
    "81": np.linspace(0,1,81),
}

CHANNELS = {
    "phase_damping_cptp": phase_channel,
    "nonlinear_phase_cptp": nonlinear_phase,
}

results = {
    "experiment": "QDK-044",
    "title": "Grid Stability of CPTP Fingerprint Minimality",
    "results": {}
}

for cname, channel in CHANNELS.items():

    print()
    print("="*70)
    print(cname)
    print("="*70)

    results["results"][cname] = {}

    for gname, grid in GRIDS.items():

        print(
            f"Grid {gname}: {len(grid)} points"
        )

        A = build_matrix(
            channel,
            grid
        )

        full_rank = rank(A)

        kmin, subsets = find_global_minimum(
            A,
            len(grid)
        )

        print(
            "  full rank:",
            full_rank
        )

        print(
            "  global minimum k:",
            kmin
        )

        results["results"][cname][gname] = {
            "points": len(grid),
            "full_rank": full_rank,
            "global_minimum_k": kmin,
            "minimal_subset_count": len(subsets),
            "minimal_subsets": [
                [OBS[i][0] for i in s]
                for s in subsets
            ]
        }

outfile = OUT / "QDK-044-GRID-STABILITY.json"

with open(outfile,"w") as f:
    json.dump(results,f,indent=2)

print()
print("="*70)
print("QDK-044 COMPLETE")
print("Saved:",outfile)
print("="*70)
