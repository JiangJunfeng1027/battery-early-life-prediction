"""Selective extraction for mechanism figures: dQ(V) curves + fade summaries."""
import h5py, numpy as np, sys

ROOT = "."
OUT  = "./cache/figcache_2018.npz"

FILES = [
    ("b12", ROOT + "/data/MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat"),
    ("b3",  ROOT + "/data/MIT/2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
]

cells = []  # dicts: key, cl, q10, q100, qd_summary
for prefix, path in FILES:
    f = h5py.File(path, "r"); batch = f["batch"]
    n = batch["summary"].shape[0]
    print(prefix, "cells:", n)
    for i in range(n):
        try:
            cl = float(f[batch["cycle_life"][i, 0]][()].flatten()[0])
        except Exception:
            cl = np.nan
        # summary discharge capacity
        try:
            summ = f[batch["summary"][i, 0]]
            qd = np.asarray(summ["QDischarge"]).flatten()
        except Exception:
            qd = np.array([])
        # Qdlin at cycle idx 9 and 99 (cycle 10 and 100)
        q10 = q100 = None
        try:
            cyc = f[batch["cycles"][i, 0]]
            refs = cyc["Qdlin"]
            if refs.shape[0] > 99:
                q10  = np.asarray(f[refs[9, 0]][()]).flatten()
                q100 = np.asarray(f[refs[99, 0]][()]).flatten()
        except Exception:
            pass
        cells.append(dict(key=f"{prefix}c{i}", cl=cl,
                          q10=q10, q100=q100, qd=qd))
    f.close()

# keep cells with valid label and both curves
keep = [c for c in cells if np.isfinite(c["cl"]) and c["cl"] > 0
        and c["q10"] is not None and c["q100"] is not None]
print("valid:", len(keep), "of", len(cells))

np.savez_compressed(
    OUT,
    keys=np.array([c["key"] for c in keep]),
    cl=np.array([c["cl"] for c in keep]),
    q10=np.stack([c["q10"] for c in keep]),
    q100=np.stack([c["q100"] for c in keep]),
    qd=np.array([c["qd"] for c in keep], dtype=object),
)
print("saved", OUT)
