"""
exp_imitation.py -- target (b), imitation of the teacher's reconstruction map on
fresh real inputs, measured with the two-candidate (near-tie-safe) oracle.

Re-trains a subset of the E1 students (same data and init seeds as
exp_main.py 'local'), plus reconstruction students of the N = 1024 imperfect teacher,
keeps their parameters, and evaluates excess real-data risk, imitation error and R_Q
with project(..., candidates=2).  Writes results/imitation.json.

Run:  python3 exp_imitation.py      (~6 min with 4 workers)
"""
import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_main as E
from canon import Decoder, levenberg_marquardt, sample_P


def eval2(dec, th, Xte, Yte_ref, RP_ref):
    _, Dx = dec.project(th, Xte, candidates=2)
    RP = 0.5 * ((Xte - Dx) ** 2).sum(1).mean()
    imit = ((Dx - Yte_ref) ** 2).sum(1).mean()
    frac_diff = float((np.linalg.norm(Dx - Yte_ref, axis=1) > 1e-6).mean())
    _, Dy = dec.project(th, Yte_ref, candidates=2)
    RQ = 0.5 * ((Yte_ref - Dy) ** 2).sum(1).mean()
    return dict(excess=float(RP - RP_ref), imit=float(imit), frac_points_differing=frac_diff, RQ=float(RQ))


def teacher_N(N):
    """Re-create the imperfect teacher of exp_main.py 'imperfect' (deterministic)."""
    T = E.load_teacher()
    dec = Decoder(E.CFG["m"], E.CFG["d"])
    XN = sample_P(E.truth(), N, E.CFG["d"], E.CFG["sigma"], np.random.default_rng(31337 + N))
    thN, _, _ = levenberg_marquardt(dec, T["theta"].copy(), XN, iters=300)
    return thN


def job(args):
    kind, n, seed, i, th0, th_teacher = args
    T = E.load_teacher()
    dec = Decoder(E.CFG["m"], E.CFG["d"])
    Xn = sample_P(E.truth(), n, E.CFG["d"], E.CFG["sigma"], np.random.default_rng(50000 + 1000 * seed + n))
    _, Yn = dec.project(th_teacher, Xn)
    Xte = T["Xte"]
    _, Yref = dec.project(th_teacher, Xte, candidates=2)          # the producing teacher's map
    RPref = 0.5 * ((Xte - Yref) ** 2).sum(1).mean()
    out = []
    for lab, data in (("raw", Xn), ("rec", Yn)):
        if kind == "imperfect" and lab == "raw":
            continue
        th, L, _ = levenberg_marquardt(dec, th0.copy(), data, iters=300)
        ev = eval2(dec, th, Xte, Yref, RPref)
        out.append(dict(kind=kind, data=lab, n=n, seed=seed, init=i, train_loss=float(L), **ev))
    return out


if __name__ == "__main__":
    T = E.load_teacher()
    th_t = T["theta"]
    dec = Decoder(E.CFG["m"], E.CFG["d"])
    jobs = []
    for n in [32, 128, 512]:
        for seed in range(3):
            for i in range(4):
                rng = np.random.default_rng(9000 + seed + 100 * n + 7919 * i)   # = exp_main 'local'
                jobs.append(("local", n, seed, i, th_t + E.CFG["perturb"] * rng.normal(size=dec.p), th_t))
    th1024 = teacher_N(1024)
    for n in [128, 512]:
        for seed in range(2):
            for i in range(3):
                rng = np.random.default_rng(5000 + seed + 100 * n + 1024 + 7919 * i)  # = exp_main 'imperfect'
                jobs.append(("imperfect", n, seed, i, th1024 + E.CFG["perturb"] * rng.normal(size=dec.p), th1024))
    jobs.sort(key=lambda j: -j[1])
    with Pool(E.CFG["workers"]) as pool:
        res = [r for rr in pool.map(job, jobs) for r in rr]
    json.dump(res, open(os.path.join(E.RES, "imitation.json"), "w"), indent=1)
    for r in sorted(res, key=lambda r: (r["kind"], r["n"], r["data"], r["seed"], r["init"])):
        print(f"{r['kind']:9s} n={r['n']:4d} {r['data']} s{r['seed']}i{r['init']}  train {r['train_loss']:.1e}  "
              f"excess {r['excess']:+.1e}  imit {r['imit']:.1e}  frac_diff {r['frac_points_differing']:.1e}  RQ {r['RQ']:.1e}")
