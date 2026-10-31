"""
exp_trap.py -- obstruction experiment: a capacity-limited canon must break a
symmetry of the data; inequivalent optimal canons then become *traps* for
gradient training on the teacher's reconstructions, but not for raw training.

Data P: noisy superellipse |x|^4+|y|^4 = 1 in R^2 (dihedral symmetry D_4), sigma = 0.04.
Decoder: D_theta(t) = W tanh(A u(t) + b) + c with m = 3 hidden units (a closed
curve that cannot be D_4-symmetric), so the optimum breaks the symmetry and
g.theta_* (W -> gW, c -> gc, g in D_4) are equally good but inequivalent canons.

Run:  python3 exp_trap.py
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canon import Decoder, adam, levenberg_marquardt

RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
M_HID, D_OUT, SIGMA = 3, 2, 0.04


def sample_square(n, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    c, s = np.cos(t), np.sin(t)
    r = (np.abs(c) ** 4 + np.abs(s) ** 4) ** (-0.25)
    xi = rng.normal(0, 1, (n, 2))
    nrm = np.linalg.norm(xi, axis=1, keepdims=True)
    xi = np.where(nrm > 3.0, xi * 3.0 / nrm, xi)
    return np.stack([r * c, r * s], 1) + SIGMA * xi


def D4():
    gs = []
    for k in range(4):
        a = k * np.pi / 2
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        gs.append(R)
        gs.append(R @ np.diag([1.0, -1.0]))
    return gs


def act(dec, g, th):
    P = dec.unpack(th)
    return dec.pack(P["A"], P["b"], g @ P["W"], g @ P["c"])


def risk(dec, th, X):
    return dec.loss(th, X)[0]


def _job(args):
    tag, th0, data_seed, teacher, n = args
    dec = Decoder(M_HID, D_OUT)
    X = sample_square(n, np.random.default_rng(data_seed))
    _, Y = dec.project(teacher, X)
    Xte = sample_square(40000, np.random.default_rng(99))
    _, Yte = dec.project(teacher, Xte)
    RPt = risk(dec, teacher, Xte)
    out = {"tag": tag, "n": n, "seed": data_seed}
    for lab, data in (("raw", X), ("rec", Y)):
        th = th0.copy()
        if tag.startswith("random"):
            th = adam(dec, th, data, steps=1500, lr=2e-2)
        th, L, _ = levenberg_marquardt(dec, th, data, iters=200)
        out[lab + "_train"] = float(L)
        out[lab + "_excess"] = float(risk(dec, th, Xte) - RPt)
        out[lab + "_RQ"] = float(dec.loss(th, Yte)[0])
    return out


def main():
    dec = Decoder(M_HID, D_OUT)
    X = sample_square(40000, np.random.default_rng(1))
    Xte = sample_square(40000, np.random.default_rng(99))
    # samples-saturated teacher: best of many restarts
    best = None
    t0 = time.time()
    for s in range(12):
        rng = np.random.default_rng(10 + s)
        th = adam(dec, dec.init(rng), X, steps=2000, lr=2e-2, batch=1024, rng=rng)
        th, L, _ = levenberg_marquardt(dec, th, X[:10000], iters=60)
        if best is None or L < best[0]:
            best = (L, th)
    th, L, _ = levenberg_marquardt(dec, best[1], X, iters=200)
    teacher = th
    RPt = risk(dec, teacher, Xte)
    _, Yte = dec.project(teacher, Xte)
    print(f"teacher R_P = {RPt:.6e} ({time.time()-t0:.0f}s)", flush=True)
    # inequivalent optimal canons g.theta_*: equal raw risk, positive reconstruction risk
    orbit = []
    for gi, g in enumerate(D4()):
        thg = act(dec, g, teacher)
        orbit.append(dict(g=gi, RP=float(risk(dec, thg, Xte) - RPt), RQ=float(dec.loss(thg, Yte)[0])))
        print(f"g={gi}: excess raw risk {orbit[-1]['RP']:.2e}, reconstruction risk {orbit[-1]['RQ']:.2e}", flush=True)
    jobs = []
    n = 2000
    rng = np.random.default_rng(5)
    for gi, g in enumerate(D4()):
        for rep in range(2):
            th0 = act(dec, g, teacher) + 0.02 * rng.normal(size=dec.p)
            jobs.append((f"orbit{gi}", th0, 700 + 10 * gi + rep, teacher, n))
    for i in range(40):
        jobs.append((f"random{i}", dec.init(np.random.default_rng(3000 + i)), 900 + i, teacher, n))
    with Pool(4) as pool:
        res = pool.map(_job, jobs)
    json.dump(dict(teacher_RP=float(RPt), orbit=orbit, runs=res, theta=teacher.tolist()),
              open(os.path.join(RES, "trap.json"), "w"), indent=1)
    for r in res:
        print(r, flush=True)


if __name__ == "__main__":
    main()
