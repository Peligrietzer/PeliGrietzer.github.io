"""
exp_restarts.py -- per-restart success probability of training from independent
random initialisations (n = 256, 40 inits, longer schedule), for raw and
reconstruction students.  Reads results/teacher.npz; writes results/restarts.json.

Run:  python3 exp_restarts.py
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
from canon import Decoder, adam, levenberg_marquardt, evaluate, sample_P

N_SAMPLES, N_INITS, ADAM, LM = 256, 40, 3000, 300


def job(i):
    T = E.load_teacher()
    th_t = T["theta"]
    dec = Decoder(E.CFG["m"], E.CFG["d"])
    Xn = sample_P(E.truth(), N_SAMPLES, E.CFG["d"], E.CFG["sigma"], np.random.default_rng(123456))
    _, Yn = dec.project(th_t, Xn)
    th0 = dec.init(np.random.default_rng(8080 + i))
    out = {"init": i}
    for lab, data in (("raw", Xn), ("rec", Yn)):
        th = adam(dec, th0.copy(), data, steps=ADAM, lr=2e-2, rng=np.random.default_rng(i))
        th, L, _ = levenberg_marquardt(dec, th, data, iters=LM)
        ev = evaluate(dec, th, T["Xte"], T["Yte"], float(T["RP_test"]))
        out[lab] = dict(train=float(L), **{k: float(v) for k, v in ev.items()})
    out["raw"]["teacher_train"] = float(dec.loss(th_t, Xn)[0])
    return out


if __name__ == "__main__":
    with Pool(E.CFG["workers"]) as pool:
        res = pool.map(job, range(N_INITS))
    json.dump(res, open(os.path.join(E.RES, "restarts.json"), "w"), indent=1)
    rec_tr = np.array([r["rec"]["train"] for r in res])
    print("reconstruction students with training loss < 1e-10:", int((rec_tr < 1e-10).sum()), "/", N_INITS)
    for r in sorted(res, key=lambda r: r["rec"]["train"]):
        print(f"init {r['init']:2d}  rec train {r['rec']['train']:.1e} excess {r['rec']['excess']:+.1e} | "
              f"raw train {r['raw']['train']:.4e} excess {r['raw']['excess']:+.1e}")
