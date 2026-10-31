"""
exp_valleys.py -- exploratory diagnosis (wider decoder, m = 10, no capacity bound):
reconstruction students from random initialisations approach zero training loss
only along ill-conditioned valleys in which the weights grow, while students
started near the teacher converge quadratically to an exact fit.

Run:  python3 exp_valleys.py      (~5 min; writes results/valleys.json)
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canon import Decoder, adam, levenberg_marquardt, make_truth, sample_P

RES = os.path.join(HERE, "..", "results")


def lm_track(dec, th, X, iters):
    """LM with identity damping; records loss and spectrum of the normal Gauss-Newton matrix."""
    T, _ = dec.project(th, X)
    L, _ = dec.loss(th, X, T)
    lam, p, rec = 1e-3, dec.p, []
    for it in range(iters):
        S, g = dec.normal_gn(th, X, T)
        for _ in range(40):
            step = np.linalg.solve(S + lam * np.eye(p), g)
            thn = th + step
            Tn, _ = dec.project(thn, X)
            Ln, _ = dec.loss(thn, X, Tn)
            if Ln < L:
                th, T, L = thn, Tn, Ln
                lam = max(lam / 5, 1e-15)
                break
            lam *= 5
        if it % 250 == 0 or it == iters - 1:
            ev = np.linalg.eigvalsh(S)
            rec.append(dict(it=it, loss=float(L), eig_2nd=float(ev[1]), eig_max=float(ev[-1]),
                            weight_norm=float(np.linalg.norm(th))))
        if L < 1e-28:
            break
    return th, L, rec


def main():
    rng = np.random.default_rng(0)
    d, m = 6, 10
    gamma = make_truth(d, rng)
    dec = Decoder(m, d)
    X = sample_P(gamma, 4000, d, 0.05, rng)
    th = adam(dec, dec.init(np.random.default_rng(102)), X, steps=1500, lr=2e-2, batch=512, rng=rng)
    tht, Lt, _ = levenberg_marquardt(dec, th, X, iters=150)
    _, Y = dec.project(tht, X)
    n = 64
    Yn = Y[:n]
    out = dict(teacher_loss=float(Lt), teacher_weight_norm=float(np.linalg.norm(tht)), perturbed=[], random=[])
    for eps in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
        thp = tht + eps * np.random.default_rng(1).normal(size=dec.p)
        _, L, h = levenberg_marquardt(dec, thp, Yn, iters=300)
        out["perturbed"].append(dict(eps=eps, train=float(L), iters=int(len(h))))
        print("perturbation", eps, "-> train loss", L, "in", len(h), "LM iterations", flush=True)
    for s in [1, 3]:
        th1 = adam(dec, dec.init(np.random.default_rng(200 + s)), Yn, steps=1500, lr=2e-2)
        th2, L2, rec = lm_track(dec, th1, Yn, 1500)
        out["random"].append(dict(init=s, train=float(L2), popRQ=float(dec.loss(th2, Y)[0]), track=rec))
        print("random init", s, rec, "pop R_Q", dec.loss(th2, Y)[0], flush=True)
    S, _ = dec.normal_gn(tht, Yn, dec.project(tht, Yn)[0])
    ev = np.linalg.eigvalsh(S)
    out["teacher_S_eigs"] = [float(e) for e in ev[:4]] + [float(ev[-1])]
    json.dump(out, open(os.path.join(RES, "valleys.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
