"""
exp_main.py -- experiments for the transmission note.

Stages (run in order; each caches its output in ../results):
  python3 exp_main.py teacher     # truth curve, data, samples-saturated teacher
  python3 exp_main.py diag        # Hessian diagnostics at the teacher (H_Q, H_P, Sigma)
  python3 exp_main.py local       # batch students initialised in the teacher's basin
  python3 exp_main.py random      # batch students from independent random inits
  python3 exp_main.py imperfect   # finite-sample (imperfect) teachers: advantage window
  python3 exp_main.py plots       # figures into ../figures

Model: D_theta(t) = W tanh(A (cos t, sin t) + b) + c, latent circle, d=6, m=8 (p=78).
Truth: x = gamma(t) + sigma*xi, gamma a random smooth closed curve in R^6, sigma=0.05,
       t ~ von Mises (non-uniform), xi ~ N(0,I) truncated (bounded support).
"""
import os
import sys
import json
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from canon import (Decoder, adam, levenberg_marquardt, evaluate, make_truth, sample_P,
                   rotation_generator)

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CFG = dict(d=6, m=8, sigma=0.05, truth_seed=0, N_teacher=40000, N_test=40000,
           n_grid=[8, 12, 16, 24, 32, 64, 128, 256, 512, 1024],
           n_grid_random=[8, 16, 32, 64, 128, 256, 512, 1024], seeds=6, inits=4,
           adam_steps=1500, lm_iters=150, perturb=0.05, workers=4)

TEACHER_FILE = os.path.join(RES, "teacher.npz")


def truth():
    rng = np.random.default_rng(CFG["truth_seed"])
    return make_truth(CFG["d"], rng)


def load_teacher():
    z = np.load(TEACHER_FILE)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- stage: teacher
def _teacher_restart(s):
    d, m, sigma = CFG["d"], CFG["m"], CFG["sigma"]
    dec = Decoder(m, d)
    Xtr = sample_P(truth(), CFG["N_teacher"], d, sigma, np.random.default_rng(11))
    rng = np.random.default_rng(1000 + s)
    th = adam(dec, dec.init(rng), Xtr, steps=3000, lr=2e-2, batch=1024, rng=rng)
    th, L, _ = levenberg_marquardt(dec, th, Xtr[:10000], iters=60)
    return L, th


def stage_teacher():
    d, m, sigma = CFG["d"], CFG["m"], CFG["sigma"]
    gamma = truth()
    dec = Decoder(m, d)
    Xtr = sample_P(gamma, CFG["N_teacher"], d, sigma, np.random.default_rng(11))
    Xte = sample_P(gamma, CFG["N_test"], d, sigma, np.random.default_rng(12))
    t0 = time.time()
    with Pool(CFG["workers"]) as pool:
        cands = pool.map(_teacher_restart, range(8))
    for s, (L, _) in enumerate(cands):
        print(f"restart {s}: subsample loss {L:.6e}", flush=True)
    print(f"restarts done ({time.time()-t0:.0f}s)", flush=True)
    cands.sort(key=lambda z: z[0])
    th = cands[0][1]
    th, L, hist = levenberg_marquardt(dec, th, Xtr, iters=250, rtol=1e-15)
    print(f"teacher full-sample loss {L:.8e}, LM iters {len(hist)}", flush=True)
    _, Yte = dec.project(th, Xte)
    RPt = 0.5 * ((Xte - Yte) ** 2).sum(1).mean()
    np.savez(TEACHER_FILE, theta=th, Xte=Xte, Yte=Yte, RP_test=RPt, R_train=L,
             restart_losses=np.array([c[0] for c in cands]))
    print("teacher test risk", RPt)


# ---------------------------------------------------------------- stage: diag
def num_hess(dec, th, X, eps=1e-5):
    p = dec.p
    H = np.zeros((p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = eps
        H[:, j] = (dec.loss_grad(th + e, X)[1] - dec.loss_grad(th - e, X)[1]) / (2 * eps)
    return 0.5 * (H + H.T)


def stage_diag():
    T = load_teacher()
    dec = Decoder(CFG["m"], CFG["d"])
    th = T["theta"]
    X = T["Xte"][:20000]
    Tx, Y = dec.project(th, X)
    R_star = 0.5 * ((X - Y) ** 2).sum(1).mean()
    HQ, _ = dec.normal_gn(th, Y, dec.project(th, Y)[0])
    HP = num_hess(dec, th, X)
    G = dec.per_sample_grads(th, X, Tx)
    Sigma = G.T @ G / len(X)
    v = rotation_generator(dec, th)
    v /= np.linalg.norm(v)
    Pn = np.eye(dec.p) - np.outer(v, v)
    eQ = np.linalg.eigvalsh(Pn @ HQ @ Pn)
    eP = np.linalg.eigvalsh(Pn @ HP @ Pn)
    # linear-transfer constant: max_{v in N} v'HP v / v'HQ v
    w, V = np.linalg.eigh(Pn @ HQ @ Pn)
    keep = w > 1e-12 * w.max()
    B = V[:, keep] / np.sqrt(w[keep])
    ratio = np.linalg.eigvalsh(B.T @ HP @ B)
    # predicted raw excess constant tr(H^+ Sigma)/2 using the (well-defined) PSD part of HP
    wp, Vp = np.linalg.eigh(Pn @ HP @ Pn)
    pos = wp > 1e-6 * wp.max()
    Hplus = (Vp[:, pos] / wp[pos]) @ Vp[:, pos].T
    c_raw = 0.5 * np.trace(Hplus @ Sigma)
    # same with HQ (Gauss-Newton) as the curvature
    posq = w > 1e-6 * w.max()
    HQplus = (V[:, posq] / w[posq]) @ V[:, posq].T
    c_raw_gn = 0.5 * np.trace(HQplus @ Sigma)
    out = dict(R_star=R_star, sqrt2R=np.sqrt(2 * R_star),
               HQ_eigs=eQ.tolist(), HP_eigs=eP.tolist(),
               HP_minus_HQ_op=float(np.linalg.norm(HP - HQ, 2)),
               transfer_ratio_max=float(ratio[-1]), transfer_ratio_min=float(ratio[0]),
               grad_at_teacher=float(np.linalg.norm(dec.loss_grad(th, X, Tx)[1])),
               tr_Sigma=float(np.trace(Sigma)), c_raw=float(c_raw), c_raw_gn=float(c_raw_gn),
               rot_grad_component=float(abs(G.mean(0) @ v)))
    json.dump(out, open(os.path.join(RES, "diag.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:4] + ["..."] + v[-2:]) for k, v in out.items()}, indent=1))


# ---------------------------------------------------------------- student runs
def _student_job(args):
    kind, n, seed, init_id, th_init, th_teacher, extra = args
    d, m, sigma = CFG["d"], CFG["m"], CFG["sigma"]
    dec = Decoder(m, d)
    gamma = truth()
    data_rng = np.random.default_rng(50000 + 1000 * seed + n)
    Xn = sample_P(gamma, n, d, sigma, data_rng)
    _, Yn = dec.project(th_teacher, Xn)
    T = load_teacher()
    Xte, Yte_star, RPt = T["Xte"], T["Yte"], float(T["RP_test"])
    # evaluation reference: the *saturated* teacher (the target of transmission is
    # whichever teacher produced Yn; for the imperfect-teacher stage the reference
    # risk is still that of the saturated teacher, i.e. ~ the population optimum)
    out = []
    which = extra.pop("which", ("raw", "rec"))
    for label, data in (("raw", Xn), ("rec", Yn)):
        if label not in which:
            continue
        rng = np.random.default_rng(777 + 31 * seed + init_id)
        th = th_init.copy()
        if kind in ("random",):
            th = adam(dec, th, data, steps=CFG["adam_steps"], lr=2e-2, rng=rng)
        th, Ltr, hist = levenberg_marquardt(dec, th, data, iters=CFG["lm_iters"] if kind == "random" else 300)
        ev = evaluate(dec, th, Xte, Yte_star, RPt)
        # imitation of the teacher that produced Yn (differs from saturated one in 'imperfect')
        if extra.get("imperfect"):
            _, Yte_T = dec.project(th_teacher, Xte)
            _, Dx = dec.project(th, Xte)
            ev["imit_own_teacher"] = float(((Dx - Yte_T) ** 2).sum(1).mean())
        rec = dict(kind=kind, data=label, n=n, seed=seed, init=init_id, train_loss=float(Ltr),
                   lm_iters=int(len(hist)), **{k: float(v) for k, v in ev.items()}, **extra)
        if label == "raw":
            rec["teacher_train_loss"] = float(dec.loss(th_teacher, Xn)[0])
        out.append(rec)
    return out


def run_pool(jobs, fname):
    t0 = time.time()
    results = []
    with Pool(CFG["workers"]) as pool:
        for i, r in enumerate(pool.imap_unordered(_student_job, jobs)):
            results.extend(r)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i+1}/{len(jobs)} jobs  ({time.time()-t0:.0f}s)", flush=True)
                json.dump(results, open(os.path.join(RES, fname), "w"))
    json.dump(results, open(os.path.join(RES, fname), "w"))
    return results


def stage_local(inits=None):
    """Students initialised at teacher + N(0, perturb^2 I); init i of seed s is
    reproducible; with inits=None all CFG['inits'] are run, otherwise only the
    listed init ids are run and merged into an existing local.json."""
    T = load_teacher()
    th_t = T["theta"]
    dec = Decoder(CFG["m"], CFG["d"])
    ids = list(range(CFG["inits"])) if inits is None else list(inits)
    jobs = []
    for n in CFG["n_grid"]:
        for seed in range(CFG["seeds"]):
            for i in ids:
                rng = np.random.default_rng(9000 + seed + 100 * n + 7919 * i)
                th0 = th_t + CFG["perturb"] * rng.normal(size=dec.p)
                jobs.append(("local", n, seed, i, th0, th_t, {}))
    jobs.sort(key=lambda j: -j[1])
    path = os.path.join(RES, "local.json")
    old = json.load(open(path)) if (inits is not None and os.path.exists(path)) else []
    old = [r for r in old if r["init"] not in ids]
    new = run_pool(jobs, "local_partial.json")
    json.dump(old + new, open(path, "w"))


def stage_random():
    T = load_teacher()
    th_t = T["theta"]
    dec = Decoder(CFG["m"], CFG["d"])
    jobs = []
    for n in CFG["n_grid_random"]:
        for seed in range(CFG["seeds"]):
            for i in range(CFG["inits"]):
                th0 = dec.init(np.random.default_rng(424242 + 97 * seed + 13 * i))  # independent of teacher
                jobs.append(("random", n, seed, i, th0, th_t, {}))
    # longest jobs first for load balance
    jobs.sort(key=lambda j: -j[1])
    run_pool(jobs, "random.json")


def stage_imperfect():
    T = load_teacher()
    th_t = T["theta"]
    dec = Decoder(CFG["m"], CFG["d"])
    gamma = truth()
    teachers = {}
    for N in [64, 256, 1024, 4096]:
        XN = sample_P(gamma, N, CFG["d"], CFG["sigma"], np.random.default_rng(31337 + N))
        thN, LN, _ = levenberg_marquardt(dec, th_t.copy(), XN, iters=300)  # raw ERM in the teacher's basin
        ev = evaluate(dec, thN, T["Xte"], T["Yte"], float(T["RP_test"]))
        teachers[N] = (thN, ev)
        print(f"imperfect teacher N={N}: excess over saturated teacher {ev['excess']:.3e}", flush=True)
    json.dump({str(N): {k: float(v) for k, v in ev.items()} for N, (thN, ev) in teachers.items()},
              open(os.path.join(RES, "imperfect_teachers.json"), "w"), indent=1)
    jobs = []
    # reconstruction students of each imperfect teacher (3 in-basin inits per seed)
    for N, (thN, ev) in teachers.items():
        for n in [32, 128, 512, 2048]:
            for seed in range(2):
                for i in range(3):
                    rng = np.random.default_rng(5000 + seed + 100 * n + N + 7919 * i)
                    th0 = thN + CFG["perturb"] * rng.normal(size=dec.p)
                    jobs.append(("imperfect", n, seed, i, th0, thN,
                                 dict(imperfect=True, N_teacher=N, teacher_excess=float(ev["excess"]),
                                      which=("rec",))))
    # extra raw students at large n (in the saturated teacher's basin); n<=1024 come from 'local'
    for n in [2048, 4096]:
        for seed in range(3):
            rng = np.random.default_rng(6000 + seed + 100 * n)
            th0 = th_t + CFG["perturb"] * rng.normal(size=dec.p)
            jobs.append(("imperfect", n, seed, 0, th0, th_t, dict(N_teacher=0, which=("raw",))))
    jobs.sort(key=lambda j: -j[1])
    res = run_pool(jobs, "imperfect.json")
    json.dump({str(N): {k: float(v) for k, v in ev.items()} for N, (thN, ev) in teachers.items()},
              open(os.path.join(RES, "imperfect_teachers.json"), "w"), indent=1)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "local_more":   # add inits 1..inits-1 to an existing local.json
        stage_local(inits=range(1, CFG["inits"]))
    else:
        {"teacher": stage_teacher, "diag": stage_diag, "local": stage_local,
         "random": stage_random, "imperfect": stage_imperfect}[stage]()
