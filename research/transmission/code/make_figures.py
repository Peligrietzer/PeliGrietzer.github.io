"""
make_figures.py -- figures and summary tables for the transmission note.
Reads ../results/*.json, writes ../figures/*.png and ../results/summary.md

Run:  python3 make_figures.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

C_RAW_COLOR = "#c0392b"
C_REC_COLOR = "#1f6fb2"
FLOOR = 1e-16


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def by(results, **conds):
    return [r for r in results if all(r.get(k) == v for k, v in conds.items())]


def agg(rows, key, ns):
    med, lo, hi = [], [], []
    for n in ns:
        v = np.array([max(r[key], FLOOR) for r in rows if r["n"] == n])
        if len(v) == 0:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(np.median(v)); lo.append(v.min()); hi.append(v.max())
    return np.array(med), np.array(lo), np.array(hi)


def select_best(rows, keys=("n", "seed")):
    """Per (n, seed[, ...]) keep the run with the lowest training loss on its own data."""
    best = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        if k not in best or r["train_loss"] < best[k]["train_loss"]:
            best[k] = r
    return list(best.values())


EXACT = 1e-15


def fig_local(diag, lines):
    R = load("local.json")
    if R is None:
        return
    ns = sorted(set(r["n"] for r in R))
    raw_all, rec_all = by(R, data="raw"), by(R, data="rec")
    raw, rec = select_best(raw_all), select_best(rec_all)
    thr = (78 - 1) / 5.0  # (p - dim orbit)/(d - k)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) excess real-data risk, best-of-inits by own training loss
    for rows, allrows, col, lab in ((raw, raw_all, C_RAW_COLOR, "raw-trained"),
                                    (rec, rec_all, C_REC_COLOR, "reconstruction-trained")):
        ax[0].scatter([r["n"] * (1.05 if col == C_REC_COLOR else 0.95) for r in allrows],
                      [max(abs(r["excess"]), FLOOR) for r in allrows], s=5, color=col, alpha=0.2)
        m, lo, hi = agg([dict(r, excess=abs(r["excess"])) for r in rows], "excess", ns)
        ax[0].plot(ns, m, "o-", color=col, label=lab + " (best of 4 by own training loss)")
    if diag:
        nn = np.array(ns, float)
        ax[0].plot(nn, diag["c_raw"] / nn, "k--", lw=1, label=r"predicted raw $\frac{1}{2}\mathrm{tr}(H^+\Sigma)/n$")
    ax[0].axvline(thr, color="gray", ls=":", lw=1)
    ax[0].text(thr * 1.05, 0.93, "$(p-1)/(d-k)$", fontsize=8, color="gray", transform=ax[0].get_xaxis_transform())
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("n (training samples)"); ax[0].set_ylabel(r"|excess real-data risk|")
    ax[0].set_title("(a) excess risk on fresh real data"); ax[0].legend(fontsize=7, loc="lower left")
    # (b) exact fits: excess of every run whose training loss is numerically zero
    ex = [r for r in rec_all if r["train_loss"] < EXACT]
    ax[1].scatter([r["n"] for r in ex], [max(abs(r["excess"]), FLOOR) for r in ex], s=14, color=C_REC_COLOR)
    ax[1].axvline(thr, color="gray", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("n")
    ax[1].set_ylabel("|excess real-data risk|")
    ax[1].set_title("(b) every exact fit to reconstructions\n(training loss < 1e-15)")
    # (c) the certificate: training loss vs |excess| for all reconstruction runs with n >= 16
    rr = [r for r in rec_all if r["n"] >= 16]
    sc = ax[2].scatter([max(r["train_loss"], 1e-30) for r in rr], [max(abs(r["excess"]), FLOOR) for r in rr],
                       c=np.log2([r["n"] for r in rr]), cmap="viridis", s=12)
    cb = plt.colorbar(sc, ax=ax[2]); cb.set_label(r"$\log_2 n$")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("reconstruction training loss"); ax[2].set_ylabel("|excess real-data risk|")
    ax[2].set_title("(c) training-loss certificate (all runs, n >= 16)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_local.png"), dpi=130); plt.close()
    lines.append("\n### E1: students initialised in the teacher's basin (teacher + N(0, 0.05^2 I); 6 seeds x 4 inits)\n")
    lines.append("| n | rec runs with exact fit | max abs(excess) among exact fits | rec excess (median, best-of-4) | raw excess (median, best-of-4) | raw / predicted | raw gap test-train (median) |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in ns:
        allq = [r for r in rec_all if r["n"] == n]
        exq = [r for r in allq if r["train_loss"] < EXACT]
        mx = max(abs(r["excess"]) for r in exq) if exq else float("nan")
        selq = [r for r in rec if r["n"] == n]; selr = [r for r in raw if r["n"] == n]
        a = np.median([abs(r["excess"]) for r in selq]); b = np.median([r["excess"] for r in selr])
        pred = diag["c_raw"] / n if diag else np.nan
        gap = np.median([r["RP"] - r["train_loss"] for r in selr])
        lines.append(f"| {n} | {len(exq)}/{len(allq)} | {mx:.1e} | {a:.1e} | {b:.2e} | {b/pred:.2f} | {gap:.2e} |")


def fig_random(diag, lines):
    R = load("random.json")
    if R is None:
        return
    ns = sorted(set(r["n"] for r in R))
    seeds = sorted(set(r["seed"] for r in R))
    # selection by training loss on own data (no extra samples)
    sel = {"raw": [], "rec": []}
    for lab in ("raw", "rec"):
        for n in ns:
            for s in seeds:
                cand = [r for r in R if r["data"] == lab and r["n"] == n and r["seed"] == s]
                if cand:
                    sel[lab].append(min(cand, key=lambda r: r["train_loss"]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab, col, name in (("raw", C_RAW_COLOR, "raw-trained"), ("rec", C_REC_COLOR, "reconstruction-trained")):
        allr = [r for r in R if r["data"] == lab]
        ax[0].scatter([r["n"] * (1.06 if lab == "rec" else 0.94) for r in allr],
                      [max(r["excess"], FLOOR) for r in allr], s=6, color=col, alpha=0.25)
        m, lo, hi = agg(sel[lab], "excess", ns)
        ax[0].plot(ns, m, "o-", color=col, label=name + " (best of 4 inits by training loss)")
    if diag:
        nn = np.array(ns, float)
        ax[0].plot(nn, diag["c_raw"] / nn, "k--", lw=1, label="predicted raw $c/n$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("n")
    ax[0].set_ylabel("excess real-data risk"); ax[0].set_title("(a) independent random initialisation")
    ax[0].legend(fontsize=7)
    for lab, col in (("raw", C_RAW_COLOR), ("rec", C_REC_COLOR)):
        m, lo, hi = agg(sel[lab], "RQ", ns)
        ax[1].plot(ns, m, "o-", color=col)
        ax[1].fill_between(ns, lo, hi, color=col, alpha=0.15)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("n")
    ax[1].set_ylabel(r"$R_Q(\hat\theta)$ (reconstruction risk on fresh teacher outputs)")
    ax[1].set_title("(b) target (c): fresh teacher outputs")
    # certificate: training loss vs excess for reconstruction students, all runs
    rec = [r for r in R if r["data"] == "rec" and r["n"] >= 16]
    sc = ax[2].scatter([max(r["train_loss"], 1e-30) for r in rec], [max(abs(r["excess"]), FLOOR) for r in rec],
                       c=np.log2([r["n"] for r in rec]), cmap="viridis", s=12)
    cb = plt.colorbar(sc, ax=ax[2]); cb.set_label(r"$\log_2 n$")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("reconstruction training loss"); ax[2].set_ylabel("|excess real-data risk|")
    ax[2].set_title("(c) training loss vs excess (all runs, n >= 16)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_random.png"), dpi=130); plt.close()
    lines.append("\n### E2: independent random initialisations (4 inits per seed, selection by own training loss)\n")
    lines.append("| n | raw excess (median sel.) | rec excess (median sel.) | ratio raw/rec | rec R_Q (median sel.) | rec train (median sel.) | frac rec runs with train<1e-5 |")
    lines.append("|---|---|---|---|---|---|---|")
    for n in ns:
        a = np.median([r["excess"] for r in sel["raw"] if r["n"] == n])
        b = np.median([r["excess"] for r in sel["rec"] if r["n"] == n])
        q = np.median([r["RQ"] for r in sel["rec"] if r["n"] == n])
        t = np.median([r["train_loss"] for r in sel["rec"] if r["n"] == n])
        allq = [r for r in R if r["data"] == "rec" and r["n"] == n]
        frac = np.mean([r["train_loss"] < 1e-5 for r in allq])
        lines.append(f"| {n} | {a:.2e} | {b:.2e} | {a/max(b,1e-300):.0f} | {q:.1e} | {t:.1e} | {frac:.2f} |")


def fig_imperfect(diag, lines):
    R = load("imperfect.json"); T = load("imperfect_teachers.json"); L = load("local.json")
    if R is None or T is None:
        return
    Ns = sorted(int(k) for k in T.keys())
    rec_all = [r for r in R if r["data"] == "rec"]
    rec = select_best(rec_all, keys=("N_teacher", "n", "seed"))
    raw = select_best([r for r in (L or []) if r["data"] == "raw"]) + \
          [r for r in R if r["data"] == "raw"]
    nr = sorted(set(r["n"] for r in raw if r["n"] >= 16))
    nq = sorted(set(r["n"] for r in rec))
    fig, ax0 = plt.subplots(1, 1, figsize=(7.5, 4.8))
    cmap = plt.get_cmap("Blues")
    for i, N in enumerate(Ns):
        col = cmap(0.45 + 0.55 * i / max(len(Ns) - 1, 1))
        rows = [r for r in rec if r["N_teacher"] == N]
        m, lo, hi = agg(rows, "excess", nq)
        te = T[str(N)]["excess"]
        neff = diag["c_raw"] / te
        ax0.plot(nq, m, "o-", color=col, label=f"rec. student of teacher trained on N={N} (N_eff={neff:.0f})")
        ax0.axhline(te, color=col, ls=":", lw=1)
        ax0.axvline(neff, color=col, ls="--", lw=0.8, alpha=0.7)
    m, lo, hi = agg(raw, "excess", nr)
    ax0.fill_between(nr, lo, hi, color=C_RAW_COLOR, alpha=0.12)
    ax0.plot(nr, m, "s-", color=C_RAW_COLOR, label="raw-trained")
    if diag:
        nn = np.array(nr, float); ax0.plot(nn, diag["c_raw"] / nn, "k--", lw=1, label="predicted raw $c/n$")
    ax0.set_xscale("log"); ax0.set_yscale("log"); ax0.set_xlabel("n (student samples)")
    ax0.set_ylabel("excess over the saturated teacher (~population optimum)")
    ax0.set_title("Imperfect teachers: students reproduce the teacher's risk (dotted);\nraw overtakes near $N_{\\mathrm{eff}}=\\frac{1}{2}\\mathrm{tr}(H^+\\Sigma)/\\Delta_T$ (dashed)", fontsize=10)
    ax0.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_imperfect.png"), dpi=130); plt.close()
    lines.append("\n### E3: imperfect teachers (teacher = raw ERM on N samples, started at the saturated teacher)\n")
    lines.append("| N | teacher excess | predicted c/N | rec runs exact | rec-student excess (median, best-of-3) | max abs(rec excess - teacher excess) over exact fits | first n with raw median below teacher excess |")
    lines.append("|---|---|---|---|---|---|---|")
    for N in Ns:
        te = T[str(N)]["excess"]
        allq = [r for r in rec_all if r["N_teacher"] == N]
        exq = [r for r in allq if r["train_loss"] < EXACT]
        rows = [r for r in rec if r["N_teacher"] == N]
        med = np.median([r["excess"] for r in rows])
        dev = max(abs(r["excess"] - te) for r in exq) if exq else float("nan")
        cross = None
        for n in nr:
            mr = np.median([r["excess"] for r in raw if r["n"] == n])
            if mr < te:
                cross = n; break
        lines.append(f"| {N} | {te:.2e} | {diag['c_raw']/N:.2e} | {len(exq)}/{len(allq)} | {med:.2e} | {dev:.1e} | {cross} |")


def fig_trap(lines):
    R = load("trap.json")
    if R is None:
        return
    runs = R["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    orb = [r for r in runs if r["tag"].startswith("orbit")]
    rnd = [r for r in runs if r["tag"].startswith("random")]
    orbit_rq = {o["g"]: o["RQ"] for o in R["orbit"]}
    # (a) grouped by g: two runs each
    for j, r in enumerate(orb):
        g = int(r["tag"].replace("orbit", ""))
        x = g + (j % 2) * 0.25 - 0.125
        ax[0].semilogy(x, max(abs(r["raw_excess"]), FLOOR), "s", color=C_RAW_COLOR, label="raw-trained |excess|" if j == 0 else None)
        ax[0].semilogy(x, max(abs(r["rec_excess"]), FLOOR), "o", color=C_REC_COLOR, label="reconstruction-trained |excess|" if j == 0 else None)
        ax[0].semilogy(x, max(r["rec_train"], FLOOR), "x", color="gray", label="reconstruction training loss" if j == 0 else None)
    ax[0].set_xticks(range(8))
    ax[0].set_xticklabels([f"g{g}\n{orbit_rq[g]:.0e}" for g in range(8)], fontsize=7)
    ax[0].set_xlabel("start: $g\\cdot\\theta_*$ + small noise   (second line: $R_Q(g\\cdot\\theta_*)$)")
    ax[0].set_title("(a) started at the images of the teacher under the data's symmetry group")
    ax[0].legend(fontsize=7)
    x = np.arange(len(rnd))
    ax[1].semilogy(x, [max(abs(r["raw_excess"]), FLOOR) for r in rnd], "s", color=C_RAW_COLOR, label="raw-trained |excess|")
    ax[1].semilogy(x, [max(abs(r["rec_excess"]), FLOOR) for r in rnd], "o", color=C_REC_COLOR, label="reconstruction-trained |excess|")
    ax[1].semilogy(x, [max(r["rec_train"], FLOOR) for r in rnd], "x", color="gray", label="reconstruction training loss")
    ax[1].set_title("(b) independent random initialisations"); ax[1].set_xlabel("run"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_trap.png"), dpi=130); plt.close()
    lines.append("\n### E5: symmetry-breaking traps (D4-symmetric data, m=3 decoder)\n")
    lines.append(f"teacher R_P = {R['teacher_RP']:.4e}\n")
    lines.append("| g in D4 | raw excess of g.theta_* | R_Q(g.theta_*) |")
    lines.append("|---|---|---|")
    for o in R["orbit"]:
        lines.append(f"| {o['g']} | {o['RP']:.1e} | {o['RQ']:.2e} |")
    lines.append("\n| init | raw train | raw excess | rec train | rec excess | rec R_Q |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        lines.append(f"| {r['tag']} | {r['raw_train']:.3e} | {r['raw_excess']:.1e} | {r['rec_train']:.1e} | {r['rec_excess']:.1e} | {r['rec_RQ']:.1e} |")
    ok = [r for r in rnd if r["rec_train"] < 1e-10]
    lines.append(f"\nrandom inits: reconstruction students reaching train<1e-10: {len(ok)}/{len(rnd)}; "
                 f"median |rec excess| among the rest: "
                 f"{np.median([abs(r['rec_excess']) for r in rnd if r['rec_train'] >= 1e-10]) if len(ok) < len(rnd) else float('nan'):.2e}; "
                 f"median |raw excess|: {np.median([abs(r['raw_excess']) for r in rnd]):.2e}")


def fig_spectra(diag, lines):
    if diag is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    eq = np.sort(np.array(diag["HQ_eigs"]))[1:]
    ep_all = np.array(diag["HP_eigs"])
    ep = np.sort(np.delete(ep_all, np.argmin(np.abs(ep_all))))  # drop the projected-out orbit direction
    ax.semilogy(np.arange(len(eq)), np.abs(eq), "o", ms=3, color=C_REC_COLOR, label=r"$|\lambda(H_Q|_N)|$")
    ax.semilogy(np.arange(len(ep)), np.abs(ep), "s", ms=3, mfc="none", color=C_RAW_COLOR, label=r"$|\lambda(H_P|_N)|$ (black outline: negative)")
    neg = ep < 0
    ax.semilogy(np.arange(len(ep))[neg], np.abs(ep[neg]), "s", ms=5, color="k", mfc="none")
    ax.axhline(diag["HP_minus_HQ_op"], color="gray", ls=":", label=r"$\Vert H_P-H_Q\Vert$")
    ax.set_xlabel("index (sorted)"); ax.set_title("Spectra at the teacher (normal space)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_spectra.png"), dpi=130); plt.close()
    eq_all = np.sort(np.array(diag["HQ_eigs"]))
    lines.append("\n### E4: diagnostics at the saturated teacher\n")
    lines.append(f"- R_* = {diag['R_star']:.4e}, sqrt(2R_*) = {diag['sqrt2R']:.3f}")
    lines.append(f"- H_Q on N: min eigenvalue {eq_all[1]:.2e}, max {eq_all[-1]:.2e} (orbit direction eigenvalue {eq_all[0]:.1e})")
    lines.append(f"- H_P on N: min {min(diag['HP_eigs']):.2e}, max {max(diag['HP_eigs']):.2e}; number of negative eigenvalues: {(np.array(diag['HP_eigs'])<0).sum()}")
    lines.append(f"- ||H_P - H_Q|| = {diag['HP_minus_HQ_op']:.3e}")
    lines.append(f"- linear-transfer constant max v'H_P v / v'H_Q v = {diag['transfer_ratio_max']:.1f}")
    lines.append(f"- predicted raw excess constant tr(H^+ Sigma)/2 = {diag['c_raw']:.4f} (Gauss-Newton version {diag['c_raw_gn']:.4f})")


def table_restarts(lines):
    R = load("restarts.json")
    if R is None:
        return
    rec_tr = np.array([r["rec"]["train"] for r in R])
    rec_ex = np.array([r["rec"]["excess"] for r in R])
    raw_ex = np.array([r["raw"]["excess"] for r in R])
    raw_tr = np.array([r["raw"]["train"] for r in R])
    ok = rec_tr < 1e-10
    lines.append("\n### E2b: per-restart success from independent random inits (n = 256, 40 inits, Adam 3000 + LM 300)\n")
    lines.append(f"- reconstruction students with training loss < 1e-10: {ok.sum()}/{len(R)}")
    if ok.any():
        lines.append(f"- their |excess| : max {np.abs(rec_ex[ok]).max():.1e}")
    if (~ok).any():
        lines.append(f"- other reconstruction students: training loss median {np.median(rec_tr[~ok]):.1e}, excess median {np.median(rec_ex[~ok]):.2e}")
    lines.append(f"- raw students: excess median {np.median(raw_ex):.2e} (min {raw_ex.min():.2e}); training loss median {np.median(raw_tr):.4e}")
    best_rec = R[int(np.argmin(rec_tr))]["rec"]; best_raw = R[int(np.argmin(raw_tr))]["raw"]
    lines.append(f"- selection by own training loss over all 40 restarts: reconstruction student excess {best_rec['excess']:.1e} "
                 f"(train {best_rec['train']:.1e}); raw student excess {best_raw['excess']:.2e}")
    lines.append(f"- predicted raw excess at n=256: {load('diag.json')['c_raw']/256:.2e}")


def table_valleys(lines):
    V = load("valleys.json")
    if V is None:
        return
    lines.append("\n### E4b: wider decoder (m = 10), no capacity bound\n")
    lines.append("- started near the teacher: " + "; ".join(f"perturbation {p['eps']}: train {p['train']:.0e} in {p['iters']} LM its" for p in V["perturbed"]))
    for r in V["random"]:
        tr = r["track"]
        lines.append(f"- random init {r['init']}: " + "; ".join(f"it {t['it']}: loss {t['loss']:.1e}, 2nd eig {t['eig_2nd']:.0e}, max eig {t['eig_max']:.0e}, |theta| {t['weight_norm']:.1f}" for t in tr)
                     + f"; population R_Q {r['popRQ']:.1e}")
    lines.append(f"- teacher |theta| = {V['teacher_weight_norm']:.1f}; teacher normal-GN eigenvalues (64 pts): {', '.join(f'{e:.1e}' for e in V['teacher_S_eigs'])}")


def table_imitation(lines):
    R = load("imitation.json")
    if R is None:
        return
    lines.append("\n### E1c: imitation of the teacher on fresh real inputs (two-candidate oracle; E1 and E3 seeds)\n")
    lines.append("| students | n | runs | exact fits | max abs(excess) over exact fits | max imitation error over exact fits | max fraction of test points with different reconstruction | raw imitation error (median) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for kind in ("local", "imperfect"):
        for n in sorted(set(r["n"] for r in R if r["kind"] == kind)):
            rec = [r for r in R if r["kind"] == kind and r["n"] == n and r["data"] == "rec"]
            raw = [r for r in R if r["kind"] == kind and r["n"] == n and r["data"] == "raw"]
            ex = [r for r in rec if r["train_loss"] < EXACT]
            name = "saturated teacher" if kind == "local" else "N=1024 teacher"
            if ex:
                lines.append(f"| {name} | {n} | {len(rec)} | {len(ex)} | {max(abs(r['excess']) for r in ex):.1e} | "
                             f"{max(r['imit'] for r in ex):.1e} | {max(r['frac_points_differing'] for r in ex):.1e} | "
                             f"{(np.median([r['imit'] for r in raw]) if raw else float('nan')):.1e} |")


if __name__ == "__main__":
    diag = load("diag.json")
    lines = ["# Summary of numerical results (auto-generated by make_figures.py)"]
    fig_spectra(diag, lines)
    fig_local(diag, lines)
    fig_random(diag, lines)
    fig_imperfect(diag, lines)
    fig_trap(lines)
    table_restarts(lines)
    table_valleys(lines)
    table_imitation(lines)
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
