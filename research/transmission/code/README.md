# Code for "Transmission from a samples-saturated teacher"

Pure numpy (plus matplotlib for figures). Tested with Python 3.11, numpy 2.x, matplotlib 3.11.

```bash
pip3 install numpy matplotlib
cd research/transmission/code
python3 exp_main.py teacher     # ~10 min: truth curve, data, samples-saturated teacher (results/teacher.npz)
python3 exp_main.py diag        # ~2 min: H_Q, H_P, Sigma at the teacher (results/diag.json)
python3 exp_main.py local       # ~12 min: batch students initialised in the teacher's basin, 6 seeds x 4 inits (results/local.json)
python3 exp_main.py random      # ~12 min: batch students from independent random inits (results/random.json)
python3 exp_main.py imperfect   # ~8 min: finite-sample teachers, advantage window (results/imperfect.json)
python3 exp_trap.py             # ~15 min: symmetry-breaking traps (results/trap.json)
python3 exp_restarts.py         # ~5 min: 40 random restarts at n=256 (results/restarts.json)
python3 exp_valleys.py          # ~5 min: wider decoder, valleys at infinity (results/valleys.json)
python3 exp_imitation.py        # ~6 min: imitation error with the two-candidate oracle (results/imitation.json)
python3 make_figures.py         # figures/*.png and results/summary.md
```

The note itself is rendered by `render_pdf.py`. It runs pandoc, then headless Chromium through Node Playwright. For KaTeX-typeset math, pass the path of a KaTeX `dist` directory (`npm install katex`):

```bash
python3 render_pdf.py /path/to/node_modules/katex/dist
```

The stages of `exp_main.py` must run in the order shown, because `diag`, `local`, `random` and `imperfect` read `results/teacher.npz`. That file (~4 MB) is not committed; regenerate it with the `teacher` stage. (`exp_main.py local_more` re-runs only initialisations 1–3 and merges them into an existing `local.json`; the committed `local.json` was produced this way, and is identical to a single `local` run.) All stages use 4 worker processes; edit `CFG["workers"]` in `exp_main.py` to change this.

## Files

| file | contents |
|---|---|
| `canon.py` | decoder $D_\theta(t)=W\tanh(A(\cos t,\sin t)+b)+c$; exact projection oracle (dense latent grid + safeguarded Newton); profiled loss and envelope gradient; per-sample Jacobians; normal Gauss–Newton matrix $\sum J^\top P^\perp J$; Adam; Levenberg–Marquardt with global re-projection; evaluation (excess real-data risk, teacher imitation, $R_Q$) |
| `exp_main.py` | main experiments (E1–E4 in the note) |
| `exp_trap.py` | obstruction experiment (E5): $D_4$-symmetric data, capacity-limited canon, inequivalent optima become traps for reconstruction training |
| `exp_restarts.py` | per-restart success probability from independent random inits (E2b) |
| `exp_valleys.py` | wider decoder without a capacity bound: near-zero valleys in which the weights grow (E4b) |
| `exp_imitation.py` | target (b): imitation error of exact fits, measured with the near-tie-safe two-candidate oracle (E1c) |
| `make_figures.py` | figures and the auto-generated `results/summary.md` |
| `render_pdf.py`, `render_pdf.js`, `note.css` | render `../transmission_note.md` to `../transmission_note.pdf` |

## Protocol notes

* **Same initialisation.** Raw and reconstruction students in a comparison start from the same weights. In the `random` stage these come from the standard initialiser with seeds that do not involve the teacher.
* **"Training fully"** means Adam (exploration) followed by Levenberg–Marquardt to convergence. Each student uses only its own $n$ samples.
* **Selection.** In the `random` stage, among 4 initialisations, the student with the lowest training loss *on its own data* is kept. This costs no extra samples.
* **Oracle.** All training runs use the single-candidate oracle (`project(..., candidates=1)`). `exp_imitation.py` re-evaluates with `candidates=2`, which also refines the best grid point on another branch of the curve. This removes mis-assigned near-tie test points, which otherwise inflate the imitation metric to about 1e-4 and the excess-risk floor to about 2e-7.
* **The `local` stage** initialises at the teacher plus Gaussian noise of 0.05 per coordinate. It tests the local/statistical theorem and is labelled as such in the note. It is *not* an independent initialisation.
* **Evaluation.** Excess real-data risk is measured on a fixed fresh test set of 40,000 points, against the same set's teacher risk (common random numbers).
