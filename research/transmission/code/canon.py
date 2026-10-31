"""
canon.py -- numpy implementation of a nonlinear neural decoder with an exact
nearest-point projection oracle, used for the transmission experiments.

Decoder (latent domain K = circle S^1, embedded as u(t) = (cos t, sin t)):

    D_theta(t) = W tanh(A u(t) + b) + c,
    A in R^{m x 2}, b in R^m, W in R^{d x m}, c in R^d.

Canon:      M_theta = D_theta(S^1)  (a closed curve in R^d)
Oracle:     pi_theta(x) = argmin_{y in M_theta} |x - y|^2
            (dense grid search on the latent circle, then safeguarded Newton)
Loss:       ell_theta(x) = 0.5 |x - pi_theta(x)|^2
Gradient:   envelope theorem, grad = -J_theta(t*)^T (x - D_theta(t*))

Symmetries of this parametrisation that preserve the canon:
  * latent rotations / reflection  (A -> A R),       continuous: O(2)
  * hidden-unit permutations and sign flips (tanh is odd)    : finite
So the teacher orbit G.theta_* is a 1-dimensional manifold (circle).
"""
import numpy as np

TWO_PI = 2.0 * np.pi


class Decoder:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.shapes = [("A", (m, 2)), ("b", (m,)), ("W", (d, m)), ("c", (d,))]
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.p = int(sum(self.sizes))
        offs = np.cumsum([0] + self.sizes)
        self.sl = {name: slice(offs[i], offs[i + 1]) for i, (name, _) in enumerate(self.shapes)}

    # ------------------------------------------------------------------ params
    def unpack(self, th):
        return {name: th[self.sl[name]].reshape(shp) for name, shp in self.shapes}

    def pack(self, A, b, W, c):
        return np.concatenate([np.ravel(A), np.ravel(b), np.ravel(W), np.ravel(c)])

    def init(self, rng, scale_A=1.5, scale_b=0.5, scale_W=None):
        m, d = self.m, self.d
        if scale_W is None:
            scale_W = 1.0 / np.sqrt(m)
        A = rng.normal(0, scale_A, (m, 2))
        b = rng.normal(0, scale_b, m)
        W = rng.normal(0, scale_W, (d, m))
        c = np.zeros(d)
        return self.pack(A, b, W, c)

    # ------------------------------------------------------------------ curve
    def _hidden(self, th, t):
        P = self.unpack(th)
        u = np.stack([np.cos(t), np.sin(t)], -1)
        pre = u @ P["A"].T + P["b"]
        h = np.tanh(pre)
        return P, u, pre, h

    def curve(self, th, t):
        P, u, pre, h = self._hidden(th, t)
        return h @ P["W"].T + P["c"]

    def curve_d2(self, th, t):
        """D, dD/dt, d^2D/dt^2 at latents t."""
        P, u, pre, h = self._hidden(th, t)
        A, b, W, c = P["A"], P["b"], P["W"], P["c"]
        s = 1.0 - h ** 2
        up = np.stack([-np.sin(t), np.cos(t)], -1)
        prep = up @ A.T
        prepp = -(pre - b)  # u'' = -u
        hp = s * prep
        hpp = -2.0 * h * s * prep ** 2 + s * prepp
        return h @ W.T + c, hp @ W.T, hpp @ W.T

    # ------------------------------------------------------------------ oracle
    def project(self, th, X, G=1024, newton=10, chunk=8192, candidates=1):
        """Exact nearest-point projection of the rows of X onto M_theta.
        Returns latents T (N,) and projections D (N,d).
        candidates=1: refine the best grid point (used for all training runs).
        candidates=2: also refine the best grid point outside a window around the
        first one (i.e. on another branch of the curve) and keep the better of the
        two; this removes mis-assignments of near-tie points (used for the
        imitation check, exp_imitation.py)."""
        tg = np.linspace(0.0, TWO_PI, G, endpoint=False)
        Dg = self.curve(th, tg)
        Dg2 = (Dg ** 2).sum(1)
        N = len(X)
        T = np.empty(N)
        T2 = np.empty(N) if candidates > 1 else None
        win = max(G // 32, 2)
        for s0 in range(0, N, chunk):
            Xc = X[s0:s0 + chunk]
            d2 = Dg2[None, :] - 2.0 * (Xc @ Dg.T)
            i1 = np.argmin(d2, 1)
            T[s0:s0 + chunk] = tg[i1]
            if candidates > 1:
                off = (np.arange(G)[None, :] - i1[:, None]) % G
                masked = np.where((off <= win) | (off >= G - win), np.inf, d2)
                T2[s0:s0 + chunk] = tg[np.argmin(masked, 1)]
        if candidates > 1:
            Ta, Da = self._refine(th, X, T, G, newton)
            Tb, Db = self._refine(th, X, T2, G, newton)
            pa = ((X - Da) ** 2).sum(1)
            pb = ((X - Db) ** 2).sum(1)
            use_b = pb < pa
            return np.where(use_b, Tb, Ta), np.where(use_b[:, None], Db, Da)
        return self._refine(th, X, T, G, newton)

    def _refine(self, th, X, T, G, newton):
        """Safeguarded Newton refinement of grid latents T (confined to the grid cell)."""
        N = len(X)
        T = T.copy()
        h_grid = TWO_PI / G
        # safeguarded Newton on phi(t) = 0.5 |x - D(t)|^2, confined to the grid cell
        T0 = T.copy()
        D = self.curve(th, T)
        phi = 0.5 * ((X - D) ** 2).sum(1)
        for _ in range(newton):
            D, Dp, Dpp = self.curve_d2(th, T)
            r = X - D
            g = -(r * Dp).sum(1)
            H = (Dp ** 2).sum(1) - (r * Dpp).sum(1)
            step = np.where(H > 1e-14, g / np.maximum(H, 1e-14), np.sign(g) * 0.25 * h_grid)
            step = np.clip(step, -h_grid, h_grid)
            accepted = np.zeros(N, bool)
            for _ls in range(6):
                Tn = T - step
                Tn = np.clip(Tn, T0 - 1.5 * h_grid, T0 + 1.5 * h_grid)
                Dn = self.curve(th, Tn)
                phin = 0.5 * ((X - Dn) ** 2).sum(1)
                ok = (phin <= phi) & (~accepted)
                T = np.where(ok, Tn, T)
                phi = np.where(ok, phin, phi)
                accepted |= ok
                step = 0.5 * step
                if accepted.all():
                    break
        D = self.curve(th, T)
        return np.mod(T, TWO_PI), D

    # ------------------------------------------------------------------ loss
    def loss(self, th, X, T=None):
        if T is None:
            T, D = self.project(th, X)
        else:
            D = self.curve(th, T)
        return 0.5 * ((X - D) ** 2).sum(1).mean(), T

    def loss_grad(self, th, X, T=None):
        """Mean profiled loss and its envelope gradient (any minimising latent)."""
        if T is None:
            T, _ = self.project(th, X)
        P, u, pre, h = self._hidden(th, T)
        A, b, W, c = P["A"], P["b"], P["W"], P["c"]
        D = h @ W.T + c
        r = X - D
        N = len(X)
        L = 0.5 * (r ** 2).sum() / N
        gW = -(r.T @ h) / N
        gc = -r.sum(0) / N
        dpre = -(r @ W) * (1.0 - h ** 2)
        gA = dpre.T @ u / N
        gb = dpre.sum(0) / N
        return L, self.pack(gA, gb, gW, gc), T

    def per_sample_grads(self, th, X, T=None):
        """(N,p) matrix of per-sample envelope gradients."""
        if T is None:
            T, _ = self.project(th, X)
        P, u, pre, h = self._hidden(th, T)
        A, b, W, c = P["A"], P["b"], P["W"], P["c"]
        D = h @ W.T + c
        r = X - D
        N = len(X)
        gW = -(r[:, :, None] * h[:, None, :]).reshape(N, -1)
        gc = -r
        dpre = -(r @ W) * (1.0 - h ** 2)
        gA = (dpre[:, :, None] * u[:, None, :]).reshape(N, -1)
        gb = dpre
        return np.concatenate([gA, gb, gW, gc], 1)

    # ------------------------------------------------------------------ jacobians
    def jacobian(self, th, T):
        """Per-sample parameter Jacobians J_i = dD_theta(t_i)/dtheta, shape (N,d,p)."""
        P, u, pre, h = self._hidden(th, T)
        W = P["W"]
        s = 1.0 - h ** 2
        N, d, m = len(T), self.d, self.m
        J = np.zeros((N, d, self.p))
        WS = W[None, :, :] * s[:, None, :]  # (N,d,m)
        J[:, :, self.sl["A"]] = (WS[:, :, :, None] * u[:, None, None, :]).reshape(N, d, 2 * m)
        J[:, :, self.sl["b"]] = WS
        JW = np.zeros((N, d, d, m))
        for j in range(d):
            JW[:, j, j, :] = h
        J[:, :, self.sl["W"]] = JW.reshape(N, d, d * m)
        J[:, :, self.sl["c"]] = np.eye(d)[None]
        return J

    def normal_gn(self, th, X, T, chunk=4096):
        """Normal Gauss-Newton matrix S = mean_i J_i^T P_i^perp J_i and
        g = mean_i J_i^T r_i (= minus the envelope gradient)."""
        p = self.p
        S = np.zeros((p, p))
        g = np.zeros(p)
        N = len(X)
        for s0 in range(0, N, chunk):
            Tc, Xc = T[s0:s0 + chunk], X[s0:s0 + chunk]
            J = self.jacobian(th, Tc)
            D, Dp, _ = self.curve_d2(th, Tc)
            r = Xc - D
            a = Dp / np.linalg.norm(Dp, axis=1, keepdims=True).clip(1e-12)
            Ja = np.einsum("ndp,nd->np", J, a)
            Jf = J.reshape(-1, p)
            S += Jf.T @ Jf - Ja.T @ Ja
            g += Jf.T @ r.reshape(-1)
        return S / N, g / N


# ====================================================================== training
def adam(dec, th, X, steps=1500, lr=2e-2, lr_end=1e-3, batch=None, rng=None, project_G=512):
    m1 = np.zeros_like(th)
    m2 = np.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-10
    for k in range(1, steps + 1):
        if batch is not None and batch < len(X):
            Xb = X[rng.choice(len(X), batch, replace=False)]
        else:
            Xb = X
        T, _ = dec.project(th, Xb, G=project_G, newton=4)
        _, g, _ = dec.loss_grad(th, Xb, T)
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        mh = m1 / (1 - b1 ** k)
        vh = m2 / (1 - b2 ** k)
        frac = (k - 1) / max(steps - 1, 1)
        lr_k = lr_end + 0.5 * (lr - lr_end) * (1 + np.cos(np.pi * frac))
        th = th - lr_k * mh / (np.sqrt(vh) + eps)
    return th


def levenberg_marquardt(dec, th, X, iters=200, lam=1e-3, rtol=1e-13, verbose=False):
    """Gauss-Newton/LM on the profiled loss with the *normal* Gauss-Newton matrix
    (variable-projection Schur complement) and a global re-projection after every
    step.  Converges quadratically on zero-residual (reconstruction) problems and
    linearly to the local minimiser on noisy (raw) problems."""
    T, _ = dec.project(th, X)
    L, _ = dec.loss(th, X, T)
    p = dec.p
    hist = [L]
    for it in range(iters):
        S, g = dec.normal_gn(th, X, T)
        improved = False
        for _ in range(30):
            M = S + lam * (np.diag(np.diag(S)) + 1e-9 * np.eye(p))
            try:
                step = np.linalg.solve(M, g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            thn = th + step
            Tn, _ = dec.project(thn, X)
            Ln, _ = dec.loss(thn, X, Tn)
            if Ln < L:
                rel = (L - Ln) / max(L, 1e-300)
                th, T, L = thn, Tn, Ln
                lam = max(lam / 5.0, 1e-12)
                improved = True
                break
            lam *= 5.0
            if lam > 1e10:
                break
        hist.append(L)
        if verbose:
            print(it, L, lam)
        if not improved:
            break
        if rel < rtol or L < 1e-28:
            break
    return th, L, np.array(hist)


def train_fully(dec, th0, X, rng, adam_steps=1500, lr=2e-2, lm_iters=150, batch=None):
    """'Train to convergence' = Adam exploration followed by LM polishing."""
    th = adam(dec, th0.copy(), X, steps=adam_steps, lr=lr, batch=batch, rng=rng)
    th, L, hist = levenberg_marquardt(dec, th, X, iters=lm_iters)
    return th, L


# ====================================================================== evaluation
def evaluate(dec, th, Xtest, Ytest, RP_teacher):
    """Excess real-data risk over the teacher, teacher-imitation error on real
    inputs, and reconstruction risk on teacher outputs."""
    _, Dx = dec.project(th, Xtest)
    RP = 0.5 * ((Xtest - Dx) ** 2).sum(1).mean()
    imit = ((Dx - Ytest) ** 2).sum(1).mean()
    _, Dy = dec.project(th, Ytest)
    RQ = 0.5 * ((Ytest - Dy) ** 2).sum(1).mean()
    return dict(excess=RP - RP_teacher, RP=RP, imit=imit, RQ=RQ)


# ====================================================================== data
def make_truth(d, rng, K=3, amp=1.0, decay=1.6):
    """Random smooth closed curve gamma(t) = sum_k a_k cos kt + b_k sin kt in R^d."""
    a = rng.normal(0, 1, (K, d)) * (amp / (np.arange(1, K + 1) ** decay))[:, None]
    bb = rng.normal(0, 1, (K, d)) * (amp / (np.arange(1, K + 1) ** decay))[:, None]

    def gamma(t):
        ks = np.arange(1, K + 1)
        return np.cos(np.outer(t, ks)) @ a + np.sin(np.outer(t, ks)) @ bb

    return gamma


def sample_P(gamma, n, d, sigma, rng, kappa=1.0):
    """x = gamma(t) + sigma * xi,  t ~ wrapped non-uniform density, xi ~ N(0, I_d)
    truncated at norm 3*sigma*sqrt(d) (bounded support)."""
    t = rng.vonmises(0.0, kappa, n) + np.pi  # non-uniform latent law
    xi = rng.normal(0, 1, (n, d))
    nrm = np.linalg.norm(xi, axis=1, keepdims=True)
    cap = 3.0 * np.sqrt(d)
    xi = np.where(nrm > cap, xi * cap / nrm, xi)
    return gamma(t) + sigma * xi


# ====================================================================== symmetry tools
def rotation_generator(dec, th):
    """Tangent vector of the latent-rotation orbit theta -> (A R_phi, b, W, c) at phi=0."""
    P = dec.unpack(th)
    Jrot = np.array([[0.0, -1.0], [1.0, 0.0]])
    v = np.zeros(dec.p)
    v[dec.sl["A"]] = (P["A"] @ Jrot).ravel()
    return v
