"""
Plackett-Luce (PL) model of the first five finishers, numpy version for research.

A race is a padded row: X (N x k features), mask (N,) of real runners, order (5,) = indices of
the first five finishers. Strength s = X @ w; at stage j the next finisher is drawn with
probability softmax(lam_j * s) over the runners not yet placed. lam_1 = 1; optional stage
discounts lam_2..lam_5 let later places be less predictable than the win (a classic failure
of plain PL in racing). Fitted by L-BFGS on the penalised log-likelihood.

Probabilities:
- win: exact softmax;
- an ordered ticket and a 5-horse set (120 orders): exact products;
- top 5 per runner and Bonus categories: Monte Carlo with the Gumbel trick (sequential stages).
"""
import itertools

import numpy as np
from scipy.optimize import minimize

NEG = -1e9


def stage_ll(s, mask, order, lam):
    """Log-likelihood per race and its gradient w.r.t. s and lam. s: (R, N)."""
    R, N = s.shape
    avail = mask.copy()
    ll = np.zeros(R)
    ds = np.zeros_like(s)
    dlam = np.zeros((R, len(lam)))
    rows = np.arange(R)
    for j in range(order.shape[1]):
        o = order[:, j]
        ok = o >= 0
        z = np.where(avail, lam[j] * s, NEG)
        zmax = z.max(axis=1, keepdims=True)
        e = np.where(avail, np.exp(z - zmax), 0.0)
        p = e / e.sum(axis=1, keepdims=True)
        lse = zmax[:, 0] + np.log(e.sum(axis=1))
        so = s[rows, np.where(ok, o, 0)]
        ll += np.where(ok, lam[j] * so - lse, 0.0)
        g = -p * lam[j]
        g[rows[ok], o[ok]] += lam[j]
        ds += np.where(ok[:, None], g, 0.0)
        dlam[:, j] = np.where(ok, so - (p * s).sum(axis=1), 0.0)
        avail = avail.copy()
        avail[rows[ok], o[ok]] = False
    return ll, ds, dlam


class PL:
    def __init__(self, penalty, discount=False):
        self.penalty = np.asarray(penalty, float)  # one L2 weight per feature column
        self.discount = discount

    def fit(self, X, mask, order):
        k = X.shape[2]

        def unpack(theta):
            w = theta[:k]
            lam = np.concatenate([[1.0], np.exp(theta[k:])]) if self.discount else np.ones(5)
            return w, lam

        def f(theta):
            w, lam = unpack(theta)
            s = X @ w
            ll, ds, dlam = stage_ll(s, mask, order, lam)
            obj = -ll.sum() + 0.5 * (self.penalty * w * w).sum()
            gw = -np.einsum("rn,rnk->k", ds, X) + self.penalty * w
            if not self.discount:
                return obj, gw
            gl = -dlam.sum(axis=0)[1:] * lam[1:] + 1.0 * theta[k:]  # weak pull of log lam to 0
            return obj + 0.5 * (theta[k:] ** 2).sum(), np.concatenate([gw, gl])

        theta0 = np.zeros(k + (4 if self.discount else 0))
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.lam = unpack(res.x)
        return self

    def strengths(self, X):
        return X @ self.w


# ---------------------------------------------------------------- probabilities of one race

def win_probs(s, lam1=1.0):
    z = lam1 * (s - s.max())
    e = np.exp(z)
    return e / e.sum()


def order_prob(s, lam, ticket):
    """Exact probability that the first finishers are exactly `ticket`, in that order."""
    left = np.ones(len(s), bool)
    p = 1.0
    for j, i in enumerate(ticket):
        z = lam[j] * s[left]
        m = z.max()
        p *= np.exp(lam[j] * s[i] - m) / np.exp(z - m).sum()
        left[i] = False
    return p


def set_prob(s, lam, ticket):
    """Exact probability that the first five are the 5 runners of `ticket`, in any order."""
    return float(perm_probs(s, lam, np.array(list(itertools.permutations(ticket)))).sum())


def perm_probs(s, lam, perms):
    """Exact probabilities of many finishing orders at once: perms (M, depth) of runner indices."""
    depth = perms.shape[1]
    E = np.exp(lam[:depth, None] * s[None, :] - (lam[:depth] * s.max())[:, None])  # (depth, n)
    tot = E.sum(axis=1)
    p = np.ones(len(perms))
    for j in range(depth):
        denom = tot[j] - sum(E[j, perms[:, l]] for l in range(j))  # runners already placed are out
        p *= E[j, perms[:, j]] / denom
    return p


def sample_orders(s, lam, n_sims, rng, depth=5):
    """(n_sims, depth) sampled finishing orders under (discounted) PL, Gumbel trick per stage."""
    n = len(s)
    taken = np.zeros((n_sims, n), bool)
    out = np.empty((n_sims, depth), int)
    for j in range(depth):
        g = lam[j] * s[None, :] - np.log(-np.log(rng.random((n_sims, n))))
        g[taken] = -np.inf
        pick = g.argmax(axis=1)
        out[:, j] = pick
        taken[np.arange(n_sims), pick] = True
    return out


def top5_probs(orders, n):
    hits = np.zeros(n)
    np.add.at(hits, orders.ravel(), 1)
    return hits / orders.shape[0]
