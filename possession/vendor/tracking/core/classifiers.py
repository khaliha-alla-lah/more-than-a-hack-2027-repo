"""Two boosted-tree classifiers for dead ball and team possession.

Training and inference use both team perspectives. Swapping teams complements the
home-possession probability and preserves dead-ball probability.
"""

import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import matrices
from ..config import GB_PARAMS

# Params
TRAIN_STRIDE = 4      # frames; neighbouring frames are near-duplicates, so subsample
SPELL_REF_S = 8.0     # s; possession-equal weights are 1 for a possession this long...
SPELL_W_LO, SPELL_W_HI = 0.25, 4.0   # ...and clipped to this range


def _poss_design(xh, xa, xs):
    return np.hstack([xh, xs]), np.hstack([xa, xs])


def spell_weights(labels, ref_frames, lo=SPELL_W_LO, hi=SPELL_W_HI):
    """Per-frame training weight that makes every possession count the same.

    spellF1 scores each possession once, however long it lasts, but a frame-level loss lets a
    30 s possession outweigh a 3 s one ten to one -- so the model learns to ignore short
    possessions, which are half of all possessions. Each frame of an H or A run of n frames gets
    ref_frames / n (a possession of ref_frames keeps weight 1), clipped to [lo, hi] so one
    two-frame blip cannot dominate. Dead-ball frames keep weight 1; X frames get 0.
    """
    labels = np.asarray(labels)
    w = np.where(labels == "X", 0.0, 1.0)
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if labels[a] in ("H", "A"):
            w[a:b] = np.clip(ref_frames / float(b - a), lo, hi)
    return w


class Stage1:
    """Possession classifiers; class name retained for the fitted checkpoint."""
    def __init__(self, gb_params=None, stride=TRAIN_STRIDE, exclude=None,
                 dead_exclude=None, possession_exclude=None):
        self.gb_params = dict(GB_PARAMS, **(gb_params or {}))
        self.stride = stride
        self.exclude = exclude   # regex: feature names to leave out (ablations)
        self.names = None        # (team names, shared names) the models were trained on
        self.deduplicate_dead = True
        self.dead_exclude = dead_exclude
        self.possession_exclude = possession_exclude

    def _matrices(self, feats):
        """(xh, xa, xs), columns picked by name in the order the models were trained on.

        The first call (in fit) fixes the column lists, minus anything matching `exclude`.
        Later calls select those names and raise if any is missing, so a model can never be fed
        a feature set it was not trained on -- not even one with the same column count.
        """
        nt, xh, xa, ns, xs = matrices(feats)
        if getattr(self, "names", None) is None:
            pat = re.compile(self.exclude) if self.exclude else None
            self.names = ([n for n in nt if not (pat and pat.search(n))],
                          [n for n in ns if not (pat and pat.search(n))])
        want_t, want_s = self.names
        missing = sorted(set(want_t) - set(nt)) + sorted(set(want_s) - set(ns))
        if missing:
            raise ValueError("features missing for this model (run python -m model.run to retrain): {}"
                             .format(", ".join(missing[:10])))
        it = {n: i for i, n in enumerate(nt)}
        js = {n: i for i, n in enumerate(ns)}
        cols_t = [it[n] for n in want_t]
        cols_s = [js[n] for n in want_s]
        return xh[:, cols_t], xa[:, cols_t], xs[:, cols_s]

    def _dead_unfiltered(self, xh, xa, xs):
        """Remove duplicated own/opponent columns from the second team block."""
        keep = [i for i, name in enumerate(self.names[0])
                if not name.startswith(("own_", "opp_"))]
        return np.hstack([xs, xh, xa[:, keep]]), np.hstack([xs, xa, xh[:, keep]])

    @staticmethod
    def _filter_inputs(first, second, names, pattern):
        if not pattern:
            return first, second
        regex = re.compile(pattern)
        keep = [i for i, name in enumerate(names) if not regex.search(name)]
        return first[:, keep], second[:, keep]

    def _dead_matrices(self, xh, xa, xs):
        first, second = self._dead_unfiltered(xh, xa, xs)
        team, shared = self.names
        second_names = [n for n in team if not n.startswith(("own_", "opp_"))]
        names = ["shared."+n for n in shared]+["team1."+n for n in team]+["team2."+n for n in second_names]
        return self._filter_inputs(first, second, names, getattr(self, "dead_exclude", None))

    def _possession_matrices(self, xh, xa, xs):
        first, second = _poss_design(xh, xa, xs)
        names = ["team."+n for n in self.names[0]]+["shared."+n for n in self.names[1]]
        return self._filter_inputs(first, second, names, getattr(self, "possession_exclude", None))

    def fit(self, items):
        """items: [(feats, labels)] where labels is an array of 'H'/'A'/'D'/'X' per frame."""
        dX, dy, pX, py, pw = [], [], [], [], []
        for feats, labels in items:
            xh, xa, xs = self._matrices(feats)
            fps = feats.get("fps", 25)
            stride = max(1, round(self.stride * fps / 25))
            sel = np.flatnonzero(np.isin(labels, ["H", "A", "D"]))[::stride]
            lab = labels[sel]
            w_all = spell_weights(labels, SPELL_REF_S * fps)
            d1, d2 = self._dead_matrices(xh[sel], xa[sel], xs[sel])
            dead = (lab == "D").astype(np.int8)
            dX += [d1, d2]
            dy += [dead, dead]
            alive = lab != "D"
            p1, p2 = self._possession_matrices(xh[sel][alive], xa[sel][alive], xs[sel][alive])
            home = (lab[alive] == "H").astype(np.int8)
            pX += [p1, p2]
            py += [home, 1 - home]
            pw += [w_all[sel][alive]] * 2
        self.dead = HistGradientBoostingClassifier(**self.gb_params).fit(
            np.vstack(dX), np.concatenate(dy))
        self.poss = HistGradientBoostingClassifier(**self.gb_params).fit(
            np.vstack(pX), np.concatenate(py), sample_weight=np.concatenate(pw))
        return self

    def predict(self, feats):
        xh, xa, xs = self._matrices(feats)
        d1, d2 = self._dead_matrices(xh, xa, xs)
        p_dead = 0.5 * (self.dead.predict_proba(d1)[:, 1] + self.dead.predict_proba(d2)[:, 1])
        p1, p2 = self._possession_matrices(xh, xa, xs)
        p_home = 0.5 * (self.poss.predict_proba(p1)[:, 1] + 1 - self.poss.predict_proba(p2)[:, 1])
        return p_dead, p_home
