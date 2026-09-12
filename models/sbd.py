import math
import torch
import torch.nn as nn
import torch.optim as optim
from models._abstract_pt_model import AbstractPTModel

LOG_S_MIN, LOG_S_MAX = -6.907755278982137, 10.505067539570582
ROOT_MIN, ROOT_MAX = 0.5, 20.0
BAD_LOSS = 1e6

INIT_S = [-2.803101, 1.386926]
ROOT = [1.739235]
ALPHA = [-1.819954]
CEXP = [0.529839]
VT = [-0.886052]
LOG_DELTA = [-1.393709]

L2_SIGMA = {
    "init_s": [1.754, 1.2339],
    "root": [0.5003],
    "alpha": [0.546],
    "cexp": [0.0989],
    "vt": [1.1685],
    "log_delta": [0.7778],
}


def _safe_log(x):
    return torch.log(x.clamp(min=1e-9))


def _safe_exp(x):
    return torch.exp(x.clamp(-30.0, 30.0))


class SBDPredictor(nn.Module):
    def __init__(
        self,
        init_s=None,
        root=None,
        alpha=None,
        cexp=None,
        vt=None,
        log_delta=None,
        n_grades=4,
    ):
        super().__init__()
        self.n_grades = n_grades
        self.grade_step = 3.0 / (n_grades - 1)
        self.init_s = nn.Parameter(torch.tensor(init_s or INIT_S, dtype=torch.float32))
        self.root = nn.Parameter(torch.tensor(root or ROOT, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha or ALPHA, dtype=torch.float32))
        self.cexp = nn.Parameter(torch.tensor(cexp or CEXP, dtype=torch.float32))
        self.vt = nn.Parameter(torch.tensor(vt or VT, dtype=torch.float32))
        self.log_delta = nn.Parameter(
            torch.tensor(log_delta or LOG_DELTA, dtype=torch.float32)
        )

    def _ladder(self):
        endpoints = torch.cummax(self.init_s, dim=0)[0].clamp(LOG_S_MIN, LOG_S_MAX)
        return endpoints[0], endpoints[1], (endpoints[1] - endpoints[0]) / 3

    def _shape(self, n):
        delta = _safe_exp(self.log_delta[0])
        return _safe_exp(
            self.alpha[0] + self.cexp[0].clamp(min=0.0) * torch.log(delta + n)
        )

    def _retr(self, t, log_s, grade_index, n):
        root = self.root[0].clamp(ROOT_MIN, ROOT_MAX)
        log_rho = self._ladder()[2]
        log_age = _safe_log(t.clamp(min=0.0)) / root
        log_b = 2 * self.vt[0] / root - grade_index * log_rho
        log_a = torch.logaddexp(log_age, 0.5 * (log_b + log_s))
        log_base = nn.functional.softplus(log_a - log_s)
        return _safe_exp(-self._shape(n) * log_base).clamp(1e-6, 1 - 1e-6)

    def _step(self, state, dt, rating):
        log_s, grade_index, n = state[..., 0], state[..., 1], state[..., 2]
        r = self._retr(dt.clamp(min=0.0), log_s, grade_index, n)
        s = _safe_exp(log_s)
        is_pass = (rating > 1.5).float()
        s_pass = s * _safe_exp(-_safe_log(r) / self._shape(n))
        s_fail = s * _safe_exp(self.cexp[0].clamp(min=0.0) * _safe_log(1.0 - r))
        s_n = is_pass * s_pass + (1.0 - is_pass) * s_fail
        log_s_n = torch.nan_to_num(_safe_log(s_n), nan=0.0).clamp(LOG_S_MIN, LOG_S_MAX)
        n_n = (n + ((3.0 - rating) / 2.0).clamp(min=0.0)).clamp(0.0, 50.0)
        return torch.stack([log_s_n, rating - 1, n_n], dim=-1)

    def decode(self, states, deltas):
        p = self._retr(deltas[..., 0], states[..., 0], states[..., 1], states[..., 2])
        return torch.nan_to_num(p, nan=0.5).clamp(1e-6, 1 - 1e-6)

    def forward(self, x, deltas):
        elapsed = x[..., 0].clamp(min=0.0)
        rating = x[..., 1].clamp(1.0, float(self.n_grades))
        rating = 1.0 + (rating - 1.0) * self.grade_step
        low, high, _ = self._ladder()
        log_s0 = low + (high - low) * (rating[:, 0] - 1) / 3
        n0 = ((3.0 - rating[:, 0]) / 2.0).clamp(min=0.0)
        states = [torch.stack([log_s0, rating[:, 0] - 1, n0], dim=-1)]
        for t in range(1, rating.shape[1]):
            states.append(self._step(states[-1], elapsed[:, t], rating[:, t]))
        states = torch.stack(states, dim=1)
        return self.decode(states, deltas), states


class SBDModel(AbstractPTModel):
    sequence_features = ["delta_t", "rating"]
    tabular_features = []
    pretrain_lr = 5e-3
    pretrain_epochs = 8
    fine_tune_lr = None
    fine_tune_epochs = None
    patience = None
    batch_size = 1024
    max_batch_tokens = 8_000_000
    recency_weighting = True
    fine_tune_l2 = 1e-3
    l2_sigma_values = L2_SIGMA
    l2_ref_targets = 2000
    fine_tune_max_iter = 200

    def _make_predictor(self):
        return SBDPredictor()

    def load(self):
        pass

    def _l2_sigmas(self):
        return {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in self.l2_sigma_values.items()
        }

    def _fine_tune_loop(self, df):
        store = self._build_store(df, recency=self.recency_weighting)
        batches = [
            store.build(idx)[:5]
            for idx in self._length_capped_batches(store, shuffle=False)
        ]
        n_targets = sum(int(m.sum()) for _, _, _, m, _ in batches)
        if n_targets == 0:
            return {"train": [], "val": []}
        if self.fine_tune_l2 is not None and self.l2_ref_targets is not None:
            self.fine_tune_l2 = self.fine_tune_l2 * self.l2_ref_targets / n_targets
        anchors = {n: p.detach().clone() for n, p in self.predictor.named_parameters()}
        sigmas = self._l2_sigmas()
        denom = sum(
            float(w[m].sum()) if w is not None else int(m.sum())
            for _, _, _, m, w in batches
        )

        def objective():
            total = 0.0
            for feats, deltas, targs, masks, weights in batches:
                if not masks.any():
                    continue
                preds, _ = self.predictor(feats, deltas)
                bce = nn.functional.binary_cross_entropy(
                    preds[masks], targs[masks], reduction="none"
                )
                if weights is not None:
                    bce = bce * weights[masks]
                total = total + bce.sum()
            loss = total / denom
            if self.fine_tune_l2 is not None:
                loss = loss + self._l2_penalty(anchors, sigmas)
            return loss

        optimizer = optim.LBFGS(
            self.predictor.parameters(),
            lr=1.0,
            max_iter=self.fine_tune_max_iter,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )
        losses = []

        def closure():
            optimizer.zero_grad()
            loss = objective()
            if torch.isfinite(loss):
                loss.backward()
            grads_finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in self.predictor.parameters()
            )
            if not torch.isfinite(loss) or not grads_finite:
                optimizer.zero_grad()
                loss = loss.detach().new_tensor(BAD_LOSS)
            losses.append(float(loss.detach()))
            return loss

        self.predictor.train()
        with torch.no_grad():
            start = float(objective())
        try:
            optimizer.step(closure)
        except RuntimeError:
            final = float("nan")
        else:
            with torch.no_grad():
                final = float(objective())
        if not math.isfinite(final) or final > start:
            with torch.no_grad():
                for n, p in self.predictor.named_parameters():
                    p.copy_(anchors[n])
            final = start
        return {"train": [start, final], "val": [], "evals": len(losses)}

    def state_size(self):
        return 3
