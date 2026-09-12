from ._abstract_pt_model import AbstractPTModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

LOWER_BOUNDS = [
    0.0001,
    0.0001,
    0.0001,
    0.0001,
    1.0,
    0.001,
    0.1,
    0.0,
    0.0,
    0.3,
    0.01,
    0.001,
    0.1,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
    0.001,
    0.001,
    0.001,
    0.0,
    0.0,
    1.0,
    2.5,
    0.0,
    0.01,
    0.01,
    0.5,
    0.5,
    0.01,
    0.1,
    0.0,
    0.1,
]
UPPER_BOUNDS = [
    50.0,
    100.0,
    100.0,
    100.0,
    10.0,
    4.0,
    4.0,
    4.0,
    1.2,
    3.0,
    1.5,
    0.9,
    1.0,
    3.5,
    1.0,
    7.0,
    4.0,
    2.0,
    6.0,
    1.5,
    2.0,
    1.0,
    5.0,
    1.0,
    7.0,
    15.0,
    1.0,
    0.25,
    0.95,
    0.85,
    0.99,
    1.0,
    1.0,
    0.9,
    1.1,
]

S_MIN, S_MAX = 0.0001, 36500.0

DEFAULT_PARAMETERS = [
    0.041,
    2.4175,
    4.1283,
    11.9709,
    5.6385,
    0.4468,
    3.262,
    2.3054,
    0.1688,
    1.3325,
    0.3524,
    0.0049,
    0.7503,
    0.0896,
    0.6625,
    1.3,
    0.882,
    0.3072,
    3.5875,
    0.303,
    0.0107,
    0.2279,
    2.6413,
    0.5594,
    1.3,
    2.5,
    1.0,
    0.0723,
    0.1634,
    0.5,
    0.9555,
    0.2245,
    0.6232,
    0.1362,
    0.3862,
]

INIT_S_MAX = 100.0
L2_WEIGHT = 0.5
L2_SIGMA = [
    9999.0,
    9999.0,
    9999.0,
    9999.0,
    0.523,
    0.2528,
    0.4329,
    0.2966,
    0.2139,
    0.2889,
    0.1862,
    0.0829,
    0.175,
    0.3812,
    0.3013,
    0.9104,
    0.3234,
    0.2448,
    0.3273,
    0.1842,
    0.1542,
    0.1735,
    0.4608,
    0.311,
    0.864,
    0.4053,
    0.162,
    0.0418,
    0.2596,
    0.0798,
    0.0682,
    0.1282,
    0.1397,
    0.1407,
    0.1489,
]
CURVE_CANDIDATES = [
    DEFAULT_PARAMETERS[27:],
    [0.0594, 0.3358, 0.598, 0.9517, 0.3122, 0.5685, 0.2371, 0.4871],
    [0.0441, 0.2533, 0.6823, 0.9598, 0.3613, 0.5202, 0.2283, 0.4783],
    [0.0621, 0.2475, 0.6496, 0.9744, 0.313, 0.5662, 0.2336, 0.4836],
    [0.0462, 0.2962, 0.6938, 0.9592, 0.3341, 0.5273, 0.2185, 0.4685],
    [0.0422, 0.2813, 0.6713, 0.9421, 0.2935, 0.5985, 0.2183, 0.4683],
    [0.0568, 0.1563, 0.6567, 0.9633, 0.3682, 0.5041, 0.1952, 0.4452],
    [0.0651, 0.2502, 0.6682, 0.9472, 0.3757, 0.4933, 0.2408, 0.4908],
    [0.0548, 0.1655, 0.6138, 0.9654, 0.3251, 0.5717, 0.1418, 0.3918],
    [0.0381, 0.2803, 0.7202, 0.9491, 0.3362, 0.5166, 0.2248, 0.4748],
    [0.0422, 0.1935, 0.694, 0.9549, 0.3871, 0.4704, 0.2413, 0.4913],
    [0.0651, 0.1916, 0.623, 0.972, 0.3528, 0.5484, 0.2373, 0.4873],
    [0.0508, 0.3743, 0.5863, 0.9448, 0.2974, 0.606, 0.1444, 0.3944],
    [0.0498, 0.3753, 0.6875, 0.9319, 0.3758, 0.4984, 0.2268, 0.4768],
    [0.0618, 0.1663, 0.5977, 0.9682, 0.3619, 0.5066, 0.2972, 0.5472],
    [0.0656, 0.197, 0.5693, 0.9692, 0.3599, 0.5374, 0.2596, 0.5096],
]
S0_LOG_ANCHORS = {1: -8.09, 2: -3.83, 3: -2.5, 4: -1.0}


def _np_retrievability(t, s, p):
    d1, d2, b1, b2, w1, w2, p1, p2 = p
    t_over_s = t / s
    r1 = (1 + (b1 ** (-1 / d1) - 1) * t_over_s) ** -d1
    r2 = (1 + (b2 ** (-1 / d2) - 1) * t_over_s) ** -d2
    weight1 = w1 * s**-p1
    weight2 = w2 * s**p2
    return (weight1 * r1 + weight2 * r2) / (weight1 + weight2)


def _bin_interval(t):
    t = np.asarray(t, dtype=np.float64)
    ten_min, two_hours = 10 / 1440, 2 / 24
    out = np.empty_like(t)
    short = t < two_hours
    out[short] = np.maximum(np.floor(t[short] / ten_min) * ten_min, ten_min)
    medium = (t >= two_hours) & (t < 1.0)
    out[medium] = np.maximum(np.floor(t[medium] / two_hours) * two_hours, two_hours)
    long = t >= 1.0
    out[long] = np.maximum(np.floor(t[long]), 1.0)
    return out


def _interpolate_s0(fitted):
    known = sorted(fitted)
    log_s = {r: np.log(fitted[r]) for r in known}
    a = S0_LOG_ANCHORS

    def interp(target, lo, hi):
        f = (a[target] - a[lo]) / (a[hi] - a[lo])
        return log_s[lo] + f * (log_s[hi] - log_s[lo])

    def extrap(target, anchor):
        return log_s[anchor] + (a[target] - a[anchor])

    if len(known) == 3:
        m = next(r for r in (1, 2, 3, 4) if r not in known)
        if m == 1:
            log_s[1] = extrap(1, 2)
        elif m == 2:
            log_s[2] = interp(2, 1, 3)
        elif m == 3:
            log_s[3] = interp(3, 2, 4)
        else:
            log_s[4] = extrap(4, 3)
    elif known == [1, 2]:
        log_s[3] = extrap(3, 2)
        log_s[4] = extrap(4, 3)
    elif known == [1, 3]:
        log_s[2] = interp(2, 1, 3)
        log_s[4] = extrap(4, 3)
    elif known == [1, 4]:
        log_s[2] = interp(2, 1, 4)
        log_s[3] = interp(3, 1, 4)
    elif known == [2, 3]:
        log_s[1] = extrap(1, 2)
        log_s[4] = extrap(4, 3)
    elif known == [2, 4]:
        log_s[3] = interp(3, 2, 4)
        log_s[1] = extrap(1, 2)
    else:
        log_s[2] = extrap(2, 3)
        log_s[1] = extrap(1, 2)

    s0 = {
        r: float(round(np.clip(np.exp(log_s[r]), 0.0001, 100), 4)) for r in (1, 2, 3, 4)
    }
    for lo, hi in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        if s0[lo] > s0[hi]:
            if lo in known and hi not in known:
                s0[hi] = s0[lo]
            elif hi in known and lo not in known:
                s0[lo] = s0[hi]
            else:
                s0[hi], s0[lo] = s0[lo], s0[hi]
    return [s0[r] for r in (1, 2, 3, 4)]


def _fit_initial_params(df):
    scorable = df["delta_t"].to_numpy() > 0
    first_rating = df.groupby("card_id")["rating"].transform("first").to_numpy()
    i = df["is_inter_day"].astype(int).groupby(df["card_id"]).cumsum().to_numpy() + 1
    sel = (i == 2) & scorable
    grouped = (
        pd.DataFrame(
            {
                "first_rating": first_rating[sel],
                "bin": _bin_interval(df["delta_t"].to_numpy()[sel]),
                "y": df["target"].to_numpy()[sel],
            }
        )
        .groupby(["first_rating", "bin"])["y"]
        .agg(["mean", "count"])
        .reset_index()
    )
    average_recall = df["target"].to_numpy()[scorable].mean() if scorable.any() else 0.0

    def evaluate_candidate(p):
        stabilities, counts, total = {}, {}, 0.0
        for r in (1, 2, 3, 4):
            g = grouped[grouped["first_rating"] == r]
            if g.empty:
                continue
            dt = g["bin"].to_numpy()
            cnt = g["count"].to_numpy()
            recall = (g["mean"].to_numpy() * cnt + average_recall) / (cnt + 1)
            init_s0 = DEFAULT_PARAMETERS[r - 1]

            def loss(s):
                y_pred = np.clip(_np_retrievability(dt, s[0], p), 0.0001, 0.9999)
                logloss = np.sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred)) * cnt
                )
                return logloss + abs(s[0] - init_s0) / 16

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((S_MIN, INIT_S_MAX),),
                options={"maxiter": int(cnt.sum())},
            )
            stabilities[r] = float(res.x[0])
            counts[r] = int(cnt.sum())
            total += float(res.fun)
        for lo, hi in [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]:
            if (
                lo in stabilities
                and hi in stabilities
                and stabilities[lo] > stabilities[hi]
            ):
                if counts[lo] > counts[hi]:
                    stabilities[hi] = stabilities[lo]
                else:
                    stabilities[lo] = stabilities[hi]
        return total, stabilities

    results = [(evaluate_candidate(p), p) for p in CURVE_CANDIDATES]
    results.sort(key=lambda x: x[0][0])
    (_, fitted), curve = results[0]

    if len(fitted) == 0:
        s0 = list(DEFAULT_PARAMETERS[:4])
    elif len(fitted) == 1:
        r, v = next(iter(fitted.items()))
        factor = v / DEFAULT_PARAMETERS[r - 1]
        s0 = [x * factor for x in DEFAULT_PARAMETERS[:4]]
    elif len(fitted) == 4:
        s0 = [fitted[r] for r in (1, 2, 3, 4)]
    else:
        s0 = _interpolate_s0(fitted)
    s0 = [min(max(x, S_MIN), INIT_S_MAX) for x in s0]
    return s0, list(curve)


def _retrievability(w, t, s):
    t_over_s = t / s
    decay1, decay2 = -w[27], -w[28]
    base1, base2 = w[29], w[30]
    r1 = (1 + (base1 ** (1 / decay1) - 1) * t_over_s) ** decay1
    r2 = (1 + (base2 ** (1 / decay2) - 1) * t_over_s) ** decay2
    weight1 = w[31] * s ** -w[33]
    weight2 = w[32] * s ** w[34]
    return (weight1 * r1 + weight2 * r2) / (weight1 + weight2)


def _initial_difficulty(w, g):
    return w[4] - torch.exp(w[5] * (g - 1)) + 1


def _stability_after_review(w, i, s, d, g, r):
    hard_penalty = torch.where(g == 2, w[i + 7], torch.ones_like(g))
    easy_bonus = torch.where(g == 4, w[i + 8], torch.ones_like(g))
    s_fail = (
        w[i + 3]
        * d ** (-w[i + 4])
        * ((s + 1) ** w[i + 5] - 1)
        * torch.exp(w[i + 6] * (1 - r))
    )
    s_fail = torch.minimum(s_fail, s)
    s_inc = (
        1
        + torch.exp(w[i] - 1.5)
        * (11 - d)
        * s ** (-w[i + 1])
        * (torch.exp(w[i + 2] * (1 - r)) - 1)
        * hard_penalty
        * easy_bonus
    )
    s_success = torch.maximum(s_fail, s * s_inc)
    return torch.where(g == 1, s_fail, s_success)


def _update_state(w, s, d, g, dt):
    r = _retrievability(w, dt, s)
    s_long = _stability_after_review(w, 7, s, d, g, r)
    s_short = _stability_after_review(w, 16, s, d, g, r)
    c = 1 - w[26] * torch.exp(-w[25] * dt)
    s = (c * s_long + (1 - c) * s_short).clamp(S_MIN, S_MAX)
    d_damped = d + (-w[6] * (g - 3)) * (10 - d) / 9
    d0_easy = _initial_difficulty(w, torch.full_like(g, 4.0))
    d = torch.clamp(0.01 * d0_easy + 0.99 * d_damped, 1.0, 10.0)
    return s, d


def _first_step(w, g, query):
    s = w[(g - 1).long()].clamp(S_MIN, S_MAX)
    d = _initial_difficulty(w, g).clamp(1.0, 10.0)
    return s, d, _retrievability(w, query, s)


def _next_step(w, s, d, g, dt, query):
    s, d = _update_state(w, s, d, g, dt)
    return s, d, _retrievability(w, query, s)


_first_step_c = torch.compile(_first_step, dynamic=True)
_next_step_c = torch.compile(_next_step, dynamic=True)


class FSRS7TwoStatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(DEFAULT_PARAMETERS, dtype=torch.float32))
        self.register_buffer("w_min", torch.tensor(LOWER_BOUNDS, dtype=torch.float32))
        self.register_buffer("w_max", torch.tensor(UPPER_BOUNDS, dtype=torch.float32))
        self.clip_parameters()

    def clip_parameters(self):
        with torch.no_grad():
            w = self.w.data
            w.clamp_(self.w_min, self.w_max)
            w[1] = torch.maximum(w[1], w[0])
            w[2] = torch.maximum(w[2], w[1])
            w[3] = torch.maximum(w[3], w[2])
            w[28] = torch.maximum(w[28], w[27])
            w[30] = torch.maximum(w[30], w[29])

    def forward(self, x, deltas):
        w = self.w
        n = x.shape[0]
        n_pad = 1 << max(3, (n - 1).bit_length())
        if n_pad != n:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, n_pad - n))
            deltas = nn.functional.pad(deltas, (0, 0, 0, 0, 0, n_pad - n))
        elapsed = x[..., 0].clamp(min=0.0)
        rating = x[..., 1].clamp(1.0, 4.0)
        preds = []
        s = d = None
        for t in range(rating.shape[1]):
            g = rating[:, t]
            if t == 0:
                s, d, p = _first_step_c(w, g, deltas[:, t, 0])
            else:
                s, d, p = _next_step_c(w, s, d, g, elapsed[:, t], deltas[:, t, 0])
            preds.append(p)
        p = torch.stack(preds, dim=1).clamp(0.0001, 0.9999)[:n]
        return p, None


class FSRS7TwoStateModel(AbstractPTModel):
    sequence_features = ["delta_t", "rating"]
    tabular_features = []
    pretrain_lr = 2e-2
    fine_tune_lr = 2e-2
    grad_clip = None
    pretrain_epochs = 8
    fine_tune_epochs = 8
    patience = None
    batch_size = 1024
    max_batch_tokens = 8_000_000
    recency_weighting = True
    max_history_len = 64
    fine_tune_l2 = L2_WEIGHT
    fine_tune_snapshots = True
    scale_steps_to_rows = True
    summed_loss = True

    def _make_predictor(self):
        return FSRS7TwoStatePredictor()

    def _fine_tune_init(self, train_df):
        s0, curve = _fit_initial_params(train_df)
        with torch.no_grad():
            w = self.predictor.w.data
            w[0:4] = w.new_tensor(s0)
            w[27:35] = w.new_tensor(curve)

    def _l2_sigmas(self):
        return {"w": torch.tensor(L2_SIGMA)}

    def _make_optimizer(self, lr):
        return optim.Adam(self.predictor.parameters(), lr=lr, betas=(0.8, 0.85))

    def _make_scheduler(self, optimizer, epochs, n_batches):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs * n_batches, 1)
        )

    def _scheduler_batch_step(self, scheduler):
        scheduler.step()

    def _scheduler_epoch_step(self, scheduler, metric):
        pass

    def _after_optimizer_step(self):
        self.predictor.clip_parameters()

    def state_size(self):
        return 2
