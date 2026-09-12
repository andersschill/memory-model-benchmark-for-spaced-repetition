from ._abstract_pt_model import AbstractPTModel
import torch
import torch.nn as nn
import torch.optim as optim

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
    0.1,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.5,
    0.001,
    0.001,
    0.0,
    0.0,
    1.0,
    0.01,
    0.01,
    0.2,
    0.5,
    0.01,
    0.1,
    0.0,
    0.1,
    0.0,
    0.0,
    0.0,
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
    1.0,
    3.5,
    1.0,
    7.0,
    4.0,
    2.0,
    6.0,
    1.5,
    1.0,
    5.0,
    1.0,
    7.0,
    0.25,
    0.95,
    0.85,
    0.99,
    1.0,
    1.0,
    0.9,
    1.1,
    1.0,
    0.6,
    0.6,
]

S_MIN, S_MAX = 0.0001, 36500.0

DEFAULT_PARAMETERS = [
    0.1104,
    2.2395,
    3.9221,
    11.7841,
    6.1686,
    0.6457,
    3.6807,
    1.9795,
    0.0,
    1.3826,
    0.7024,
    0.5999,
    0.8146,
    0.6398,
    1.0,
    1.3207,
    0.6707,
    3.8668,
    0.4416,
    0.0934,
    1.8631,
    0.6162,
    1.0869,
    0.1567,
    0.0801,
    0.2421,
    0.9464,
    0.1433,
    0.7145,
    0.0,
    0.5667,
    0.3734,
    0.5333,
    0.3048,
]

L2_WEIGHT = 0.3333
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
    0.175,
    0.3812,
    0.3013,
    0.9104,
    0.3234,
    0.2448,
    0.3273,
    0.1842,
    0.1735,
    0.4608,
    0.311,
    0.864,
    0.0418,
    0.2596,
    0.0798,
    0.0682,
    0.1282,
    0.1397,
    0.1407,
    0.1489,
    0.2,
    0.15,
    0.15,
]


def _short_recall(w, t, s_short):
    t = t.clamp(min=0.0)
    decay1 = -(w[23] * s_short ** (w[33] - 0.3)).clamp(0.01, 0.95)
    factor1 = (torch.log(w[25]) / decay1).clamp(max=60.0).exp() - 1
    return (t / s_short * factor1 + 1) ** decay1


def _retrievability(w, t, s, s_short, d):
    t = t.clamp(min=0.0)
    r1 = _short_recall(w, t, s_short)
    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26] ** (1.0 / decay2) - 1
    d_timescale = torch.exp((d - 5.0) * (w[32] - 0.3))
    r2 = (t / s * factor2 * d_timescale + 1) ** decay2
    weight1 = w[27] * s_short ** -w[29]
    weight2 = w[28] * s ** w[30] * torch.exp((d - 5.0) * (w[31] - 0.5))
    retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    return retention * (1.0 - 2e-5) + 1e-5


def _initial_difficulty(w, g):
    return w[4] - torch.exp(w[5] * (g - 1)) + 1


def _stability_after_review(w, i, s, d, g, r):
    hard_penalty = torch.where(g == 2, w[i + 6], torch.ones_like(g))
    easy_bonus = torch.where(g == 4, w[i + 7], torch.ones_like(g))
    s_fail = w[i + 3] * ((s + 1) ** w[i + 4] - 1) * torch.exp(w[i + 5] * (1 - r))
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


def _next_difficulty(w, d, g, r):
    delta_d = -w[6] * (g - 3)
    delta_d = torch.where(g == 1, delta_d * (r + 0.1), delta_d)
    d_damped = d + delta_d * (10 - d) / 9
    d0_easy = _initial_difficulty(w, torch.full_like(g, 4.0))
    return torch.clamp(0.01 * d0_easy + 0.99 * d_damped, 1.0, 10.0)


def _update_state(w, s, ss, d, g, dt):
    r = _retrievability(w, dt, s, ss, d)
    s_long = _stability_after_review(w, 7, s, d, g, r)
    r1 = _short_recall(w, dt, ss)
    s_short = _stability_after_review(w, 15, ss, d, g, r1)
    s_short = torch.where(g == 1, torch.minimum(s_short, 0.8 * s_long), s_short)
    d = _next_difficulty(w, d, g, r)
    return s_long.clamp(S_MIN, S_MAX), s_short.clamp(S_MIN, S_MAX), d


def _first_step(w, g, query):
    s = w[(g - 1).long()].clamp(S_MIN, S_MAX)
    ss = (0.8 * s).clamp(S_MIN, S_MAX)
    d = _initial_difficulty(w, g).clamp(1.0, 10.0)
    return s, ss, d, _retrievability(w, query, s, ss, d)


def _next_step(w, s, ss, d, g, dt, query):
    s, ss, d = _update_state(w, s, ss, d, g, dt)
    return s, ss, d, _retrievability(w, query, s, ss, d)


_first_step_c = torch.compile(_first_step, dynamic=True)
_next_step_c = torch.compile(_next_step, dynamic=True)


class FSRS7ThreeStatePredictor(nn.Module):
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
            w[26] = torch.maximum(w[26], w[25])

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
        s = ss = d = None
        for t in range(rating.shape[1]):
            g = rating[:, t]
            if t == 0:
                s, ss, d, p = _first_step_c(w, g, deltas[:, t, 0])
            else:
                s, ss, d, p = _next_step_c(
                    w, s, ss, d, g, elapsed[:, t], deltas[:, t, 0]
                )
            preds.append(p)
        p = torch.stack(preds, dim=1).clamp(0.0, 1.0)[:n]
        return p, None


class FSRS7ThreeStateModel(AbstractPTModel):
    sequence_features = ["delta_t", "rating"]
    tabular_features = []
    pretrain_lr = 0.0118
    fine_tune_lr = 0.0118
    grad_clip = None
    pretrain_epochs = 9
    fine_tune_epochs = 9
    patience = None
    batch_size = 512
    max_batch_tokens = 8_000_000
    recency_weighting = True
    max_history_len = 64
    fine_tune_l2 = L2_WEIGHT
    fine_tune_snapshots = False
    keep_final_state = True
    scale_steps_to_rows = True
    summed_loss = True

    def _make_predictor(self):
        return FSRS7ThreeStatePredictor()

    def _l2_sigmas(self):
        return {"w": torch.tensor(L2_SIGMA)}

    def _make_optimizer(self, lr):
        return optim.Adam(self.predictor.parameters(), lr=lr, betas=(0.7, 0.98))

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
        return 3
