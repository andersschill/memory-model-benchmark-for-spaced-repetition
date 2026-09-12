from models.sbd import SBDModel, SBDPredictor

INIT_S = [-1.843473, 0.893936]
ROOT = [2.121049]
ALPHA = [-1.442474]
CEXP = [0.457927]
VT = [-1.222906]
LOG_DELTA = [-1.123054]

L2_SIGMA = {
    "init_s": [1.754, 1.2339],
    "root": [0.5003],
    "alpha": [0.546],
    "cexp": [0.0989],
    "vt": [1.1685],
    "log_delta": [0.7778],
}


class SBDBinaryModel(SBDModel):
    sequence_features = ["delta_t", "binary_rating"]
    l2_sigma_values = L2_SIGMA

    def _make_predictor(self):
        return SBDPredictor(INIT_S, ROOT, ALPHA, CEXP, VT, LOG_DELTA, n_grades=2)
