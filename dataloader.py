import hashlib
import os
import random
import numpy as np
import pandas as pd

SAMPLE_SIZE = -1
PRETRAIN_FINETUNE_USER_SPLIT = 0.5
EVAL_TEST_USER_SPLIT = 0.5
N_SPLITS = 5
MIN_USER_REVIEWS = N_SPLITS + 1
PRETRAIN_VAL_STRIDE = 5
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "preprocessed"
)


def _is_finetune_user(fname):
    h = int(hashlib.md5(fname.encode()).hexdigest(), 16)
    return (h % 10_000) < PRETRAIN_FINETUNE_USER_SPLIT * 10_000


def _is_test_user(fname):
    h = int(hashlib.md5(("eval:" + fname).encode()).hexdigest(), 16)
    return (h % 10_000) < EVAL_TEST_USER_SPLIT * 10_000


def _get_user_splits():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    if 0 < SAMPLE_SIZE < len(files):
        files = sorted(random.Random(42).sample(files, SAMPLE_SIZE))
    finetune = [f for f in files if _is_finetune_user(f)]
    pretrain = [f for f in files if not _is_finetune_user(f)]
    return pretrain, finetune


def _validation_test_files():
    _, finetune_files = _get_user_splits()
    validation = [f for f in finetune_files if not _is_test_user(f)]
    test = [f for f in finetune_files if _is_test_user(f)]
    return validation, test


def _load_user(fname, uid):
    df = pd.read_parquet(os.path.join(DATA_DIR, fname))
    df["user_id"] = uid
    return df


def _fold_bounds(n):
    test_size = n // (N_SPLITS + 1)
    return [
        (n - (N_SPLITS - i) * test_size, n - (N_SPLITS - i - 1) * test_size)
        for i in range(N_SPLITS)
    ]


def _mark_folds(df):
    scorable = np.flatnonzero(df["delta_t"].to_numpy() > 0)
    fold = np.zeros(len(df), dtype=int)
    for k, (start, end) in enumerate(_fold_bounds(len(scorable)), 1):
        fold[scorable[start:end]] = k
    return df.assign(fold=fold, in_split=fold > 0)


def _prepared_or_none(df):
    n_scorable = int((df["delta_t"].to_numpy() > 0).sum())
    return _mark_folds(df) if n_scorable >= MIN_USER_REVIEWS else None


class _UserStream:
    def __init__(self, files, transform):
        self._files = files
        self._transform = transform

    def __len__(self):
        return len(self._files)

    @property
    def files(self):
        return list(self._files)

    def __iter__(self):
        for uid, fname in enumerate(self._files):
            df = self._transform(_load_user(fname, uid))
            if df is not None:
                yield df


def get_pretrain_train_set():
    pretrain_files, _ = _get_user_splits()
    return _UserStream(pretrain_files, lambda df: df)


def get_pretrain_val_set():
    validation, _ = _validation_test_files()
    return _UserStream(validation[::PRETRAIN_VAL_STRIDE], _prepared_or_none)


def get_validation_set():
    validation, _ = _validation_test_files()
    return _UserStream(validation, _prepared_or_none)


def get_test_set():
    _, test = _validation_test_files()
    return _UserStream(test, _prepared_or_none)
