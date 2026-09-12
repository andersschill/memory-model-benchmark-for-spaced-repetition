import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import gc
import hashlib
import pickle
import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score
from models import (
    FSRS7TwoStateModel,
    FSRS7ThreeStateModel,
    SBDModel,
    SBDBinaryModel,
)
import dataloader
import torch
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

EVALUATE_ON_TEST_SET = True
EVALUATE_ON_SAME_DAY_REVIEWS = True
QUERY_COLUMN = "delta_t"
OUTPUT_DIR = "./output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "eval_cache")
METRIC_LABELS = {"bce": "BCE", "rmse_bins": "RMSE(bins)", "auc": "AUC"}


def main():

    models = {
        "FSRS v7 2-state": FSRS7TwoStateModel(
            pretrain=False,
            fine_tune=False,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "fsrs_v7_2_state"),
        ),
        "FSRS v7 2-state fine-tuned": FSRS7TwoStateModel(
            pretrain=False,
            fine_tune=True,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "fsrs_v7_2_state"),
        ),
        "FSRS v7 3-state": FSRS7ThreeStateModel(
            pretrain=False,
            fine_tune=False,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "fsrs_v7_3_state"),
        ),
        "FSRS v7 3-state fine-tuned": FSRS7ThreeStateModel(
            pretrain=False,
            fine_tune=True,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "fsrs_v7_3_state"),
        ),
        "SBD": SBDModel(
            pretrain=False,
            fine_tune=False,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "sbd"),
        ),
        "SBD fine-tuned": SBDModel(
            pretrain=False,
            fine_tune=True,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "sbd"),
        ),
        "SBD Binary": SBDBinaryModel(
            pretrain=False,
            fine_tune=False,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "sbd_binary"),
        ),
        "SBD Binary fine-tuned": SBDBinaryModel(
            pretrain=False,
            fine_tune=True,
            query_column=QUERY_COLUMN,
            loss_on_same_day_reviews=EVALUATE_ON_SAME_DAY_REVIEWS,
            output_dir=os.path.join(OUTPUT_DIR, "sbd_binary"),
        ),
    }

    # ========================================================
    # Print config
    # ========================================================

    print("\033[32mConfig\033[0m")
    print(f"Evaluate on: {'test set' if EVALUATE_ON_TEST_SET else 'validation set'}")
    print(
        f"Same-day reviews: {'included' if EVALUATE_ON_SAME_DAY_REVIEWS else 'excluded'}"
    )
    print(f"Sample size: {dataloader.SAMPLE_SIZE}")
    print(f"Models: {', '.join(list(models.keys()))}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ========================================================
    # Pretrain models
    # ========================================================

    print("\033[32mPretrain\033[0m")

    pretrain_train_data = dataloader.get_pretrain_train_set()
    pretrain_val_data = dataloader.get_pretrain_val_set()
    print(f"Pretrain on: {len(pretrain_train_data)} users")

    for model_name, model in models.items():
        if os.path.exists(model.checkpoint_path()):
            print(f"Loading {model_name} checkpoint from {model.checkpoint_path()}")
            model.load()
        else:
            print(f"Pretraining {model_name}")
            model.pretrain(pretrain_train_data, pretrain_val_data)
            model.save()

    del pretrain_train_data
    gc.collect()

    # ========================================================
    # Fine-tune and evaluate models
    # ========================================================

    print("\033[32mFine-tune and evaluate models\033[0m")

    eval_users = (
        dataloader.get_test_set()
        if EVALUATE_ON_TEST_SET
        else dataloader.get_validation_set()
    )
    eval_files = eval_users.files
    print(f"Fine-tune and evaluate on: {len(eval_users)} users")

    # per-user fits are small and kernel-launch-bound. Parallel CPU workers
    # beat a single GPU
    for model in models.values():
        model.to_device("cpu")
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Evaluating with {n_workers} CPU workers")

    metric_labels = METRIC_LABELS
    results = {name: {m: {} for m in metric_labels} for name in models}
    depth_rows = {name: [] for name in models}

    eval_fingerprint = _eval_fingerprint(eval_files)
    cache_keys = {
        name: _cache_key(model, eval_fingerprint) for name, model in models.items()
    }
    pending = {}
    for model_name, model in models.items():
        entry = _load_cache(cache_keys[model_name])
        if entry is None:
            pending[model_name] = model
            continue
        print(f"Using cached results for {model_name} ({cache_keys[model_name][:12]})")
        results[model_name] = entry["metrics"]
        depth_rows[model_name] = entry["depth"]
        model.fine_tune_losses_per_user.extend(entry["losses"])

    only = os.environ.get("EVAL_ONLY")
    if only:
        keep = {n.strip() for n in only.split(",")}
        for name in [n for n in pending if n not in keep]:
            pending.pop(name)
            results.pop(name)
            depth_rows.pop(name)
            models.pop(name)

    if pending:
        print(f'Evaluating: {", ".join(pending)}')
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(pending,)) as pool:
            for uid, user_out in tqdm(
                pool.imap_unordered(_evaluate_user, eval_users), total=len(eval_users)
            ):
                if user_out is None:
                    continue
                for model_name, out in user_out.items():
                    for metric, value in out["metrics"].items():
                        results[model_name][metric][uid] = value
                    if out["depth"] is not None:
                        depth_rows[model_name].append(out["depth"])
                    models[model_name].fine_tune_losses_per_user.extend(out["losses"])
        for model_name in pending:
            _save_cache(
                cache_keys[model_name],
                {
                    "metrics": results[model_name],
                    "depth": depth_rows[model_name],
                    "losses": models[model_name].fine_tune_losses_per_user,
                },
            )
    else:
        print("All models cached, skipping evaluation")

    for model in models.values():
        model.plot_fine_tuning()

    # ========================================================
    # Print results table
    # ========================================================

    print("\033[32mPrint results\033[0m")

    data = {}
    for model_name, res in results.items():
        model = models[model_name]
        row = {}
        row["Fine-Tuned"] = "x" if model.do_fine_tune else ""
        row["Parameters"] = model.nbr_params()
        row["State Dimensions"] = model.state_size()
        for metric, label in metric_labels.items():
            values = list(res[metric].values())
            row[label] = round(np.mean(values), 4) if values else np.nan
        data[model_name] = row
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Model"
    df = df.sort_values(by="BCE", ascending=True)
    colalign = ["left"] + [
        ("center" if col == "Fine-Tuned" else "right") for col in df.columns
    ]

    def fmt(v):
        return "" if pd.isna(v) else f"{v:.4f}"

    display_df = df.copy()
    display_df["Parameters"] = display_df["Parameters"].map("{:,}".format)
    for label in metric_labels.values():
        display_df[label] = display_df[label].map(fmt)
    output = display_df.to_markdown(colalign=colalign, disable_numparse=True)
    print(output)
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write(output + "\n")

    latex_rows = []
    for model_name, row in df.iterrows():
        cells = [
            f"\\text{{{model_name}}}",
            row["Fine-Tuned"],
            str(row["Parameters"]),
            str(row["State Dimensions"]),
        ]
        cells += [fmt(row[label]) for label in metric_labels.values()]
        latex_rows.append(" & ".join(cells) + " \\\\")

    eval_set = "test" if EVALUATE_ON_TEST_SET else "validation"
    header_cells = [
        "\\textbf{Model}",
        "\\multicolumn{1}{l}{\\textbf{Fine-Tuned}}",
        "\\multicolumn{1}{l}{\\textbf{Parameters}}",
        "\\multicolumn{1}{l}{\\textbf{State Dimensions}}",
    ]
    header_cells += [
        f"\\multicolumn{{1}}{{l}}{{\\textbf{{{label}}}}}"
        for label in metric_labels.values()
    ]
    latex = (
        "\n".join(
            [
                "\\begin{table*}",
                "\\centering",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{lcrr" + "r" * len(metric_labels) + "}",
                "\\hline",
                "",
                " & ".join(header_cells) + " \\\\",
                "",
                "\\hline",
                "",
                *latex_rows,
                "",
                "\\hline",
                "",
                "\\end{tabular}%",
                "}",
                f"\\caption{{Results on {eval_set} set}}",
                "\\label{results-table}",
                "\\end{table*}",
            ]
        )
        + "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "results.tex"), "w") as f:
        f.write(latex)

    # ========================================================
    # Run significance tests
    # ========================================================

    print("\033[32mRun significance tests\033[0m")

    model_order = list(df.index)
    n = len(model_order)

    def pvalue_matrix(metric):
        pvals = np.full((n, n), np.nan)
        for i, name_a in enumerate(model_order):
            for j in range(i + 1, n):
                res_a = results[name_a][metric]
                res_b = results[model_order[j]][metric]
                uids = sorted(set(res_a) & set(res_b))
                if len(uids) < 2:
                    continue
                a = np.array([res_a[u] for u in uids])
                b = np.array([res_b[u] for u in uids])
                test = permutation_test(
                    (a, b),
                    lambda x, y: np.mean(x) - np.mean(y),
                    permutation_type="samples",
                    n_resamples=10_000,
                    random_state=0,
                )
                pvals[i, j] = pvals[j, i] = test.pvalue
        return pvals

    sig_green, not_sig_red, empty_gray = "#cee9ce", "#f4d5d5", "#ececeb"
    fig, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(len(metric_labels) * (1.4 * n + 1.6), 0.8 * n + 1.6),
        squeeze=False,
    )
    for c, (metric, label) in enumerate(metric_labels.items()):
        ax = axes[0][c]
        pvals = pvalue_matrix(metric)
        colors = np.empty((n, n, 3))
        for i in range(n):
            for j in range(n):
                if np.isnan(pvals[i, j]):
                    colors[i, j] = to_rgb(empty_gray)
                else:
                    colors[i, j] = to_rgb(
                        sig_green if pvals[i, j] < 0.05 else not_sig_red
                    )
        ax.imshow(colors)
        for i in range(n):
            for j in range(n):
                if not np.isnan(pvals[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{pvals[i, j]:.4f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#1a1a19",
                        fontweight="bold" if pvals[i, j] < 0.05 else "normal",
                    )
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(n), model_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), model_order if c == 0 else [""] * n, fontsize=8)
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Pairwise p-values, paired permutation test across users (green: p < 0.05)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    sig_path = os.path.join(OUTPUT_DIR, "significance.png")
    fig.savefig(sig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {sig_path}")


# ========================================================
# Result cache
# ========================================================


def _eval_fingerprint(eval_files):
    return "|".join(
        [
            f"test_set={EVALUATE_ON_TEST_SET}",
            f"same_day={EVALUATE_ON_SAME_DAY_REVIEWS}",
            f"metrics={sorted(METRIC_LABELS)}",
            f"query_column={QUERY_COLUMN}",
            f"sample_size={dataloader.SAMPLE_SIZE}",
            f"n_splits={dataloader.N_SPLITS}",
            f"min_reviews={dataloader.MIN_USER_REVIEWS}",
            "personalization_split=strict_before_heldout_timestamp_v1",
            f"users={sorted(eval_files)!r}",
        ]
    )


def _cache_key(model, eval_fingerprint):
    cls = type(model)
    parts = [eval_fingerprint, f"{cls.__module__}.{cls.__qualname__}"]
    for klass in reversed(cls.__mro__):
        for name, value in sorted(vars(klass).items()):
            if name.startswith("_") or callable(value):
                continue
            if isinstance(value, (property, staticmethod, classmethod)):
                continue
            parts.append(f"{klass.__name__}.{name}={value!r}")
    skip = {
        "scaler",
        "predictor",
        "device",
        "output_dir",
        "pretrain_losses",
        "fine_tune_losses_per_user",
    }
    for name, value in sorted(vars(model).items()):
        if name not in skip:
            parts.append(f"{name}={value!r}")
    parts.append(repr(model.predictor))

    h = hashlib.sha256("|".join(parts).encode())
    for name, tensor in sorted(model.predictor.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    for name in ["mean_", "scale_"]:
        arr = getattr(model.scaler, name, None)
        if arr is not None:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _cache_path(key):
    return os.path.join(CACHE_DIR, key + ".pkl")


def _load_cache(key):
    if not os.path.exists(_cache_path(key)):
        return None
    with open(_cache_path(key), "rb") as f:
        return pickle.load(f)


def _save_cache(key, entry):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "wb") as f:
        pickle.dump(entry, f)


# ========================================================
# Helpers
# ========================================================

_worker_models = None


def _init_worker(models):
    global _worker_models
    torch.set_num_threads(1)
    _worker_models = models


def _fold_training_prefix(user_df, fold):
    heldout_times = user_df.loc[user_df["fold"] == fold, "timestamp"]
    if heldout_times.empty:
        raise ValueError("The requested evaluation fold has no rows")
    return user_df[user_df["timestamp"] < heldout_times.min()].assign(in_split=True)


def _evaluate_user(user_df):
    uid = int(user_df["user_id"].iloc[0])
    if user_df[user_df["in_split"]]["target"].nunique() < 2:
        return uid, None
    user_out = {}
    for model_name, model in _worker_models.items():
        n_losses = len(model.fine_tune_losses_per_user)
        if model.do_fine_tune:
            parts = []
            for fold in range(1, int(user_df["fold"].max()) + 1):
                train_df = _fold_training_prefix(user_df, fold)
                prefix = user_df[user_df["fold"] <= fold]
                eval_df = prefix.assign(in_split=prefix["fold"] == fold)
                fold_model = model.copy()
                fold_model.fine_tune(train_df)
                parts.append(scored_rows(fold_model.predict(eval_df)))
            eval_pred_df = pd.concat(parts)
        else:
            eval_pred_df = scored_rows(model.predict(user_df))
        out = {
            "metrics": {},
            "depth": None,
            "losses": model.fine_tune_losses_per_user[n_losses:],
        }
        user_out[model_name] = out
        if eval_pred_df["target"].nunique() < 2:
            continue
        var_df = (
            eval_pred_df
            if EVALUATE_ON_SAME_DAY_REVIEWS
            else eval_pred_df[eval_pred_df["is_inter_day"]]
        )
        if var_df["target"].nunique() >= 2:
            ys = var_df["target"].to_numpy()
            y_hats = var_df["prediction"].to_numpy()
            out["metrics"] = {
                "bce": log_loss(ys, y_hats, labels=[0, 1]),
                "rmse_bins": rmse_bins(var_df),
                "auc": roc_auc_score(ys, y_hats),
            }
        if "n_interday_seen" in eval_pred_df.columns:
            out["depth"] = eval_pred_df[
                ["target", "prediction", "n_interday_seen"]
            ].assign(uid=uid)
    return uid, user_out


def scored_rows(pred_df):
    pred_df = pred_df.drop_duplicates(subset=["timestamp"], keep="first").copy()
    by_card = pred_df.groupby("card_id")
    pred_df["i"] = by_card.cumcount() + 1
    pred_df["elapsed_days"] = by_card["session_id"].diff().fillna(0).astype(int)
    pred_df["interday_i"] = by_card["is_inter_day"].cumsum() + 1
    again = pred_df["rating"] == 1
    if not EVALUATE_ON_SAME_DAY_REVIEWS:
        again &= pred_df["is_inter_day"]
    again = again.astype(int)
    pred_df["lapse"] = again.groupby(pred_df["card_id"]).cumsum() - again
    pred_df = pred_df[pred_df["i"] > 1]
    pred_df = pred_df[pred_df["in_split"]]
    return pred_df


def _log_bin(x, scale, base, decimals):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.round(
        scale * np.power(base, np.floor(np.log(x[pos]) / np.log(base))), decimals
    )
    return out


def rmse_bins(df):
    binned = pd.DataFrame(
        {
            "delta_t": _log_bin(df["elapsed_days"], 2.48, 3.62, 2),
            "i": _log_bin(df["interday_i"], 1.99, 1.89, 0),
            "lapse": _log_bin(df["lapse"], 1.65, 1.73, 0),
            "target": df["target"].to_numpy(dtype=float),
            "prediction": df["prediction"].to_numpy(dtype=float),
        }
    )
    g = binned.groupby(["delta_t", "i", "lapse"]).agg(
        target=("target", "mean"),
        prediction=("prediction", "mean"),
        n=("target", "size"),
    )
    return float(
        np.sqrt(np.average((g["target"] - g["prediction"]) ** 2, weights=g["n"]))
    )


if __name__ == "__main__":
    main()
