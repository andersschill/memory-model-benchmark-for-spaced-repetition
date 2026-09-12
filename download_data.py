import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

RAW_DIR = "./data/raw"
OUTPUT_DIR = "./data/preprocessed"
MAX_REVIEWS_PER_CARD = 128
SAMPLE_SIZE = -1


def process_user(args):
    path, user_id = args
    df = pd.read_parquet(path)
    df["timestamp"] = range(1, len(df) + 1)
    df = df[df.groupby("card_id").cumcount() < MAX_REVIEWS_PER_CARD]
    df["delta_t"] = (df["elapsed_seconds"].clip(lower=0) / (3600 * 24)).astype(
        "float32"
    )
    df["is_inter_day"] = df["elapsed_days"] > 0
    df["target"] = (df["rating"] >= 2).astype("int8")
    df["binary_rating"] = (df["target"] + 1).astype("int8")
    df["session_id"] = df["day_offset"]
    is_first = df.groupby("card_id").cumcount() == 0
    df = df[is_first | (df["delta_t"] > 0)]
    df[
        [
            "timestamp",
            "card_id",
            "duration",
            "rating",
            "binary_rating",
            "target",
            "delta_t",
            "is_inter_day",
            "session_id",
        ]
    ].to_parquet(os.path.join(OUTPUT_DIR, f"user{user_id}.parquet"), index=False)


def main():
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    pathlib.Path(RAW_DIR).mkdir(parents=True, exist_ok=False)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=False)

    print("\033[32mDownload\033[0m")
    REPO_ID = "open-spaced-repetition/anki-revlogs-10k"
    users = None if SAMPLE_SIZE == -1 else list(range(1, SAMPLE_SIZE + 1))
    patterns = (
        ["revlogs/*"] if users is None else [f"revlogs/user_id={u}/*" for u in users]
    )
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", allow_patterns=patterns, local_dir=RAW_DIR
    )

    revlogs = os.path.join(RAW_DIR, "revlogs")
    if users is None:
        users = sorted(
            int(d.split("=")[1])
            for d in os.listdir(revlogs)
            if d.startswith("user_id=")
        )
    jobs = [(os.path.join(revlogs, f"user_id={u}", "data.parquet"), u) for u in users]

    print("\033[32mProcess\033[0m")
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(process_user, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()
    print(f"{len(jobs)} users -> {OUTPUT_DIR}")
    shutil.rmtree(RAW_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
