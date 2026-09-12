import copy
import os
import queue
import random
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ._abstract_model import Model
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class SeqStore:

    def __init__(
        self,
        seq_arr,
        tab_arr,
        query,
        target,
        weight,
        mask,
        starts,
        ends,
        device,
        row_labels,
    ):
        self.seq_arr = seq_arr
        self.tab_arr = tab_arr
        self.query = query
        self.target = target
        self.weight = weight
        self.mask = mask
        self.starts = starts
        self.ends = ends
        self.device = device
        self.row_labels = row_labels

    def __len__(self):
        return len(self.starts)

    @property
    def lengths(self):
        return self.ends - self.starts - 1

    def subset(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        return SeqStore(
            self.seq_arr,
            self.tab_arr,
            self.query,
            self.target,
            self.weight,
            self.mask,
            self.starts[idxs],
            self.ends[idxs],
            self.device,
            self.row_labels,
        )

    def build(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        starts = self.starts[idxs]
        seq_lens = self.ends[idxs] - starts - 1

        total = int(seq_lens.sum())
        seg_start = np.repeat(np.cumsum(seq_lens) - seq_lens, seq_lens)
        pos = np.arange(total) - seg_start
        hist_idx = np.repeat(starts, seq_lens) + pos
        targ_idx = hist_idx + 1

        feats = self.seq_arr[hist_idx]
        if self.tab_arr is not None:
            feats = np.concatenate([feats, self.tab_arr[targ_idx]], axis=-1)
        mask_flat = (
            np.ones(total, dtype=bool) if self.mask is None else self.mask[targ_idx]
        )
        deltas_flat = self.query[targ_idx]
        targs_flat = self.target[targ_idx]

        dev = self.device
        split = seq_lens.tolist()

        B = len(idxs)
        max_len = int(seq_lens.max()) if B else 0
        seq_id = np.repeat(np.arange(B), seq_lens)

        feat_pad = np.zeros((B, max_len, feats.shape[1]), dtype=np.float32)
        feat_pad[seq_id, pos] = feats
        delta_pad = np.zeros((B, max_len, 1), dtype=np.float32)
        delta_pad[seq_id, pos, 0] = deltas_flat
        targ_pad = np.zeros((B, max_len), dtype=np.float32)
        targ_pad[seq_id, pos] = targs_flat
        mask_pad = np.zeros((B, max_len), dtype=bool)
        mask_pad[seq_id, pos] = mask_flat

        feat_seqs = torch.as_tensor(feat_pad, dtype=torch.float32, device=dev)
        delta_seqs = torch.as_tensor(delta_pad, dtype=torch.float32, device=dev)
        targ_seqs = torch.as_tensor(targ_pad, dtype=torch.float32, device=dev)
        targ_masks = torch.as_tensor(mask_pad, dtype=torch.bool, device=dev)
        weight_seqs = None
        if self.weight is not None:
            weight_pad = np.zeros((B, max_len), dtype=np.float32)
            weight_pad[seq_id, pos] = self.weight[targ_idx]
            weight_seqs = torch.as_tensor(weight_pad, dtype=torch.float32, device=dev)
        return feat_seqs, delta_seqs, targ_seqs, targ_masks, weight_seqs, split


class AbstractPTModel(Model, ABC):
    grad_clip = 1.0
    lr_patience = 1
    batch_size = 2048
    max_batch_tokens = 150_000
    max_seq_len = 512
    scale_features = False
    weight_decay = 0.0
    ema_decay = None
    recency_weighting = False
    max_history_len = None
    fine_tune_l2 = None
    fine_tune_snapshots = False
    scale_steps_to_rows = False
    row_batch_threshold = 32_768
    summed_loss = False
    recency_floor = 0.0667
    recency_power = 11.25
    l2_ref_targets = None
    keep_final_state = False
    failure_weight = None

    @property
    @abstractmethod
    def sequence_features(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def tabular_features(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def pretrain_lr(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def fine_tune_lr(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def pretrain_epochs(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def fine_tune_epochs(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def patience(self):
        raise NotImplementedError

    def __init__(
        self, pretrain, fine_tune, query_column, loss_on_same_day_reviews, output_dir
    ):
        super().__init__(pretrain, fine_tune, output_dir)
        self.query_column = query_column
        # when False, same-day reviews feed the history but are not loss targets
        self.loss_on_same_day_reviews = loss_on_same_day_reviews
        self.scaler = StandardScaler()
        self.features = list(
            dict.fromkeys(self.sequence_features + self.tabular_features)
        )
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.pretrain_losses = None
        self.fine_tune_losses_per_user = []
        self.predictor = self._make_predictor().to(self.device)

    @abstractmethod
    def _make_predictor(self):
        raise NotImplementedError

    def _make_optimizer(self, lr):
        return optim.AdamW(
            self.predictor.parameters(), lr=lr, weight_decay=self.weight_decay
        )

    def _make_scheduler(self, optimizer, epochs, n_batches):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=self.lr_patience
        )

    def _scheduler_batch_step(self, scheduler):
        pass

    def _scheduler_epoch_step(self, scheduler, metric):
        scheduler.step(metric)

    def _backward_loss(self, loss, n, avg_n):
        return loss * (n / avg_n)

    def _after_optimizer_step(self):
        pass

    def copy(self):
        # share the loss list so per-user fine-tuning accumulates on the original
        c = copy.deepcopy(self)
        c.fine_tune_losses_per_user = self.fine_tune_losses_per_user
        return c

    def to_device(self, device):
        self.device = torch.device(device)
        self.predictor.to(self.device)

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(
            {"state_dict": self.predictor.state_dict(), "scaler": self.scaler},
            self.checkpoint_path(),
        )

    def load(self):
        ckpt = torch.load(
            self.checkpoint_path(), map_location=self.device, weights_only=False
        )
        self.predictor.load_state_dict(ckpt["state_dict"])
        self.scaler = ckpt["scaler"]

    def nbr_params(self):
        return sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)

    def _plot_pretraining(self, losses):
        if self.output_dir is None:
            return
        plt.figure()
        plt.plot(losses["train"], label="train")
        plt.plot(losses["val"], label="val")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "pretraining.png"))
        plt.close()

    def plot_fine_tuning(self):
        if not self.fine_tune_losses_per_user or self.output_dir is None:
            return
        plt.figure()
        for losses in self.fine_tune_losses_per_user:
            plt.plot(losses["train"])
        plt.savefig(os.path.join(self.output_dir, "fine_tuning.png"))
        plt.close()

    def pretrain(self, train_df, val_df=None):
        if self.do_pretrain:
            self.pretrain_losses = self._train(
                train_df,
                self.pretrain_lr,
                self.pretrain_epochs,
                fit_scaler=True,
                verbose=True,
                use_ema=True,
                val_df=val_df,
                val_frac=0.1 if val_df is None else 0.0,
            )

    def fine_tune(self, train_df, val_df=None):
        if self.do_fine_tune:
            self.fine_tune_losses_per_user.append(self._fine_tune_loop(train_df))

    def _fine_tune_init(self, train_df):
        pass

    def _l2_sigmas(self):
        return {}

    def _l2_penalty(self, anchors, sigmas):
        total = None
        for name, p in self.predictor.named_parameters():
            d = (p - anchors[name]) ** 2
            sigma = sigmas.get(name)
            if sigma is not None:
                d = d / sigma.to(p.device) ** 2
            d = d.sum()
            total = d if total is None else total + d
        return self.fine_tune_l2 * total

    def _fine_tune_loop(self, df):
        if self.max_history_len is not None:
            keep = df.groupby("card_id").cumcount().to_numpy() <= self.max_history_len
            if "in_split" in df.columns:
                keep &= df["in_split"].to_numpy()
            df = df.assign(in_split=keep)

        store = self._build_store(
            df, fit_scaler=not self.do_pretrain, recency=self.recency_weighting
        )
        if store.mask is None:
            n_targets = int((store.ends - store.starts - 1).sum())
        else:
            n_targets = int(
                sum(
                    store.mask[s + 1 : e].sum()
                    for s, e in zip(store.starts, store.ends)
                )
            )
        if n_targets == 0:
            return {"train": [], "val": []}

        self._fine_tune_init(df)
        if self.fine_tune_l2 is not None and self.l2_ref_targets is not None:
            self.fine_tune_l2 = self.fine_tune_l2 * self.l2_ref_targets / n_targets
        anchors = sigmas = None
        if self.fine_tune_l2 is not None:
            anchors = {
                n: p.detach().clone() for n, p in self.predictor.named_parameters()
            }
            sigmas = self._l2_sigmas()

        cls_batch = type(self).batch_size
        if self.scale_steps_to_rows and n_targets > self.row_batch_threshold:
            self.batch_size = max(1, round(cls_batch * len(store) / n_targets))
        batches = [
            store.build(idx)[:5]
            for idx in self._length_capped_batches(store, shuffle=False)
        ]
        n_batches = len(batches)
        epochs, eval_every = self.fine_tune_epochs, 1
        if self.scale_steps_to_rows:
            steps = self.fine_tune_epochs * -(-n_targets // cls_batch)
            epochs = max(epochs, round(steps / max(n_batches, 1)))
            eval_every = max(1, round(epochs / self.fine_tune_epochs))

        optimizer = self._make_optimizer(self.fine_tune_lr)
        scheduler = self._make_scheduler(optimizer, epochs, n_batches)
        avg_n = max(float(n_targets) / max(n_batches, 1), 1.0)

        def batch_bce(feats, deltas, targs, masks, weights):
            preds, _ = self.predictor(feats, deltas)
            bce = nn.functional.binary_cross_entropy(
                preds[masks], targs[masks], reduction="none"
            )
            if weights is not None:
                bce = bce * weights[masks]
            if self.failure_weight is not None:
                bce = bce * (1 + (self.failure_weight - 1) * (1 - targs[masks]))
            return bce

        def clean_loss():
            self.predictor.eval()
            with torch.no_grad():
                total = 0.0
                for feats, deltas, targs, masks, weights in batches:
                    if masks.any():
                        total += float(
                            batch_bce(feats, deltas, targs, masks, weights).sum()
                        )
                if anchors is not None:
                    total += float(self._l2_penalty(anchors, sigmas))
                return total / n_targets

        best_loss, best_state, counter = float("inf"), None, 0
        train_losses = []
        if self.fine_tune_snapshots:
            best_loss = clean_loss()
            best_state = copy.deepcopy(self.predictor.state_dict())
            train_losses.append(best_loss)
        order = list(range(n_batches))
        for epoch in range(epochs):
            self.predictor.train()
            random.shuffle(order)
            epoch_loss, epoch_n = 0.0, 0
            for feats, deltas, targs, masks, weights in (batches[i] for i in order):
                if not masks.any():
                    continue
                optimizer.zero_grad()
                bce = batch_bce(feats, deltas, targs, masks, weights)
                n = int(masks.sum())
                if self.summed_loss:
                    loss = bce.sum()
                    backward = loss
                else:
                    denom = weights[masks].sum() if weights is not None else n
                    loss = bce.sum() / denom
                    backward = self._backward_loss(loss, n, avg_n)
                if anchors is not None:
                    backward = backward + self._l2_penalty(anchors, sigmas) * (
                        n / n_targets
                    )
                backward.backward()
                if self.grad_clip is not None:
                    gnorm = nn.utils.clip_grad_norm_(
                        self.predictor.parameters(), self.grad_clip
                    )
                    if not torch.isfinite(gnorm):
                        continue
                optimizer.step()
                self._after_optimizer_step()
                self._scheduler_batch_step(scheduler)
                epoch_loss += loss.item() * n
                epoch_n += n
            if self.fine_tune_snapshots:
                if (epoch + 1) % eval_every and epoch != epochs - 1:
                    continue
                metric = clean_loss()
                train_losses.append(metric)
                self._scheduler_epoch_step(scheduler, metric)
                if metric < best_loss:
                    best_loss = metric
                    best_state = copy.deepcopy(self.predictor.state_dict())
            else:
                metric = epoch_loss / max(epoch_n, 1)
                train_losses.append(metric)
                self._scheduler_epoch_step(scheduler, metric)
                if metric < best_loss - 1e-4:
                    best_loss, counter = metric, 0
                    best_state = copy.deepcopy(self.predictor.state_dict())
                else:
                    counter += 1
                    if self.patience is not None and counter >= self.patience:
                        break
        if best_state is not None and not self.keep_final_state:
            self.predictor.load_state_dict(best_state)
        return {"train": train_losses, "val": []}

    def _recency_weights(self, df):
        w = np.zeros(len(df), dtype=np.float32)
        idx = np.flatnonzero(df["delta_t"].to_numpy() > 0)
        x = np.arange(len(idx), dtype=np.float32) / max(len(idx), 1)
        w[idx] = self.recency_floor + (1 - self.recency_floor) * x**self.recency_power
        return w

    def _scaled_features(self, df, fit_scaler):
        if not self.scale_features:
            return df[self.features].to_numpy(dtype=np.float32)
        vals = df[self.features].to_numpy(dtype=np.float64)
        if fit_scaler:
            self.scaler.fit(vals)
        return self.scaler.transform(vals).astype(np.float32, copy=False)

    def _ranges(self, keys):
        change = np.ones(len(keys), dtype=bool)
        change[1:] = (keys[1:] != keys[:-1]).any(axis=1)
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(keys))

        keep = (ends - starts) >= 2
        starts, ends = starts[keep], ends[keep]

        n_chunks = -(-(ends - starts - 1) // self.max_seq_len)
        rep_starts = np.repeat(starts, n_chunks)
        rep_ends = np.repeat(ends, n_chunks)
        k = np.arange(n_chunks.sum()) - np.repeat(
            np.cumsum(n_chunks) - n_chunks, n_chunks
        )
        chunk_starts = rep_starts + k * self.max_seq_len
        return chunk_starts, np.minimum(chunk_starts + self.max_seq_len + 1, rep_ends)

    def _frame_arrays(self, df, fit_scaler=False, sort_by_time=False, recency=False):
        group_cols = ["user_id", "card_id"] if "user_id" in df.columns else ["card_id"]

        sort_keys = [df[c].to_numpy() for c in reversed(group_cols)]
        if sort_by_time:
            sort_keys.insert(0, df["timestamp"].to_numpy())
        order = np.lexsort(sort_keys)

        weight = None
        if recency:
            weight = self._recency_weights(df)[order]

        feats = self._scaled_features(df, fit_scaler)[order]
        col = {c: i for i, c in enumerate(self.features)}
        seq_arr = np.ascontiguousarray(
            feats[:, [col[c] for c in self.sequence_features]]
        )
        tab_arr = (
            np.ascontiguousarray(feats[:, [col[c] for c in self.tabular_features]])
            if self.tabular_features
            else None
        )
        del feats

        query = df[self.query_column].to_numpy(dtype=np.float32)[order]
        target = df["target"].to_numpy(dtype=np.float32)[order]
        mask = None if self.loss_on_same_day_reviews else df["is_inter_day"].to_numpy()
        if "in_split" in df.columns:
            in_split = df["in_split"].to_numpy()
            mask = in_split if mask is None else (mask & in_split)
        if mask is not None:
            mask = mask[order]

        starts, ends = self._ranges(
            np.column_stack([df[c].to_numpy()[order] for c in group_cols])
        )
        return (
            seq_arr,
            tab_arr,
            query,
            target,
            weight,
            mask,
            starts,
            ends,
            df.index.to_numpy()[order],
        )

    def _build_seqs(self, df, fit_scaler=False, sort_by_time=False, recency=False):
        seq_arr, tab_arr, query, target, weight, mask, starts, ends, row_labels = (
            self._frame_arrays(df, fit_scaler, sort_by_time, recency)
        )
        return SeqStore(
            seq_arr,
            tab_arr,
            query,
            target,
            weight,
            mask,
            starts,
            ends,
            self.device,
            row_labels=row_labels,
        )

    def _build_seqs_streaming(self, user_frames, fit_scaler=False, recency=False):
        assert (
            not self.scale_features
        ), "streaming build has no global scaler pass; use scale_features=False"
        seq_parts, tab_parts, query_parts, target_parts = [], [], [], []
        weight_parts, mask_parts, starts_parts, ends_parts, label_parts = (
            [],
            [],
            [],
            [],
            [],
        )
        offset = 0
        for df in user_frames:
            if len(df) == 0:
                continue
            (
                seq_a,
                tab_a,
                query_a,
                target_a,
                weight_a,
                mask_a,
                starts_a,
                ends_a,
                labels_a,
            ) = self._frame_arrays(df, fit_scaler=False, recency=recency)
            seq_parts.append(seq_a)
            if tab_a is not None:
                tab_parts.append(tab_a)
            query_parts.append(query_a)
            target_parts.append(target_a)
            if weight_a is not None:
                weight_parts.append(weight_a)
            if mask_a is not None:
                mask_parts.append(mask_a)
            starts_parts.append(starts_a + offset)
            ends_parts.append(ends_a + offset)
            label_parts.append(labels_a)
            offset += len(df)

        if not seq_parts:
            return SeqStore(
                np.empty((0, len(self.sequence_features)), dtype=np.float32),
                None,
                np.empty(0, np.float32),
                np.empty(0, np.float32),
                None,
                None,
                np.empty(0, np.int64),
                np.empty(0, np.int64),
                self.device,
                row_labels=np.empty(0, np.int64),
            )

        return SeqStore(
            np.concatenate(seq_parts),
            np.concatenate(tab_parts) if tab_parts else None,
            np.concatenate(query_parts),
            np.concatenate(target_parts),
            np.concatenate(weight_parts) if weight_parts else None,
            np.concatenate(mask_parts) if mask_parts else None,
            np.concatenate(starts_parts),
            np.concatenate(ends_parts),
            self.device,
            row_labels=np.concatenate(label_parts),
        )

    def _build_store(self, data, fit_scaler=False, sort_by_time=False, recency=False):
        if isinstance(data, pd.DataFrame):
            return self._build_seqs(
                data, fit_scaler=fit_scaler, sort_by_time=sort_by_time, recency=recency
            )
        return self._build_seqs_streaming(data, fit_scaler=fit_scaler, recency=recency)

    def _capped(self, order, lengths):
        # input is length-sorted, so a batch's max length is at its tail
        batches, i, n = [], 0, len(order)
        while i < n:
            j = i + 1
            while (
                j < n
                and j - i < self.batch_size
                and (j - i + 1) * lengths[j] <= self.max_batch_tokens
            ):
                j += 1
            batches.append(order[i:j])
            i = j
        return batches

    def _length_capped_batches(self, store, shuffle):
        if not shuffle:
            order = np.argsort(store.lengths, kind="stable")
            return self._capped(order, store.lengths[order])
        # length-sort within random pools so batch composition varies per epoch
        perm = np.random.permutation(len(store))
        pool = self.batch_size * 50
        batches = []
        for p in [perm[i : i + pool] for i in range(0, len(perm), pool)]:
            p = p[np.argsort(store.lengths[p], kind="stable")]
            batches += self._capped(p, store.lengths[p])
        random.shuffle(batches)
        return batches

    def _prefetch(self, gen, depth=2):
        q = queue.Queue(maxsize=depth)
        end = object()

        def worker():
            try:
                for item in gen:
                    q.put(item)
                q.put(end)
            except BaseException as e:
                q.put(e)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _iter_batches(self, store, shuffle=True):
        def gen():
            for idx in self._length_capped_batches(store, shuffle):
                yield store.build(idx)[:5]

        return self._prefetch(gen())

    def _batch_loss(self, loss_fn, feats, deltas, targs, masks, weights=None):
        preds, _ = self.predictor(feats, deltas)
        if weights is None:
            return loss_fn(preds[masks], targs[masks])
        w = weights[masks]
        losses = nn.functional.binary_cross_entropy(
            preds[masks], targs[masks], reduction="none"
        )
        return (losses * w).sum() / w.sum()

    def _ema_update(self, ema_state):
        if ema_state is None:
            return
        with torch.no_grad():
            for k, v in self.predictor.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].lerp_(v, 1 - self.ema_decay)
                else:
                    ema_state[k].copy_(v)

    def _train(
        self,
        df,
        lr,
        epochs,
        fit_scaler=True,
        verbose=False,
        val_frac=0.0,
        val_df=None,
        use_ema=False,
        recency=False,
    ):
        seqs = self._build_store(df, fit_scaler=fit_scaler, recency=recency)

        val_seqs = None
        if val_df is not None:
            vs = self._build_store(val_df)
            if len(vs):
                val_seqs = vs
        elif val_frac > 0 and len(seqs) > 1:
            n_val = max(1, int(len(seqs) * val_frac))
            val_idxs = set(random.sample(range(len(seqs)), n_val))
            keep = np.fromiter(
                (i not in val_idxs for i in range(len(seqs))),
                dtype=bool,
                count=len(seqs),
            )
            val_seqs = seqs.subset(np.flatnonzero(~keep))
            seqs = seqs.subset(np.flatnonzero(keep))

        optimizer = self._make_optimizer(lr)
        n_batches = len(self._length_capped_batches(seqs, shuffle=True))
        avg_n = max(float(seqs.lengths.sum()) / max(n_batches, 1), 1.0)
        scheduler = self._make_scheduler(optimizer, epochs, n_batches)
        ema_state = None
        if use_ema and self.ema_decay is not None:
            ema_state = {
                k: v.detach().clone() for k, v in self.predictor.state_dict().items()
            }
        loss_fn = nn.BCELoss()
        best_loss, counter, min_delta = float("inf"), 0, 1e-4
        best_state = None
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.predictor.train()
            epoch_loss, n_targets, skipped = 0.0, 0, 0
            batches = self._iter_batches(seqs)
            if verbose:
                batches = tqdm(
                    batches, total=n_batches, desc=f"Epoch {epoch + 1}", leave=False
                )
            for feats, deltas, targs, masks, weights in batches:
                if not masks.any():
                    continue
                optimizer.zero_grad()
                loss = self._batch_loss(loss_fn, feats, deltas, targs, masks, weights)
                n = int(masks.sum())
                self._backward_loss(loss, n, avg_n).backward()
                if self.grad_clip is not None:
                    gnorm = nn.utils.clip_grad_norm_(
                        self.predictor.parameters(), self.grad_clip
                    )
                    if not torch.isfinite(gnorm):
                        skipped += 1
                        continue
                optimizer.step()
                self._after_optimizer_step()
                self._ema_update(ema_state)
                self._scheduler_batch_step(scheduler)
                epoch_loss += loss.item() * n
                n_targets += n
            train_loss = epoch_loss / max(n_targets, 1)

            train_losses.append(train_loss)
            if val_seqs is not None:
                if ema_state is not None:
                    backup = {
                        k: v.detach().clone()
                        for k, v in self.predictor.state_dict().items()
                    }
                    self.predictor.load_state_dict(ema_state)
                self.predictor.eval()
                with torch.no_grad():
                    vloss, vn = 0.0, 0
                    for feats, deltas, targs, masks, _ in self._iter_batches(
                        val_seqs, shuffle=False
                    ):
                        if not masks.any():
                            continue
                        n = int(masks.sum())
                        vloss += (
                            self._batch_loss(
                                loss_fn, feats, deltas, targs, masks
                            ).item()
                            * n
                        )
                        vn += n
                    val_losses.append(vloss / max(vn, 1))
                if ema_state is not None:
                    self.predictor.load_state_dict(backup)
            if verbose:
                self._plot_pretraining({"train": train_losses, "val": val_losses})
            stop_metric = val_losses[-1] if val_losses else train_loss
            self._scheduler_epoch_step(scheduler, stop_metric)
            if stop_metric < best_loss - min_delta:
                best_loss, counter = stop_metric, 0
                best_state = copy.deepcopy(
                    ema_state if ema_state is not None else self.predictor.state_dict()
                )
            else:
                counter += 1
                if self.patience is not None and counter >= self.patience:
                    break
            if verbose:
                val_str = f" — Val loss: {val_losses[-1]:.4f}" if val_losses else ""
                skip_str = f" — skipped {skipped} non-finite batches" if skipped else ""
                print(
                    f"Epoch {epoch + 1} — Train loss: {train_loss:.4f}{val_str}{skip_str}"
                )

        if best_state is not None:
            self.predictor.load_state_dict(best_state)

        return {"train": train_losses, "val": val_losses}

    def predict(self, df):
        df_out = df.copy()
        df_out["prediction"] = 0.0

        store = self._build_seqs(df, sort_by_time=True)
        if not len(store):
            return df_out

        self.predictor.eval()
        all_idxs, all_preds = [], []
        for idx in self._length_capped_batches(store, shuffle=False):
            feats, deltas, _, _, _, lens = store.build(idx)
            with torch.no_grad():
                preds, _ = self.predictor(feats, deltas)
            for row, i in enumerate(idx):
                s = int(store.starts[i])
                all_idxs.append(store.row_labels[s + 1 : s + 1 + lens[row]])
                all_preds.append(preds[row][: lens[row]].cpu())
        df_out.loc[np.concatenate(all_idxs), "prediction"] = torch.cat(
            all_preds
        ).numpy()

        return df_out
