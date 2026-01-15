"""
Dataset and data loading utilities for brain-to-text decoding.
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BrainToTextDataset(Dataset):
    """
    Dataset for loading brain signal trials and their phoneme labels.
    Returns batches of data.
    """

    def __init__(
        self,
        trial_indices: Dict[int, Dict[str, Any]],
        n_batches: int,
        split: str = "train",
        batch_size: int = 64,
        days_per_batch: int = 1,
        random_seed: int = -1,
        must_include_days: Optional[List[int]] = None,
        feature_subset: Optional[List[int]] = None,
    ):
        """
        Args:
            trial_indices (dict): Day indices mapping to trial lists and file paths.
            n_batches (int): Number of random batches to generate per epoch.
            split (str): 'train' or 'test'.
            batch_size (int): Trials per batch.
            days_per_batch (int): Unique days allowed in a single batch.
            random_seed (int): Seed for reproducibility.
            must_include_days (list): Days that must appear in every batch.
            feature_subset (list): Indices of neural features to keep.
        """
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.split = split
        self.days_per_batch = days_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.trial_indices = trial_indices
        self.n_days = len(trial_indices)
        self.feature_subset = feature_subset

        self.n_trials = sum(len(d["trials"]) for d in trial_indices.values())

        if must_include_days:
            # Handle negative indexing (e.g., -1 for last day)
            self.must_include_days = [
                d if d >= 0 else self.n_days + d for d in must_include_days
            ]
            if len(self.must_include_days) > self.days_per_batch:
                raise ValueError("must_include_days > days_per_batch")
        else:
            self.must_include_days = None

        if self.split == "train":
            if self.days_per_batch > self.n_days:
                raise ValueError("days_per_batch > available days")
            self.batch_index = self._create_train_batch_index()
        else:
            self.batch_index = self._create_test_batch_index()
            # Validation data has a fixed length
            self.n_batches = len(self.batch_index)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[str]]:
        """
        Retrieves a full batch of data.
        """
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indices": [],
            "transcriptions": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]

        for day in index:
            with h5py.File(self.trial_indices[day]["session_path"], "r") as f:
                for trial_idx in index[day]:
                    try:
                        group = f[f"trial_{trial_idx:04d}"]

                        features = torch.from_numpy(group["input_features"][:])
                        if self.feature_subset:
                            features = features[:, self.feature_subset]

                        batch["input_features"].append(features)
                        batch["seq_class_ids"].append(
                            torch.from_numpy(group["seq_class_ids"][:])
                        )
                        batch["transcriptions"].append(
                            torch.from_numpy(group["transcription"][:])
                        )
                        batch["n_time_steps"].append(group.attrs["n_time_steps"])
                        batch["phone_seq_lens"].append(group.attrs["seq_len"])
                        batch["day_indices"].append(int(day))
                        batch["block_nums"].append(group.attrs["block_num"])
                        batch["trial_nums"].append(group.attrs["trial_num"])

                    except Exception as e:
                        print(f"Error loading trial {trial_idx} day {day}: {e}")
                        continue

        # Standardize batch dimensions
        batch["input_features"] = pad_sequence(
            batch["input_features"], batch_first=True, padding_value=0
        )
        batch["seq_class_ids"] = pad_sequence(
            batch["seq_class_ids"], batch_first=True, padding_value=0
        )

        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indices"] = torch.tensor(batch["day_indices"])
        batch["transcriptions"] = torch.stack(batch["transcriptions"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    def _create_train_batch_index(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Pre-computes random batch indices for training.
        """
        batch_index = {}
        day_keys = [d for d, info in self.trial_indices.items() if len(info["trials"]) > 0]
        if not day_keys:
            raise ValueError("No days with trials found in trial_indices")

        if self.must_include_days:
            non_must_days = [d for d in day_keys if d not in self.must_include_days]

        for b_idx in range(self.n_batches):
            if self.must_include_days:
                n_random = min(len(non_must_days), self.days_per_batch - len(self.must_include_days))
                random_days = np.random.choice(non_must_days, size=n_random, replace=False)
                days = np.concatenate((self.must_include_days, random_days))
            else:
                n_select = min(len(day_keys), self.days_per_batch)
                days = np.random.choice(day_keys, size=n_select, replace=False)

            trials_per_day = math.ceil(self.batch_size / self.days_per_batch)
            batch = {}
            total_trials = 0

            for d in days:
                indices = np.random.choice(
                    self.trial_indices[d]["trials"], size=trials_per_day, replace=True
                )
                batch[d] = indices
                total_trials += len(indices)

            # Prune extra trials to match exact batch_size
            while total_trials > self.batch_size:
                d = np.random.choice(days)
                if len(batch[d]) > 0:
                    batch[d] = batch[d][:-1]
                    total_trials -= 1

            batch_index[b_idx] = batch

        return batch_index

    def _create_test_batch_index(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Creates sequential batch indices for validation/test data.
        """
        batch_index = {}
        batch_idx = 0

        for day in self.trial_indices:
            trials = self.trial_indices[day]["trials"]
            num_trials = len(trials)
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size

            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, num_trials)
                batch_index[batch_idx] = {day: trials[start:end]}
                batch_idx += 1

        return batch_index


def train_test_split_indices(
    file_paths: List[str],
    test_percentage: float = 0.1,
    seed: int = -1,
    bad_trials_dict: Optional[Dict[str, Dict[str, List[int]]]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Splits recording trials into training and testing sets.
    """
    if seed != -1:
        np.random.seed(seed)

    trials_per_day = {}
    for i, path in enumerate(file_paths):
        # Extract session name (e.g., t15.2023.08.21)
        session_parts = [s for s in path.split("/") if s.startswith(("t15.20", "t12.20"))]
        if not session_parts:
            continue
        session = session_parts[0]

        good_indices = []
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                for t_idx in range(len(f)):
                    key = f"trial_{t_idx:04d}"
                    block = f[key].attrs["block_num"]
                    trial = f[key].attrs["trial_num"]

                    if (
                        bad_trials_dict
                        and session in bad_trials_dict
                        and str(block) in bad_trials_dict[session]
                        and trial in bad_trials_dict[session][str(block)]
                    ):
                        continue
                    good_indices.append(t_idx)

        trials_per_day[i] = {
            "num_trials": len(good_indices),
            "trial_indices": good_indices,
            "session_path": path,
        }

    train_trials, test_trials = {}, {}

    for day, info in trials_per_day.items():
        all_idxs = info["trial_indices"]
        num_trials = info["num_trials"]

        if test_percentage <= 0:
            train_trials[day] = {"trials": all_idxs, "session_path": info["session_path"]}
            test_trials[day] = {"trials": [], "session_path": info["session_path"]}
        elif test_percentage >= 1:
            train_trials[day] = {"trials": [], "session_path": info["session_path"]}
            test_trials[day] = {"trials": all_idxs, "session_path": info["session_path"]}
        else:
            num_test = max(1, int(num_trials * test_percentage))
            test_idxs = np.random.choice(all_idxs, size=num_test, replace=False).tolist()
            train_idxs = [idx for idx in all_idxs if idx not in test_idxs]

            train_trials[day] = {"trials": train_idxs, "session_path": info["session_path"]}
            test_trials[day] = {"trials": test_idxs, "session_path": info["session_path"]}

    return train_trials, test_trials
