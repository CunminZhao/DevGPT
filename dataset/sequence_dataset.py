from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def get_label_from_path(path, delimiter="_", index=-1):
    parts = Path(path).stem.split(delimiter)
    return parts[index]


def list_npy_files(data_dir):
    return sorted(Path(data_dir).glob("*.npy"))


def normalize_array(arr, eps):
    mean = arr.mean()
    std = arr.std() + eps
    return (arr - mean) / std


class UnlabeledCellDataset(Dataset):
    def __init__(self, file_paths, max_seq_len, normalization_eps):
        self.file_paths = list(file_paths)
        self.max_seq_len = max_seq_len
        self.normalization_eps = normalization_eps

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        arr = np.load(self.file_paths[idx]).astype(np.float32)
        length = min(arr.shape[0], self.max_seq_len)
        arr = arr[:length].reshape(length, -1)
        arr = normalize_array(arr, self.normalization_eps)
        return torch.from_numpy(arr), length


class LabeledCellDataset(Dataset):
    def __init__(self, file_paths, labels, label2idx, max_seq_len, normalization_eps):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.label2idx = label2idx
        self.max_seq_len = max_seq_len
        self.normalization_eps = normalization_eps

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        arr = np.load(self.file_paths[idx]).astype(np.float32)
        length = min(arr.shape[0], self.max_seq_len)
        arr = arr[:length].reshape(length, -1)
        arr = normalize_array(arr, self.normalization_eps)
        return torch.from_numpy(arr), self.label2idx[self.labels[idx]], length


class InferenceDataset(Dataset):
    def __init__(self, file_paths, raw_labels, label2idx, max_seq_len, normalization_eps):
        self.file_paths = list(file_paths)
        self.raw_labels = list(raw_labels)
        self.label2idx = label2idx
        self.max_seq_len = max_seq_len
        self.normalization_eps = normalization_eps

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        arr = np.load(self.file_paths[idx]).astype(np.float32)
        length = min(arr.shape[0], self.max_seq_len)
        arr = arr[:length].reshape(length, -1)
        arr = normalize_array(arr, self.normalization_eps)
        label = self.label2idx.get(self.raw_labels[idx], -1)
        return torch.from_numpy(arr), label, length, self.raw_labels[idx], str(self.file_paths[idx])


class TimeToEndDataset(Dataset):
    def __init__(
        self,
        file_paths,
        max_seq_len,
        normalization_eps,
        min_obs_steps,
        min_obs_ratio,
        max_obs_ratio,
        sampling_mode,
        fixed_obs_ratio=None,
    ):
        self.file_paths = list(file_paths)
        self.max_seq_len = max_seq_len
        self.normalization_eps = normalization_eps
        self.min_obs_steps = min_obs_steps
        self.min_obs_ratio = min_obs_ratio
        self.max_obs_ratio = max_obs_ratio
        self.sampling_mode = sampling_mode
        self.fixed_obs_ratio = fixed_obs_ratio

    def __len__(self):
        return len(self.file_paths)

    def _choose_obs_len(self, total_len):
        if total_len <= 2:
            return 1

        ratio_low = int(np.ceil(total_len * self.min_obs_ratio))
        ratio_high = int(np.floor(total_len * self.max_obs_ratio))
        low = max(1, self.min_obs_steps, ratio_low)
        high = min(total_len - 1, ratio_high)

        if high < low:
            low = min(max(1, low), total_len - 1)
            high = low

        if self.sampling_mode == "fixed" and self.fixed_obs_ratio is not None:
            obs_len = int(round(total_len * self.fixed_obs_ratio))
            return int(np.clip(obs_len, 1, total_len - 1))
        return int(np.random.randint(low, high + 1))

    def __getitem__(self, idx):
        arr = np.load(self.file_paths[idx]).astype(np.float32)
        total_len = min(arr.shape[0], self.max_seq_len)
        arr = arr[:total_len].reshape(total_len, -1)
        arr = normalize_array(arr, self.normalization_eps)

        obs_len = self._choose_obs_len(total_len)
        remain_len = float(total_len - obs_len)
        return (
            torch.from_numpy(arr[:obs_len]),
            torch.tensor(remain_len, dtype=torch.float32),
            torch.tensor(obs_len, dtype=torch.long),
            torch.tensor(total_len, dtype=torch.long),
        )


class CellCycleInferenceDataset(Dataset):
    def __init__(self, file_paths, min_input_ratio, max_input_ratio, max_seq_len, normalization_eps):
        self.file_paths = list(file_paths)
        self.min_input_ratio = min_input_ratio
        self.max_input_ratio = max_input_ratio
        self.max_seq_len = max_seq_len
        self.normalization_eps = normalization_eps

    def __len__(self):
        return len(self.file_paths)

    def _choose_obs_len(self, total_len):
        if total_len <= 2:
            return 1

        low = max(1, int(np.ceil(total_len * self.min_input_ratio)))
        high = min(total_len - 1, int(np.floor(total_len * self.max_input_ratio)))
        if high < low:
            low = min(max(1, low), total_len - 1)
            high = low
        return int(np.random.randint(low, high + 1))

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        arr = np.load(file_path).astype(np.float32)
        total_len = min(arr.shape[0], self.max_seq_len)
        arr = arr[:total_len].reshape(total_len, -1)
        arr = normalize_array(arr, self.normalization_eps)

        obs_len = self._choose_obs_len(total_len)
        remain_len = float(total_len - obs_len)
        return (
            torch.from_numpy(arr[:obs_len]),
            torch.tensor(obs_len, dtype=torch.long),
            torch.tensor(total_len, dtype=torch.long),
            torch.tensor(remain_len, dtype=torch.float32),
            Path(file_path).name,
        )


def collate_unlabeled(batch):
    arrays, lengths = zip(*batch)
    padded, mask = pad_arrays(arrays, lengths)
    return padded, mask


def collate_labeled(batch):
    arrays, labels, lengths = zip(*batch)
    padded, mask = pad_arrays(arrays, lengths)
    return padded, torch.tensor(labels), mask, torch.tensor(lengths, dtype=torch.long)


def collate_inference(batch):
    arrays, labels, lengths, raw_labels, file_paths = zip(*batch)
    padded, mask = pad_arrays(arrays, lengths)
    return (
        padded,
        torch.tensor(labels),
        mask,
        torch.tensor(lengths, dtype=torch.long),
        list(raw_labels),
        list(file_paths),
    )


def collate_regression(batch):
    arrays, targets, lengths, total_lengths = zip(*batch)
    padded, mask = pad_arrays(arrays, [int(x) for x in lengths])
    return (
        padded,
        torch.stack(targets),
        mask,
        torch.stack(lengths),
        torch.stack(total_lengths),
    )


def collate_cellcycle_inference(batch):
    arrays, obs_lens, total_lens, remain_gts, names = zip(*batch)
    padded, mask = pad_arrays(arrays, [int(x) for x in obs_lens])
    return (
        padded,
        mask,
        torch.stack(obs_lens),
        torch.stack(total_lens),
        torch.stack(remain_gts),
        list(names),
    )


def pad_arrays(arrays, lengths):
    max_len = max(lengths)
    feat_dim = arrays[0].shape[-1]
    padded = torch.zeros(len(arrays), max_len, feat_dim)
    mask = torch.ones(len(arrays), max_len, dtype=torch.bool)
    for i, (arr, length) in enumerate(zip(arrays, lengths)):
        padded[i, :length] = arr
        mask[i, :length] = False
    return padded, mask
