import numpy as np
import torch
from torch.utils.data import Dataset

from models.geometry import compute_geometry_features


class NpyDataset(Dataset):
    def __init__(
        self,
        file_list,
        target_shape,
        voxel_size,
        surface_level,
        boundary_step_size,
    ):
        self.files = file_list
        self.target_shape = target_shape
        self.voxel_size = voxel_size
        self.surface_level = surface_level
        self.boundary_step_size = boundary_step_size

        if len(self.files) == 0:
            raise ValueError("file_list is empty; please check the data path.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = np.load(path)
        except Exception:
            data = np.zeros(self.target_shape, dtype=np.uint8)

        data = data.astype(np.uint8)

        if np.random.rand() > 0.5:
            data = np.flip(data, axis=0)
        if np.random.rand() > 0.5:
            data = np.flip(data, axis=1)
        if np.random.rand() > 0.5:
            data = np.flip(data, axis=2)

        data = self._crop_or_pad(data)

        if np.random.rand() > 0.5:
            axes = [0, 1, 2]
            np.random.shuffle(axes)
            data = np.transpose(data, axes)

        if np.random.rand() > 0.5:
            ax1, ax2 = np.random.choice([0, 1, 2], 2, replace=False)
            k = np.random.randint(1, 4)
            data = np.rot90(data, k=k, axes=(ax1, ax2))

        geom = compute_geometry_features(
            np.ascontiguousarray(data),
            self.voxel_size,
            self.surface_level,
            self.boundary_step_size,
        )
        data = data.copy()

        tensor = torch.from_numpy(data).to(torch.long).unsqueeze(0)
        geom_tensor = torch.from_numpy(geom).to(torch.float32)
        return tensor, geom_tensor

    def _crop_or_pad(self, data):
        d, h, w = data.shape
        td, th, tw = self.target_shape

        start_d = (d - td) // 2 if d > td else 0
        start_h = (h - th) // 2 if h > th else 0
        start_w = (w - tw) // 2 if w > tw else 0
        data = data[start_d : start_d + td, start_h : start_h + th, start_w : start_w + tw]

        cd, ch, cw = data.shape
        pad_d, pad_h, pad_w = td - cd, th - ch, tw - cw
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pd_bef, ph_bef, pw_bef = pad_d // 2, pad_h // 2, pad_w // 2
            data = np.pad(
                data,
                (
                    (pd_bef, pad_d - pd_bef),
                    (ph_bef, pad_h - ph_bef),
                    (pw_bef, pad_w - pw_bef),
                ),
                mode="constant",
                constant_values=0,
            )

        return data
