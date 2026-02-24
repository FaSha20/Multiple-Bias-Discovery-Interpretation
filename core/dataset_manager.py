# core/dataset_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from core.dataloader import SuperCIFAR100, ImbalancedSuperCIFAR100

@dataclass(frozen=True)
class SampleRecord:
    index: int
    true_label: int
    pred_label: int
    correct: int  # 0/1


class DatasetManager:
    """
    Loads CIFAR-100 test set + trained model, reads bias_group.csv,
    and provides utilities to retrieve indices by DIM dimension and compute accuracy.

    Assumption:
    - bias_group.csv rows align with CIFAR-100 test sample ordering (index = row number).
      If you have an explicit index column, set index_col in _load_bias_df().
    """

    def __init__(
        self,
        cifar_root: str | Path,
        model_path: str | Path,
        bias_csv_path: str | Path,
        dataset_name: str = "cifar100",
        batch_size: int = 256,
        num_workers: int = 2,
        device: Optional[str] = None,
        logger=None,
    ):
        self.cifar_root = Path(cifar_root)
        self.model_path = Path(model_path)
        self.bias_csv_path = Path(bias_csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.dataset_name = dataset_name

        self.test_dataset = self._load_cifar_test()
        self.bias_df = self._load_bias_df()

        # if len(self.bias_df) != len(self.test_dataset):
        #     raise ValueError(
        #         f"bias_group.csv length ({len(self.bias_df)}) != CIFAR100 test length ({len(self.test_dataset)}). "
        #         "If your CSV has an explicit index column, update _load_bias_df() accordingly."
        #     )

        self.model = self._load_model()
        self.model.eval()

        # Cached inference results
        self._preds: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._correct: Optional[np.ndarray] = None

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _load_cifar_test(self):
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        return datasets.CIFAR100(root=str(self.cifar_root), train=False, download=True, transform=tfm)
        # if self.dataset_name == "cifar100":
        #     ds = datasets.CIFAR100(
        #         root=str(self.cifar_root),
        #         train=False,
        #         download=True,
        #         transform=tfm,
        #     )

        # elif self.dataset_name == "supercifar100":
        #     ds = SuperCIFAR100(
        #         root=str(self.cifar_root),
        #         train=False,
        #         download=True,
        #         transform=tfm,
        #     )

        # elif self.dataset_name == "imbalancedsupercifar100":
        #     ds = ImbalancedSuperCIFAR100(
        #         split="test",
        #         root=str(self.cifar_root),
        #         download=True,
        #         transform=tfm,
        #     )
        # elif self.dataset_name == "imbalancedsupercifar100_intervention":
        #     ds = ImbalancedSuperCIFAR100(
        #         split='intervention', 
        #         root=str(self.cifar_root), 
        #         transform=tfm, 
        #         download=True
        #     )
        
        # else:
        #     raise ValueError(f"Unknown dataset_name: {self.dataset_name}")

        # return ds
    
    def _load_bias_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.bias_csv_path)
        required = {"true_group", "random_group", "DIM"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"bias_group.csv missing required columns: {missing}")

        # If you have an index column, use it like:
        # df = pd.read_csv(self.bias_csv_path).set_index("index").sort_index()
        # and then adjust access accordingly.
        return df.reset_index(drop=True)

    def _load_model(self) -> nn.Module:
        """
        Loads a ResNet-18 for CIFAR-100.
        You may need to adjust depending on how you saved the model.
        """
        model = models.resnet18(num_classes=20)
        ckpt = torch.load(self.model_path, map_location="cpu")

        # Common patterns:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
            state = ckpt
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported checkpoint format. Expected a state_dict-like dict.")

        # Strip possible "module." prefix
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)

        model.to(self.device)
        self._log(f"Loaded model from {self.model_path} on {self.device}")
        return model

    # -----------------------------
    # Public accessors
    # -----------------------------

    def get_dim_ids(self) -> List[int]:
        return sorted(self.bias_df["DIM"].unique().tolist())

    def get_indices_for_dim(self, dim_id: int) -> np.ndarray:
        mask = self.bias_df["DIM"].values == dim_id
        return np.where(mask)[0].astype(int)

    def get_true_groups_for_dim(self, dim_id: int) -> List[int]:
        idx = self.get_indices_for_dim(dim_id)
        groups = self.bias_df.loc[idx, "true_group"].unique().tolist()
        return sorted([int(x) for x in groups])

    def get_records_for_indices(self, indices: Sequence[int]) -> List[SampleRecord]:
        self.ensure_inference_cache()
        assert self._preds is not None and self._labels is not None and self._correct is not None

        recs: List[SampleRecord] = []
        for i in indices:
            recs.append(
                SampleRecord(
                    index=int(i),
                    true_label=int(self._labels[i]),
                    pred_label=int(self._preds[i]),
                    correct=int(self._correct[i]),
                )
            )
        return recs

    def get_images_for_indices(self, indices: Sequence[int]) -> List[torch.Tensor]:
        """
        Returns transformed tensors (C,H,W). Useful for grids.
        """
        imgs = []
        for i in indices:
            x, _y = self.test_dataset[int(i)]
            imgs.append(x)
        return imgs

    def ensure_inference_cache(self) -> None:
        """
        Runs model inference over full CIFAR-100 test set once and caches outputs.
        """
        if self._preds is not None:
            return

        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        preds = np.zeros(len(self.test_dataset), dtype=np.int64)
        labels = np.zeros(len(self.test_dataset), dtype=np.int64)

        self.model.eval()
        ptr = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                p = torch.argmax(logits, dim=1)

                bs = x.size(0)
                preds[ptr : ptr + bs] = p.detach().cpu().numpy()
                labels[ptr : ptr + bs] = y.detach().cpu().numpy()
                ptr += bs

        correct = (preds == labels).astype(np.int32)

        self._preds = preds
        self._labels = labels
        self._correct = correct

        acc = float(correct.mean())
        self._log(f"Cached inference for CIFAR100 test. Overall accuracy = {acc:.4f}")

    def correctness_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        self.ensure_inference_cache()
        assert self._correct is not None
        return self._correct[np.array(indices, dtype=int)]

    def accuracy_for_indices(self, indices: Sequence[int]) -> float:
        c = self.correctness_for_indices(indices)
        if c.size == 0:
            return float("nan")
        return float(c.mean())