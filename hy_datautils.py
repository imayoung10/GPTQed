# cali_dataloader.py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from caliset_builder import build


class CalibrationDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]["audio"]  # np.ndarray (float32)


def collate_fn(batch):
    """
    batch: list of np.ndarray (가변 길이 오디오)
    model(batch) 호출 시 batch가 audio list로 전달됨
    """
    return batch  # list[np.ndarray] 그대로


def build_cali_dataloader(
    n_per_split: int = 64,
    output_dir: str = "./calibration_set",
    seed: int = 42,
    batch_size: int = 1,  # sequential에서 샘플 단위로 Catcher에 넣으므로 1 권장
) -> DataLoader:
    records = build(n_per_split=n_per_split, output_dir=output_dir, seed=seed)
    dataset = CalibrationDataset(records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )