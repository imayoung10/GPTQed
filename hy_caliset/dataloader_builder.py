import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from datasets import Audio, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset


def _load_split(dataset_name, config_name, split):
    if config_name is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, config_name, split=split)


@lru_cache(maxsize=4096)
def _search_in_hf_cache(filename):
    home = Path.home()
    roots = [
        home / ".cache" / "huggingface" / "datasets",
        home / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return str(path)
        except Exception:
            continue
    return None


def _audio_path_or_bytes(audio_field):
    if not isinstance(audio_field, dict):
        return None, None
    return audio_field.get("path"), audio_field.get("bytes")


def _resolve_audio_path(path, sample=None):
    candidates = []
    if path:
        candidates.append(path)
    if sample is not None:
        file_field = sample.get("file")
        if isinstance(file_field, str) and file_field:
            candidates.append(file_field)

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)
        if not candidate_path.is_absolute():
            abs_path = (Path.cwd() / candidate_path).resolve()
            if abs_path.is_file():
                return str(abs_path)
            cached = _search_in_hf_cache(candidate_path.name)
            if cached is not None:
                return cached
    return None


def load_audio_array(audio_field, sample=None):
    path, audio_bytes = _audio_path_or_bytes(audio_field)
    resolved_path = _resolve_audio_path(path, sample)

    if resolved_path:
        try:
            import torchaudio
            wav, sr = torchaudio.load(resolved_path)
            if wav.ndim == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).numpy(), int(sr)
        except Exception:
            import soundfile as sf
            wav, sr = sf.read(resolved_path, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            return wav, int(sr)

    if audio_bytes is not None:
        import soundfile as sf
        wav, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        return wav, int(sr)

    raise RuntimeError(f"audio path/bytes unavailable: path={path!r}")


class HYCalibrationSet(Dataset):
    def __init__(self, metapath):
        with open(metapath, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.from_libri = metadata["from_libri"]
        self.dataset_name = metadata["dataset_name"]
        self.config_name = metadata["config_name"]
        self.split = metadata["split"]
        self.caliset_name = metadata["caliset_name"]
        self.indices = metadata["hf_indices"]

        if isinstance(self.config_name, list):
            datasets = []
            for cfg, indices in zip(self.config_name, self.indices):
                part = _load_split(self.dataset_name, cfg, self.split)
                part = part.cast_column("audio", Audio(decode=False))
                part = part.select(indices)
                part = part.map(lambda ex, cfg=cfg: {"cv_lang": cfg}, desc=f"Re-annotating cv_lang={cfg}")
                datasets.append(part)
            self.caliset = concatenate_datasets(datasets)
        else:
            dataset = _load_split(self.dataset_name, self.config_name, self.split)
            dataset = dataset.cast_column("audio", Audio(decode=False))
            self.caliset = dataset.select(self.indices)

            if (not self.from_libri) and self.caliset_name.startswith("cv_single_"):
                lang = self.config_name
                self.caliset = self.caliset.map(
                    lambda ex: {"cv_lang": lang},
                    desc=f"Re-annotating cv_lang={lang}",
                )

    def __len__(self):
        return len(self.caliset)

    def __getitem__(self, idx):
        item = self.caliset[idx]
        audio, sr = load_audio_array(item["audio"], sample=item)
        return {
            "audio": audio,
            "sampling_rate": sr,
            "context": item["text"],
            "language": item.get("cv_lang", "en"),
        }


def simple_collate(batch):
    return batch[0]


def make_dataloader(mpath):
    dataset = HYCalibrationSet(mpath)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=simple_collate,
    )


def main():
    ls_b_loader = make_dataloader("./metadatas/librispeech_basic.json")
    ls_l_loader = make_dataloader("./metadatas/librispeech_long.json")
    cv_s_loader = make_dataloader("./metadatas/cv_single_en.json")
    cv_m_loader = make_dataloader("./metadatas/cv_multi.json")

    print(next(iter(ls_b_loader)).keys())
    print(next(iter(ls_l_loader)).keys())
    print(next(iter(cv_s_loader)).keys())
    print(next(iter(cv_m_loader)).keys())


if __name__ == "__main__":
    main()
