import json
import os
import random
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from datasets import Audio, concatenate_datasets, load_dataset

OUTDIR = "./metadatas"
os.makedirs(OUTDIR, exist_ok=True)

LS_META = {
    "dataset_name": "openslr/librispeech_asr",
    "config_name": None,
    "split": "train.clean.100",
}

CV_META = {
    "dataset_name": "mozilla-foundation/common_voice_13_0",
    "split": "train",
}


def _load_split(dataset_name, config_name, split):
    if config_name is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, config_name, split=split)


def _audio_path_or_bytes(audio_field):
    if not isinstance(audio_field, dict):
        return None, None
    return audio_field.get("path"), audio_field.get("bytes")


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


def _get_audio_duration(audio_field, sample=None):
    path, audio_bytes = _audio_path_or_bytes(audio_field)
    resolved_path = _resolve_audio_path(path, sample)

    if resolved_path:
        try:
            import torchaudio
            info = torchaudio.info(resolved_path)
            return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            import soundfile as sf
            info = sf.info(resolved_path)
            return float(info.frames) / float(info.samplerate)

    if audio_bytes is not None:
        import soundfile as sf
        info = sf.info(BytesIO(audio_bytes))
        return float(info.frames) / float(info.samplerate)

    raise RuntimeError(f"audio path/bytes unavailable: path={path!r}")


def add_duration(example):
    example["duration"] = _get_audio_duration(example["audio"], sample=example)
    return example


def _normalize_indices(indices):
    if hasattr(indices, "to_pylist"):
        indices = indices.to_pylist()
    if isinstance(indices, tuple):
        return list(indices)
    if isinstance(indices, list):
        return [_normalize_indices(item) for item in indices]
    return indices


def save_metadata(from_libri, caliset_name, config_name, hf_indices, out_path):
    payload = {
        "from_libri": from_libri,
        "dataset_name": LS_META["dataset_name"] if from_libri else CV_META["dataset_name"],
        "config_name": config_name,
        "split": LS_META["split"] if from_libri else CV_META["split"],
        "caliset_name": caliset_name,
        "hf_indices": _normalize_indices(hf_indices),
    }

    out_file = os.path.join(out_path, f"{caliset_name}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_librispeech_calisets(
    nsamples_b=128,
    nsamples_l=128,
    seed=0,
    basic_min_duration=2.0,
    basic_max_duration=8.0,
    long_min_duration=10.0,
):
    ds = _load_split(LS_META["dataset_name"], LS_META["config_name"], LS_META["split"])
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(add_duration, desc="Adding duration to LibriSpeech")

    durations = ds["duration"]
    basic_indices = [idx for idx, duration in enumerate(durations) if basic_min_duration <= duration <= basic_max_duration]
    long_indices = [idx for idx, duration in enumerate(durations) if duration >= long_min_duration]

    if len(basic_indices) < nsamples_b:
        raise ValueError(f"Not enough samples for Basic: requested {nsamples_b}, found {len(basic_indices)}")
    if len(long_indices) < nsamples_l:
        raise ValueError(f"Not enough samples for Long: requested {nsamples_l}, found {len(long_indices)}")

    rng_basic = random.Random(seed)
    selected_basic_indices = rng_basic.sample(basic_indices, nsamples_b)
    basic_index_set = set(selected_basic_indices)

    long_indices = [idx for idx in long_indices if idx not in basic_index_set]
    if len(long_indices) < nsamples_l:
        raise ValueError(
            "Not enough disjoint samples for Long after overlap removal: "
            f"requested {nsamples_l}, found {len(long_indices)}"
        )

    rng_long = random.Random(seed + 1)
    selected_long_indices = rng_long.sample(long_indices, nsamples_l)

    basic_set = ds.select(selected_basic_indices)
    long_set = ds.select(selected_long_indices)

    save_metadata(True, "librispeech_basic", LS_META["config_name"], selected_basic_indices, OUTDIR)
    save_metadata(True, "librispeech_long", LS_META["config_name"], selected_long_indices, OUTDIR)

    return basic_set, long_set


def build_commonvoice_caliset(
    lang,
    nsamples=128,
    seed=0,
    min_duration=2.0,
    max_duration=15.0,
    save=True,
    return_indices=False,
):
    ds = _load_split(CV_META["dataset_name"], lang, CV_META["split"])
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(add_duration, desc=f"Adding duration to Common Voice ({lang})")

    durations = ds["duration"]
    valid_indices = [idx for idx, duration in enumerate(durations) if min_duration <= duration <= max_duration]

    if len(valid_indices) < nsamples:
        raise ValueError(f"Not enough samples for CV-{lang}: requested {nsamples}, found {len(valid_indices)}")

    rng = random.Random(seed)
    selected_indices = rng.sample(valid_indices, nsamples)
    cv_set = ds.select(selected_indices)
    cv_set = cv_set.map(lambda ex: {"cv_lang": lang}, desc=f"Annotating cv_lang={lang}")

    if save:
        save_metadata(False, f"cv_single_{lang}", lang, selected_indices, OUTDIR)

    if return_indices:
        return cv_set, selected_indices
    return cv_set


def build_commonvoice_multi_caliset(
    langs,
    nsamples_per_lang=43,
    seed=0,
    min_duration=2.0,
    max_duration=15.0,
):
    parts = []
    configs = []
    hf_indices = []

    for offset, lang in enumerate(langs):
        part, selected_indices = build_commonvoice_caliset(
            lang=lang,
            nsamples=nsamples_per_lang,
            seed=seed + offset,
            min_duration=min_duration,
            max_duration=max_duration,
            save=False,
            return_indices=True,
        )
        parts.append(part)
        configs.append(lang)
        hf_indices.append(selected_indices)

    cv_multi_set = concatenate_datasets(parts)
    save_metadata(False, "cv_multi", configs, hf_indices, OUTDIR)
    return cv_multi_set


def main():
    build_librispeech_calisets()
    build_commonvoice_caliset(lang="en", nsamples=128, seed=0)
    build_commonvoice_multi_caliset(langs=["en", "de", "fr"], nsamples_per_lang=43, seed=0)


if __name__ == "__main__":
    main()
