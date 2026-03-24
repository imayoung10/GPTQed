"""
caliset_builder.py
────────────────────────────
모듈별 Quantization 비교 실험용 calibration dataset 빌더.

핵심: calibration은 반드시 train split에서 뽑아야 함.
      test split은 WER 평가에서만 사용.

구성 (총 128개)
  train.clean.100 : 64개
  train.other.500 : 64개  ← noisy 환경 커버
  
저장 metadata list -> json으로 저장
- id
- split
- speak_id
- transcript
- duration_sec
- sample_rate

"""

import json
import random
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset


# 16kHz로 통일
TARGET_SR   = 16000


# ──────────────────────────────────────────────
# 오디오 유틸
# ──────────────────────────────────────────────

def resample_if_needed(audio_array: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == TARGET_SR:
        return audio_array.astype(np.float32), sr
    try:
        import torchaudio, torch
        waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return waveform.squeeze(0).numpy(), TARGET_SR
    except ImportError:
        # torchaudio 없으면 그냥 반환 (이미 16kHz인 경우가 대부분)
        return audio_array.astype(np.float32), sr


def get_duration(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    return len(audio) / sr


def _audio_path_or_bytes(audio_field: dict) -> tuple[str | None, bytes | None]:
    """Audio(decode=False) entry에서 path/bytes를 안전하게 추출."""
    if not isinstance(audio_field, dict):
        return None, None
    return audio_field.get("path"), audio_field.get("bytes")


@lru_cache(maxsize=4096)
def _search_in_hf_cache(filename: str) -> str | None:
    """HF 캐시 하위에서 basename으로 실제 파일 경로를 탐색."""
    home = Path.home()
    roots = [
        home / ".cache" / "huggingface" / "datasets",
        home / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob(filename):
                if p.is_file():
                    return str(p)
        except Exception:
            continue
    return None


def _resolve_audio_path(path: str | None, sample: dict | None = None) -> str | None:
    """상대경로/축약경로를 실제 파일 경로로 해석."""
    candidates: list[str] = []
    if path:
        candidates.append(path)
    if sample is not None:
        file_field = sample.get("file")
        if isinstance(file_field, str) and file_field:
            candidates.append(file_field)

    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return str(p)
        if not p.is_absolute():
            abs_p = (Path.cwd() / p).resolve()
            if abs_p.is_file():
                return str(abs_p)
            found = _search_in_hf_cache(p.name)
            if found is not None:
                return found

    return None


def get_audio_duration_fast(audio_field: dict, sample: dict | None = None) -> float:
    """헤더 정보만 읽어 duration 계산 (전체 split 스캔용)."""
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


def load_audio_array(audio_field: dict, sample: dict | None = None) -> tuple[np.ndarray, int]:
    """Audio(decode=False) entry에서 waveform(np.float32, mono) + sr 로드."""
    path, audio_bytes = _audio_path_or_bytes(audio_field)
    resolved_path = _resolve_audio_path(path, sample)

    if resolved_path:
        try:
            import torchaudio
            wav, sr = torchaudio.load(resolved_path)
            if wav.ndim == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).numpy().astype(np.float32), int(sr)
        except Exception:
            import soundfile as sf
            wav, sr = sf.read(resolved_path, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr)

    if audio_bytes is not None:
        import soundfile as sf
        wav, sr = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)

    raise RuntimeError(f"audio path/bytes unavailable: path={path!r}")


# ──────────────────────────────────────────────
# 샘플링
# ──────────────────────────────────────────────

def sample_from_split(split: str, n: int, seed: int) -> list[dict]:
    """
    LibriSpeech train split에서 n개를 duration 버킷 균등 샘플링.
    test split은 WER 평가 전용 — calibration에 절대 사용 금지.
    """
    print(f"  Loading librispeech_asr [{split}] ...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        split=split,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    # duration 버킷별 quota
    rng = random.Random(seed)
    per_bucket = n // 3
    counts = {
        "short":  per_bucket,
        "medium": per_bucket,
        "long":   n - 2 * per_bucket,   # 나머지 long에 배정
    }
    selected_counts = {"short": 0, "medium": 0, "long": 0}

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    selected_indices: list[int] = []
    for idx in indices:
        if len(selected_indices) >= n:
            break
        sample = ds[idx]
        dur = get_audio_duration_fast(sample["audio"], sample=sample)
        if dur < 3.0:
            bname = "short"
        elif dur < 8.0:
            bname = "medium"
        else:
            bname = "long"

        if selected_counts[bname] < counts[bname]:
            selected_indices.append(idx)
            selected_counts[bname] += 1

    if len(selected_indices) < n:
        selected_set = set(selected_indices)
        for idx in indices:
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= n:
                break

    # 레코드 생성
    records = []
    for _, idx in enumerate(selected_indices):
        sample = ds[idx]
        raw_audio, raw_sr = load_audio_array(sample["audio"], sample=sample)
        audio, sr = resample_if_needed(raw_audio, raw_sr)
        dur = get_duration(audio, sr)

        records.append({
            "id":           str(sample.get("id", idx)),
            "split":        split,
            "speaker_id":   sample.get("speaker_id", -1),
            "transcript":   sample["text"],
            "duration_sec": round(float(dur), 3),
            "sample_rate":  sr,
            "audio":        audio,
        })

    print(f"    {split}: {len(records)} samples  "
          f"(short={selected_counts['short']}, medium={selected_counts['medium']}, long={selected_counts['long']})")
    return records


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────

def save_metadata(records: list[dict], out_dir: Path) -> None:
    """audio array 제외한 메타데이터만 JSON으로 저장."""
    meta = [
        {k: v for k, v in r.items() if k != "audio"}
        for r in records
    ]
    path = out_dir / "calibration_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata → {path}")


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 50)
    print("CALIBRATION SET SUMMARY")
    print("=" * 50)
    print(f"Total          : {len(records)}")

    for split in ["train.clean.100", "train.other.500"]:
        cnt = sum(1 for r in records if r["split"] == split)
        print(f"  {split:20s}: {cnt}")

    durations = [r["duration_sec"] for r in records]
    print(f"\nDuration (sec)")
    print(f"  mean : {np.mean(durations):.2f}")
    print(f"  min  : {np.min(durations):.2f}")
    print(f"  max  : {np.max(durations):.2f}")
    print("=" * 50)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def build(n_per_split: int = 64,
          output_dir: str = "./calibration_set",
          seed: int = 42) -> list[dict]:
    """
    Calibration dataset을 빌드하고 반환.

    Parameters
    ----------
    n_per_split : int
        split(clean/other)당 샘플 수. 기본 64 → 총 128.
    output_dir : str
        메타데이터 JSON 저장 경로.
    seed : int
        재현성 시드.

    Returns
    -------
    list[dict]
        각 요소: {id, split, speaker_id, transcript,
                  duration_sec, sample_rate, audio(np.ndarray)}

    """
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for split in ["train.clean.100", "train.other.500"]:
        records = sample_from_split(split, n_per_split, seed)
        all_records.extend(records)

    print_summary(all_records)
    save_metadata(all_records, out_dir)

    return all_records
