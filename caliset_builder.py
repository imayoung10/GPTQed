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
from pathlib import Path

import numpy as np
from datasets import load_dataset


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
        trust_remote_code=True,
    )

    # duration 계산 후 버킷 분류
    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {"short": [], "medium": [], "long": []}

    for idx, sample in enumerate(ds):
        dur = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        if dur < 3.0:
            buckets["short"].append(idx)
        elif dur < 8.0:
            buckets["medium"].append(idx)
        else:
            buckets["long"].append(idx)

    per_bucket = n // 3
    counts = {
        "short":  per_bucket,
        "medium": per_bucket,
        "long":   n - 2 * per_bucket,   # 나머지 long에 배정
    }

    selected_indices = []
    for bname, cnt in counts.items():
        pool = buckets[bname]
        rng.shuffle(pool)
        selected_indices.extend(pool[:cnt])

    # 인덱스 순서 섞기
    rng.shuffle(selected_indices)

    # 레코드 생성
    records = []
    for _, idx in enumerate(selected_indices):
        sample = ds[idx]
        audio, sr = resample_if_needed(
            np.array(sample["audio"]["array"], dtype=np.float32),
            sample["audio"]["sampling_rate"],
        )
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
          f"(short={counts['short']}, medium={counts['medium']}, long={counts['long']})")
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


