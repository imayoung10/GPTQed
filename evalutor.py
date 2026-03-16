import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import evaluate as evaluate_lib
from datasets import load_dataset, concatenate_datasets
from whisper.normalizers import EnglishTextNormalizer

from qwen_asr import Qwen3ASRModel



_normalizer = EnglishTextNormalizer()


# =========================================================
# 설정
# =========================================================

# LibriSpeech 평가 split
# test-clean : 발음 명확한 오디오북 낭독체
# test-other : 더 어려운 화자 (noise, accent 등)
LIBRISPEECH_SPLITS = {
    "test-clean": "test.clean",
    "test-other": "test.other",
}


# =========================================================
# Text normalization
# =========================================================

def normalize_text(text: str) -> str:
    """
    Whisper EnglishTextNormalizer 기반 정규화.
    숫자 영어 단어 변환, 축약형 펼치기, 구두점 제거 등을 포함하므로
    LibriSpeech 벤치마크 수치 재현에 적합.
    """
    if text is None:
        return ""
    return _normalizer(text)


# =========================================================
# 데이터 로드
# =========================================================

def load_librispeech_samples(split_name: str) -> List[Dict[str, Any]]:
    """
    HuggingFace에서 LibriSpeech split을 로드해 샘플 리스트로 반환.

    Parameters
    ----------
    split_name : str
        "test-clean" 또는 "test-other"

    Returns
    -------
    list of dict
        [{"id", "audio_array", "sampling_rate", "reference"}, ...]
    """
    hf_split = LIBRISPEECH_SPLITS[split_name]
    print(f"  Loading LibriSpeech [{hf_split}] ...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        split=hf_split,
        trust_remote_code=True,
    )

    samples = []
    for i, row in enumerate(ds):
        samples.append({
            "id":            str(row.get("id", i)),
            "audio_array":   row["audio"]["array"],
            "sampling_rate": row["audio"]["sampling_rate"],
            "reference":     row["text"],
        })

    print(f"  Loaded {len(samples)} samples")
    return samples


# =========================================================
# Utils
# =========================================================

def batchify(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


# =========================================================
# 평가
# =========================================================

def evaluate_split(
    model: "Qwen3ASRModel",
    samples: List[Dict[str, Any]],
    wer_metric,
    batch_size: int,
) -> Dict[str, Any]:
    """
    단일 split에 대해 추론 + WER 계산.

    Returns
    -------
    dict
        {"wer": float, "records": [...]}
    """
    predictions = []
    references  = []
    records     = []

    for batch in batchify(samples, batch_size):
        # Qwen3ASRModel.transcribe는 numpy array 또는 파일 경로 둘 다 지원
        batch_audio = [x["audio_array"] for x in batch]
        batch_refs  = [x["reference"]   for x in batch]

        results = model.transcribe(
            audio=batch_audio,
            context="",
            return_time_stamps=False,
        )

        for sample, result, ref in zip(batch, results, batch_refs):
            pred_raw  = result.text
            ref_raw   = ref
            pred_norm = normalize_text(pred_raw)
            ref_norm  = normalize_text(ref_raw)

            predictions.append(pred_norm)
            references.append(ref_norm)

            records.append({
                "id":               sample["id"],
                "language_pred":    result.language,
                "prediction_raw":   pred_raw,
                "reference_raw":    ref_raw,
                "prediction_norm":  pred_norm,
                "reference_norm":   ref_norm,
            })

    wer = wer_metric.compute(predictions=predictions, references=references)
    return {"wer": wer, "records": records}


def run_evaluation(
    model_name: str = "Qwen/Qwen3-ASR-0.6B",
    splits: List[str] = ("test-clean", "test-other"),
    batch_size: int = 8,
    max_new_tokens: int = 512,
    save_path: str | None = None,
) -> Dict[str, Any]:
    """
    LibriSpeech WER 평가.

    Parameters
    ----------
    model_name : str
        HuggingFace 모델 ID 또는 로컬 경로.
    splits : list of str
        평가할 split 목록. "test-clean", "test-other" 중 선택.
    batch_size : int
        배치 크기.
    max_new_tokens : int
        생성 최대 토큰 수.
    save_path : str, optional
        결과를 저장할 JSON 경로. None이면 저장 안 함.

    Returns
    -------
    dict
        {"test-clean": {"wer": float, "records": [...]}, ...}

    Example
    -------
    >>> result = run_evaluation("Qwen/Qwen3-ASR-0.6B")
    >>> # test-clean WER: 2.11%
    >>> # test-other WER: 4.55%
    """
    print("Loading metrics...")
    wer_metric = evaluate_lib.load("wer")

    print("Loading model...")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    output = {}

    for split_name in splits:
        print(f"\n[{split_name}]")
        samples = load_librispeech_samples(split_name)

        print(f"  Running inference (batch_size={batch_size}) ...")
        result  = evaluate_split(model, samples, wer_metric, batch_size)

        wer_val = result["wer"]
        output[split_name] = result

        print(f"  Total samples : {len(result['records'])}")
        print(f"  WER           : {wer_val:.4f}  ({wer_val * 100:.2f}%)")

    # ── 요약 출력 ──
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for split_name, result in output.items():
        wer_val = result["wer"]
        print(f"  {split_name:12s}: WER={wer_val * 100:.2f}%")

    # ── 저장 ──
    if save_path is not None:
        _save_results(output, save_path)

    return output


# =========================================================
# 저장
# =========================================================

def _save_results(output: Dict[str, Any], save_path: str):
    save_path = Path(save_path)

    serializable = {}
    for split_name, result in output.items():
        serializable[split_name] = {
            "wer":     float(result["wer"]),
            "records": result["records"],
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {save_path}")
    
import time
import torch

def evaluate_split_with_perf(
    model,
    samples,
    wer_metric,
    batch_size: int,
    dev: str,
):
    predictions = []
    references = []
    records = []

    total_infer_time = 0.0
    total_audio_sec = 0.0
    batch_latencies = []

    use_cuda = str(dev).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    for batch in batchify(samples, batch_size):
        batch_audio = [x["audio_array"] for x in batch]
        batch_refs = [x["reference"] for x in batch]

        # audio duration accumulation
        batch_audio_sec = sum(len(x["audio_array"]) / x["sampling_rate"] for x in batch)
        total_audio_sec += batch_audio_sec

        if use_cuda:
            torch.cuda.synchronize(dev)
        start = time.perf_counter()

        results = model.transcribe(
            audio=batch_audio,
            context="",
            return_time_stamps=False,
        )

        if use_cuda:
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start

        total_infer_time += elapsed
        batch_latencies.append(elapsed)

        for sample, result, ref in zip(batch, results, batch_refs):
            pred_raw = result.text
            ref_raw = ref
            pred_norm = normalize_text(pred_raw)
            ref_norm = normalize_text(ref_raw)

            predictions.append(pred_norm)
            references.append(ref_norm)

            records.append({
                "id": sample["id"],
                "language_pred": result.language,
                "prediction_raw": pred_raw,
                "reference_raw": ref_raw,
                "prediction_norm": pred_norm,
                "reference_norm": ref_norm,
            })

    wer = wer_metric.compute(predictions=predictions, references=references)

    peak_vram_mb = None
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    avg_latency_sec = total_infer_time / max(len(batch_latencies), 1)
    avg_rtf = total_infer_time / max(total_audio_sec, 1e-8)

    return {
        "wer": wer,
        "records": records,
        "total_infer_time_sec": total_infer_time,
        "total_audio_sec": total_audio_sec,
        "avg_batch_latency_sec": avg_latency_sec,
        "rtf": avg_rtf,
        "peak_vram_mb": peak_vram_mb,
    }