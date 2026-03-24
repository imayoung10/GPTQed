import json
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets

"""
처음 gpt 제안 데이터셋 이름
A : librispeech Basic dataset
B : librispeech Long dataset (duration >= 10s)
C : CommonVoice single ln
Optional : CommonVoice multilingual

바꾼 것
LS_b(asic) : LibriSpeech clean, normal-length
LS_l(ong) : LibriSpeech clean, long-form
CV_s(ingle) : Common Voice single-language hard/distribution-shift set
CV_m(ulti) : Common Voice multilingual

"""
def add_duration(example):
    example["duration"] = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return example


def load_librispeech_clean_train():
    ds = load_dataset("librispeech_asr", "train-clean-100", split="train")
    ds = ds.map(add_duration, desc="Adding duration to LibriSpeech")
    return ds


def build_librispeech_calisets(
    nsamples_b=128,
    nsamples_l=128,
    seed=0,
    basic_min_duration=2.0,
    basic_max_duration=8.0,
    long_min_duration=10.0,
):
    """
    A: LibriSpeech clean, normal-length
    B: LibriSpeech clean, long-form
    A/B are made disjoint by original HF row index.
    """
    ds = load_librispeech_clean_train()

    # Add original HF row index so we can save exact subset membership
    ds = ds.map(lambda ex, idx: {"orig_idx": idx}, with_indices=True, desc="Adding orig_idx")

    ls_b_pool = ds.filter(
        lambda x: basic_min_duration <= x["duration"] <= basic_max_duration,
        desc="Filtering A pool",
    )

    ls_l_pool = ds.filter(
        lambda x: x["duration"] >= long_min_duration,
        desc="Filtering B pool",
    )

    if len(ls_b_pool) < nsamples_b:
        raise ValueError(f"Not enough samples for A: requested {nsamples_b}, found {len(ls_b_pool)}")
    if len(ls_l_pool) < nsamples_l:
        raise ValueError(f"Not enough samples for B: requested {nsamples_l}, found {len(ls_l_pool)}")

    rng_ls_b = random.Random(seed)
    ls_b_local_indices = rng_ls_b.sample(range(len(ls_b_pool)), nsamples_b)
    ls_b_set = ls_b_pool.select(ls_b_local_indices)

    ls_b_orig_idx = set(ls_b_set["orig_idx"])

    ls_l_pool_no_overlap = ls_l_pool.filter(
        lambda x: x["orig_idx"] not in ls_b_orig_idx,
        desc="Removing A/B overlap",
    )

    if len(ls_l_pool_no_overlap) < nsamples_l:
        raise ValueError(
            f"Not enough disjoint samples for ls_l after overlap removal: "
            f"requested {nsamples_l}, found {len(ls_l_pool_no_overlap)}"
        )

    rng_ls_l = random.Random(seed + 1)
    ls_l_local_indices = rng_ls_l.sample(range(len(ls_l_pool_no_overlap)), nsamples_l)
    ls_l_set = ls_l_pool_no_overlap.select(ls_l_local_indices)

    return ls_b_set, ls_l_set


def build_commonvoice_caliset(
    lang,
    nsamples=128,
    seed=0,
    min_duration=2.0,
    max_duration=15.0,
):
    """
    C: Common Voice single-language hard/distribution-shift set
    """
    ds = load_dataset("mozilla-foundation/common_voice_13_0", lang, split="train")
    ds = ds.map(add_duration, desc=f"Adding duration to Common Voice ({lang})")
    ds = ds.map(lambda ex, idx: {"orig_idx": idx}, with_indices=True, desc="Adding orig_idx")

    ds = ds.filter(
        lambda x: min_duration <= x["duration"] <= max_duration,
        desc="Filtering Common Voice by duration",
    )

    if len(ds) < nsamples:
        raise ValueError(f"Not enough samples for CV: requested {nsamples}, found {len(ds)}")

    rng = random.Random(seed)
    local_indices = rng.sample(range(len(ds)), nsamples)
    cv_s_set = ds.select(local_indices)

    return cv_s_set


def build_commonvoice_calisets(
    langs,
    nsamples_per_lang=43,
    seed=0,
    min_duration=2.0,
    max_duration=15.0,
):

    parts = []
    for i, lang in enumerate(langs):
        part = build_commonvoice_caliset(
            lang=lang,
            nsamples=nsamples_per_lang,
            seed=seed + i,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        part = part.map(lambda ex: {"cv_lang": lang}, desc=f"Annotating lang={lang}")
        parts.append(part)

    cv_m_set = concatenate_datasets(parts)
    return cv_m_set


def save_subset_metadata(dataset_name, config, split_name, subset_name, hf_indices, ids, path):
    payload = {
        "dataset_name": dataset_name,
        "config" : config,
        "split": split_name,
        "subset_name": subset_name,
        "hf_indices": list(hf_indices),
        "ids": list(ids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    out_dir = "./calibration_sets"

    # A/B from LibriSpeech
    A_set, B_set = build_librispeech_calisets(
        nsamples_A=128,
        nsamples_B=128,
        seed=0,
        A_min_duration=2.0,
        A_max_duration=8.0,
        B_min_duration=10.0,
    )

    # C from Common Voice (single-language)
    C_set = build_commonvoice_caliset(
        nsamples=128,
        seed=0,
        lang="en",
        min_duration=2.0,
        max_duration=15.0,
    )

    # Save all
    save_subset_to_disk_and_json(
        A_set,
        subset_name="calib_A",
        dataset_name="librispeech_asr",
        split_name="train-clean-100/train",
        out_dir=out_dir,
    )
    save_subset_to_disk_and_json(
        B_set,
        subset_name="calib_B",
        dataset_name="librispeech_asr",
        split_name="train-clean-100/train",
        out_dir=out_dir,
    )
    save_subset_to_disk_and_json(
        C_set,
        subset_name="calib_C",
        dataset_name="mozilla-foundation/common_voice_13_0",
        split_name="train",
        out_dir=out_dir,
    )

    print("\nSummary")
    print(f"A size: {len(A_set)}, mean duration: {sum(A_set['duration']) / len(A_set):.2f}s")
    print(f"B size: {len(B_set)}, mean duration: {sum(B_set['duration']) / len(B_set):.2f}s")
    print(f"C size: {len(C_set)}, mean duration: {sum(C_set['duration']) / len(C_set):.2f}s")


if __name__ == "__main__":
    main()