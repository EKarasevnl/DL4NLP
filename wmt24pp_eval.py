#!/usr/bin/env python3
"""
Evaluate NLLB-200-3.3B on google/wmt24pp.

- Auto-detects dataset configs and fields
- Evaluates en<->target for given 2-letter language codes
- Modes: baseline | int8 | int4 | pruned
- Metrics: corpus BLEU, chrF, COMET-22, kiwi-23
"""

import argparse
import csv
import os
from glob import glob
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
from sacrebleu import corpus_bleu, corpus_chrf

# Import shared utilities
from eval_utils import (
    prepare_model_and_tokenizer,
    batch_translate,
    compute_advanced_metrics,
    print_env,
    WMT24PP_TO_NLLB,
    get_scores_filename,
    format_metric,
)


NLLB_CODE_BY_WMT24PP = {k.replace("eng_Latn", "en"): v for k, v in WMT24PP_TO_NLLB.items() if k != "eng_Latn"}
NLLB_CODE_BY_WMT24PP["en"] = "eng_Latn"


@dataclass
class EvalConfig:
    dataset_name: str
    model_id: str
    mode: str
    langs: List[str]
    directions: str
    split: str
    batch_size: int
    max_new_tokens: int
    output_dir: str
    sparsity: float




@dataclass
class EvalConfig:
    dataset_name: str
    model_id: str
    mode: str
    langs: List[str]
    directions: str
    split: str
    batch_size: int
    max_new_tokens: int
    output_dir: str
    sparsity: float = 0.5


def _first_available_split(dataset_name: str, config_name: Optional[str], preferred: str) -> str:
    candidates_primary = [preferred]
    candidates_secondary = [
        "test",
        "validation",
        "dev",
        "devtest",
        "val",
        "test2024",
        "test_2024",
        "test-2024",
        "test_matched",
        "validation_matched",
        "test1",
        "test2",
        "train",
    ]
    # Try to inspect available splits if possible
    try:
        ds_any = load_dataset(dataset_name, name=config_name, split=None, trust_remote_code=True)
        available_splits: List[str] = []
        if hasattr(ds_any, "keys"):
            available_splits = list(ds_any.keys())  # type: ignore[attr-defined]
        # Prefer explicit candidates if present
        for s in candidates_primary + candidates_secondary:
            if s in available_splits:
                return s
        if available_splits:
            return available_splits[0]
    except Exception:
        pass

    # Fallback: probe common split names directly
    for s in candidates_primary + candidates_secondary:
        try:
            _ = load_dataset(dataset_name, name=config_name, split=s, trust_remote_code=True)
            return s
        except Exception:
            continue
    raise RuntimeError("No suitable split found for dataset")


def _load_wmt24pp(dataset_name: str, l2: str, direction: str, split: str):
    assert direction in {"en2x", "x2en"}
    configs = []
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        pass

    config_candidates: List[Optional[str]] = []
    if configs:
        pair1 = f"en-{l2}"
        pair2 = f"{l2}-en"
        # Exact pair hits first
        if pair1 in configs:
            config_candidates.append(pair1)
        if pair2 in configs:
            config_candidates.append(pair2)
        # Heuristic matches: any config containing both tokens (en and l2)
        seen: set = set(config_candidates)
        for c in configs:
            tokens = set(re.split(r"[-_]", c))
            if {"en", l2}.issubset(tokens) and c not in seen:
                config_candidates.append(c)
                seen.add(c)
    # Always try without an explicit config as a last resort
    config_candidates.append(None)

    last_err = None
    for config_name in config_candidates:
        try:
            use_split = split
            try:
                _ = load_dataset(dataset_name, name=config_name, split=use_split, trust_remote_code=True)
            except Exception:
                use_split = _first_available_split(dataset_name, config_name, split)
            ds = load_dataset(dataset_name, name=config_name, split=use_split, trust_remote_code=True)
            return ds
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load {dataset_name} for {l2}: {last_err}")


def _extract_pairs(example, l2: str, direction: str) -> Optional[Tuple[str, str]]:
    if "translation" in example and isinstance(example["translation"], dict):
        trans = example["translation"]
        if direction == "en2x":
            src = trans.get("en") or trans.get("eng")
            tgt = trans.get(l2)
        else:
            src = trans.get(l2)
            tgt = trans.get("en") or trans.get("eng")
        if isinstance(src, str) and isinstance(tgt, str):
            return src, tgt
    keys = set(example.keys())
    if {"source", "target"}.issubset(keys):
        if direction == "en2x":
            return example["source"], example["target"]
        else:
            return example["target"], example["source"]
    for src_key, tgt_key in (("src", "tgt"), ("inputs", "labels"), ("input", "label")):
        if src_key in example and tgt_key in example:
            return example[src_key], example[tgt_key]
    return None


def _list_wmt24pp_local_files(directory: str, l2: str, direction: str) -> Tuple[List[str], bool]:
    # Prefer files that match the requested direction, fall back to the opposite and swap
    files: List[str] = []
    swap = False
    # Patterns like en-de_DE.jsonl or de-en_DE.jsonl
    en2x_patterns = [
        os.path.join(directory, f"en-{l2}_*.jsonl"),
        os.path.join(directory, f"en-{l2}-*.jsonl"),
        os.path.join(directory, f"en-{l2}.jsonl"),
    ]
    x2en_patterns = [
        os.path.join(directory, f"{l2}-en_*.jsonl"),
        os.path.join(directory, f"{l2}-en-*.jsonl"),
        os.path.join(directory, f"{l2}-en.jsonl"),
    ]
    if direction == "en2x":
        for p in en2x_patterns:
            files.extend(sorted(glob(p)))
        if not files:
            # Fallback to opposite and swap later
            for p in x2en_patterns:
                files.extend(sorted(glob(p)))
            swap = bool(files)
    else:
        for p in x2en_patterns:
            files.extend(sorted(glob(p)))
        if not files:
            for p in en2x_patterns:
                files.extend(sorted(glob(p)))
            swap = bool(files)
    return files, swap


def _load_wmt24pp_local(directory: str, l2: str, direction: str) -> Tuple[List[str], List[str]]:
    data_files, swap = _list_wmt24pp_local_files(directory, l2, direction)
    if not data_files:
        raise RuntimeError(f"No local files found for {l2} in {directory}")
    ds = load_dataset("json", data_files=data_files, split="train")
    sources: List[str] = []
    references: List[str] = []
    for ex in ds:
        if ex.get("is_bad_source", False):
            continue
        src = ex.get("source")
        tgt = ex.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if swap:
            src, tgt = tgt, src
        if src and tgt:
            sources.append(src)
            references.append(tgt)
    return sources, references


def _collect_sources_refs(ds, l2: str, direction: str) -> Tuple[List[str], List[str]]:
    sources: List[str] = []
    references: List[str] = []
    for ex in ds:
        pair = _extract_pairs(ex, l2, direction)
        if pair is None:
            continue
        src, ref = pair
        if isinstance(src, str) and isinstance(ref, str) and src and ref:
            sources.append(src)
            references.append(ref)
    return sources, references


def run(cfg: EvalConfig) -> int:
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Name output file to include mode and sparsity (when applicable)
    if getattr(cfg, "mode", None) == "pruned":
        sparsity_str = f"s{cfg.sparsity:.2f}".replace('.', 'p')
        csv_name = f"scores_{cfg.mode}_{sparsity_str}.csv"
    else:
        csv_name = f"scores_{cfg.mode}.csv"
    csv_path = os.path.join(cfg.output_dir, csv_name)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lang", "direction", "mode", "split", "bleu", "chrf", "comet22", "kiwi23", "num_sentences"]) 

    eval_utils.print_env()
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Languages: {cfg.langs}")
    print(f"Directions: {cfg.directions}")
    print(f"Mode: {cfg.mode}")
    print(f"Sparsity: {cfg.sparsity}")

    model, tokenizer = eval_utils.prepare_model_and_tokenizer(cfg.model_id, cfg.mode, cfg.sparsity)

    directions_to_run: Iterable[str]
    if cfg.directions == "both":
        directions_to_run = ("en2x", "x2en")
    else:
        directions_to_run = (cfg.directions,)

    for l2 in cfg.langs:
        if l2 not in NLLB_CODE_BY_WMT24PP:
            print(f"Skipping unsupported language code: {l2}")
            continue
        nllb_tgt_code = NLLB_CODE_BY_WMT24PP[l2]
        for direction in directions_to_run:
            print("\n" + "=" * 60)
            print(f"EVAL: {direction} {l2} [{cfg.split}]")
            print("=" * 60)
            t0 = time.time()
            try:
                # If dataset_name is a directory, use local JSONL loader; else use HF hub
                if os.path.isdir(cfg.dataset_name):
                    sources, references = _load_wmt24pp_local(cfg.dataset_name, l2, direction)
                else:
                    ds = _load_wmt24pp(cfg.dataset_name, l2, direction, cfg.split)
                    sources, references = _collect_sources_refs(ds, l2, direction)
                if not sources:
                    raise RuntimeError("No sentence pairs extracted from dataset")
                hypotheses = eval_utils.batch_translate(
                    model=model,
                    tokenizer=tokenizer,
                    inputs_texts=sources,
                    src_lang="eng_Latn" if direction == "en2x" else nllb_tgt_code,
                    tgt_lang=nllb_tgt_code if direction == "en2x" else "eng_Latn",
                    batch_size=cfg.batch_size,
                    max_new_tokens=cfg.max_new_tokens,
                )
                bleu = corpus_bleu(hypotheses, [references]).score
                chrf = corpus_chrf(hypotheses, [references]).score
                
                # Compute additional metrics
                advanced_metrics = eval_utils.compute_advanced_metrics(sources, hypotheses, references)
                comet22 = advanced_metrics["comet22"]
                kiwi23 = advanced_metrics["kiwi23"]
                
                dt = time.time() - t0
                
                # Format metric scores for display and CSV
                comet22_str = f"{comet22:.4f}" if comet22 is not None else "N/A"
                kiwi23_str = f"{kiwi23:.4f}" if kiwi23 is not None else "N/A"
                
                print(f"BLEU: {bleu:.2f}  chrF: {chrf:.2f}  COMET-22: {comet22_str}  kiwi-23: {kiwi23_str}  N: {len(hypotheses)}  time: {dt:.1f}s")
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([l2, direction, cfg.mode, cfg.split, 
                                   f"{bleu:.2f}", f"{chrf:.2f}", comet22_str, kiwi23_str, len(hypotheses)])
            except Exception as e:
                print(f"FAILED for {l2} {direction}: {e}")

    print("\nResults saved to:", csv_path)
    return 0


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="google/wmt24pp")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--mode", choices=["baseline", "int8", "int4", "pruned"], default="int4")
    parser.add_argument("--langs", default="de,ru,fr,nl,pl,lv,zu,te,sw")
    parser.add_argument("--directions", choices=["en2x", "x2en", "both"], default="both")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", default="outputs/wmt24pp_eval")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level for pruned mode (0.0-1.0)")
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    return EvalConfig(
        dataset_name=args.dataset,
        model_id=args.model,
        mode=args.mode,
        langs=langs,
        directions=args.directions,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        sparsity=args.sparsity,
    )


def main() -> int:
    cfg = parse_args()
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())



