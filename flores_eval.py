#!/usr/bin/env python3
"""
FLORES-200 evaluation with NLLB-200-3.3B.

- Loads FLORES from a local snapshot directory (path to the directory that has flores.py)
- Supports modes: baseline (fp16/bf16), int8, int4 (using bitsandbytes)
- Computes corpus BLEU and chrF for eng<->target across selected languages

Example:
  python flores_eval.py \
    --dataset_path /home/scur1844/.cache/huggingface/hub/datasets--facebook--flores/snapshots/2db78afdeaccaedc3b33a95442a4e55766887e17 \
    --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn \
    --mode int4 --directions both --batch_size 8 --max_new_tokens 256
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm

# Additional metrics
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET not available. Install with: pip install unbabel-comet")

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("Warning: evaluate library not available. Install with: pip install evaluate")

# Reuse helpers from nllb_test.py (same directory)
from nllb_test import (
    _dtype_baseline,
    _load_model_baseline,
    _load_model_int4,
    _load_model_int8,
    _load_tokenizer,
    _print_env,
)


@dataclass
class EvalConfig:
    dataset_path: str
    model_id: str
    mode: str
    langs: List[str]
    directions: str
    split: str
    batch_size: int
    max_new_tokens: int
    output_dir: str


def _prepare_model_and_tokenizer(model_id: str, mode: str):
    tokenizer = _load_tokenizer(model_id)
    if mode == "baseline":
        model = _load_model_baseline(model_id, _dtype_baseline())
    elif mode == "int8":
        model = _load_model_int8(model_id)
    elif mode == "int4":
        model = _load_model_int4(model_id)
    else:
        raise ValueError("Unknown mode")
    return model, tokenizer


def _batch_translate(
    model,
    tokenizer,
    inputs_texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_id is None or forced_bos_id < 0:
        raise ValueError(f"Could not resolve target language token id for {tgt_lang}")

    outputs: List[str] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(inputs_texts), batch_size), desc=f"{src_lang}->{tgt_lang}"):
            end = min(start + batch_size, len(inputs_texts))
            batch_texts = inputs_texts[start:end]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
            gen = model.generate(
                **enc,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            outputs.extend(decoded)
    return outputs


def _compute_advanced_metrics(sources, hypotheses, references):
    """Compute COMET-22 and kiwi-23 metrics."""
    metrics = {"comet22": None, "kiwi23": None}
    
    # COMET-22 (reference-based)
    if COMET_AVAILABLE:
        try:
            # Load COMET-22 model
            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)
            
            # Prepare data for COMET
            comet_data = []
            for src, hyp, ref in zip(sources, hypotheses, references):
                comet_data.append({"src": src, "mt": hyp, "ref": ref})
            
            # Compute COMET score
            comet_score = comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
            metrics["comet22"] = comet_score.system_score
        except Exception as e:
            print(f"Warning: Failed to compute COMET-22: {e}")
    
    # kiwi-23 (reference-free quality estimation)
    if COMET_AVAILABLE:
        try:
            # Load kiwi-23 model (reference-free)
            model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
            kiwi_model = load_from_checkpoint(model_path)
            
            # Prepare data for kiwi (reference-free, only source and MT)
            kiwi_data = []
            for src, hyp in zip(sources, hypotheses):
                kiwi_data.append({"src": src, "mt": hyp})
            
            # Compute kiwi score
            kiwi_score = kiwi_model.predict(kiwi_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
            metrics["kiwi23"] = kiwi_score.system_score
        except Exception as e:
            print(f"Warning: Failed to compute kiwi-23: {e}")
    
    return metrics


def _resolve_dataset_path(path: str) -> str:
    """Return a path suitable for datasets.load_dataset.

    If a directory is provided, append 'flores.py' inside it. If a file path
    is provided, return as-is.
    """
    if os.path.isdir(path):
        candidate = os.path.join(path, "flores.py")
        if os.path.exists(candidate):
            return candidate
    return path


def _eval_direction(
    cfg: EvalConfig,
    model,
    tokenizer,
    tgt_lang: str,
    direction: str,
) -> Tuple[float, float, float, float, float, int]:
    assert direction in {"en2x", "x2en"}

    config_name = f"eng_Latn-{tgt_lang}"
    ds = load_dataset(_resolve_dataset_path(cfg.dataset_path), name=config_name, split=cfg.split)

    if direction == "en2x":
        src_lang = "eng_Latn"
        sources = ds["sentence_eng_Latn"]
        references = ds[f"sentence_{tgt_lang}"]
    else:
        src_lang = tgt_lang
        sources = ds[f"sentence_{tgt_lang}"]
        references = ds["sentence_eng_Latn"]

    hypotheses = _batch_translate(
        model=model,
        tokenizer=tokenizer,
        inputs_texts=sources,
        src_lang=src_lang,
        tgt_lang="eng_Latn" if direction == "x2en" else tgt_lang,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
    )

    bleu = corpus_bleu(hypotheses, [references]).score
    chrf = corpus_chrf(hypotheses, [references]).score
    
    # Compute additional metrics
    advanced_metrics = _compute_advanced_metrics(sources, hypotheses, references)
    comet22 = advanced_metrics["comet22"]
    kiwi23 = advanced_metrics["kiwi23"]
    
    return bleu, chrf, comet22, kiwi23, len(hypotheses)


def _write_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lang", "direction", "mode", "split", "bleu", "chrf", "comet22", "kiwi23", "num_sentences"]) 


def run(cfg: EvalConfig) -> int:
    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_path = os.path.join(cfg.output_dir, "scores.csv")
    _write_csv_header(csv_path)

    _print_env()
    print(f"Using dataset from: {cfg.dataset_path}")
    print(f"Languages: {cfg.langs}")
    print(f"Directions: {cfg.directions}")
    print(f"Mode: {cfg.mode}")

    model, tokenizer = _prepare_model_and_tokenizer(cfg.model_id, cfg.mode)

    directions_to_run: Iterable[str]
    if cfg.directions == "both":
        directions_to_run = ("en2x", "x2en")
    else:
        directions_to_run = (cfg.directions,)

    for tgt_lang in cfg.langs:
        for direction in directions_to_run:
            print("\n" + "=" * 60)
            print(f"EVAL: {direction} {tgt_lang} [{cfg.split}]")
            print("=" * 60)
            t0 = time.time()
            try:
                bleu, chrf, comet22, kiwi23, n = _eval_direction(cfg, model, tokenizer, tgt_lang, direction)
                dt = time.time() - t0
                
                # Format metric scores for display and CSV
                comet22_str = f"{comet22:.4f}" if comet22 is not None else "N/A"
                kiwi23_str = f"{kiwi23:.4f}" if kiwi23 is not None else "N/A"
                
                print(f"BLEU: {bleu:.2f}  chrF: {chrf:.2f}  COMET-22: {comet22_str}  kiwi-23: {kiwi23_str}  N: {n}  time: {dt:.1f}s")
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([tgt_lang, direction, cfg.mode, cfg.split, 
                                   f"{bleu:.2f}", f"{chrf:.2f}", comet22_str, kiwi23_str, n])
            except Exception as e:
                print(f"FAILED for {tgt_lang} {direction}: {e}")

    print("\nResults saved to:", csv_path)
    return 0


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to local FLORES snapshot directory containing flores.py")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--mode", choices=["baseline", "int8", "int4"], default="int4")
    parser.add_argument("--langs", default="deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn")
    parser.add_argument("--directions", choices=["en2x", "x2en", "both"], default="both")
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", default="outputs/flores_eval")
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    return EvalConfig(
        dataset_path=args.dataset_path,
        model_id=args.model,
        mode=args.mode,
        langs=langs,
        directions=args.directions,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


def main() -> int:
    cfg = parse_args()
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())



