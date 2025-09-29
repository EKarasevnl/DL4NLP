#!/usr/bin/env python3
"""
Multi-language evaluation with NLLB-200-3.3B for non-English language pairs.

Performs evaluation between all pairs of non-English languages, excluding
translations to/from English (which are handled by other evaluation scripts).

Supports both FLORES-200 and WMT24++ datasets.

Example:
  python multilang_eval.py \
    --flores_path /home/scur1832/.cache/huggingface/modules/datasets_modules/datasets/facebook--flores \
    --wmt24pp_path data/wmt24pp \
    --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn \
    --mode int4 --batch_size 8 --max_new_tokens 256
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
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
    flores_path: str
    wmt24pp_path: str
    model_id: str
    mode: str
    langs: List[str]
    src_lang: str  # Specific source language for parallel processing
    split: str
    batch_size: int
    max_new_tokens: int
    output_dir: str


def _prepare_model_and_tokenizer(model_id: str, mode: str):
    """Load model and tokenizer based on mode."""
    tokenizer = _load_tokenizer(model_id)
    if mode == "baseline":
        dtype = _dtype_baseline()
        model = _load_model_baseline(model_id, dtype)
    elif mode == "int8":
        model = _load_model_int8(model_id)
    elif mode == "int4":
        model = _load_model_int4(model_id)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model, tokenizer


def _translate_batch(
    model, tokenizer, src_lang: str, tgt_lang: str, inputs_texts: List[str],
    batch_size: int, max_new_tokens: int
) -> List[str]:
    """Translate a batch of texts from src_lang to tgt_lang."""
    
    # NLLB requires setting tokenizer.src_lang and forcing BOS of target
    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_id is None or forced_bos_id < 0:
        raise ValueError(f"Could not resolve target language token id for {tgt_lang}")

    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs_texts), batch_size):
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


def _lang_code_to_wmt_code(lang_code: str) -> str:
    """Convert FLORES language code to WMT24++ 2-letter code."""
    mapping = {
        "deu_Latn": "de", "rus_Cyrl": "ru", "fra_Latn": "fr", "nld_Latn": "nl",
        "pol_Latn": "pl", "lvs_Latn": "lv", "zul_Latn": "zu", "tel_Telu": "te", 
        "swh_Latn": "sw"
    }
    return mapping.get(lang_code, lang_code)


def _eval_flores_pair(
    cfg: EvalConfig, model, tokenizer, src_lang: str, tgt_lang: str
) -> Tuple[float, float, float, float, int]:
    """Evaluate translation between a language pair using FLORES dataset."""
    
    # Try loading FLORES dataset with multiple fallback approaches
    dataset = None
    
    # Method 1: Try the standard HuggingFace approach first (most reliable)
    try:
        dataset = load_dataset('facebook/flores', 'all', trust_remote_code=True)
        print(f"Loaded FLORES dataset using 'facebook/flores' configuration")
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        # Method 2: Try using the provided flores_path
        try:
            flores_path = cfg.flores_path
            if os.path.isdir(flores_path):
                candidate = os.path.join(flores_path, "flores.py")
                if os.path.exists(candidate):
                    flores_path = candidate
            
            # Try specific language pair config
            config_name = f"{src_lang}-{tgt_lang}"
            dataset = load_dataset(flores_path, config_name, trust_remote_code=True)
            print(f"Loaded FLORES dataset using local path with config '{config_name}'")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            raise ValueError(f"Failed to load FLORES dataset with both methods. HF error: {e1}. Local error: {e2}")
    
    if cfg.split not in dataset:
        raise ValueError(f"Split '{cfg.split}' not found in FLORES dataset. Available: {list(dataset.keys())}")
    
    data = dataset[cfg.split]
    
    # Extract source and reference texts using column names
    src_col = f"sentence_{src_lang}"
    tgt_col = f"sentence_{tgt_lang}"
    
    # Check if columns exist
    sample = data[0]
    if src_col not in sample:
        raise ValueError(f"Source language column '{src_col}' not found in FLORES dataset")
    if tgt_col not in sample:
        raise ValueError(f"Target language column '{tgt_col}' not found in FLORES dataset")
    
    src_texts = [ex[src_col] for ex in data]
    ref_texts = [ex[tgt_col] for ex in data]
    
    # Translate
    hyp_texts = _translate_batch(
        model, tokenizer, src_lang, tgt_lang, src_texts,
        cfg.batch_size, cfg.max_new_tokens
    )
    
    # Compute BLEU and chrF
    bleu = corpus_bleu(hyp_texts, [ref_texts]).score
    chrf = corpus_chrf(hyp_texts, [ref_texts]).score
    
    # Compute advanced metrics
    advanced = _compute_advanced_metrics(src_texts, hyp_texts, ref_texts)
    
    return bleu, chrf, advanced["comet22"], advanced["kiwi23"], len(src_texts)


def _eval_wmt24pp_pair(
    cfg: EvalConfig, model, tokenizer, src_lang: str, tgt_lang: str
) -> Tuple[float, float, float, float, float, int]:
    """Evaluate translation between a language pair using WMT24++ dataset."""
    
    # Convert to WMT codes
    src_wmt = _lang_code_to_wmt_code(src_lang)
    tgt_wmt = _lang_code_to_wmt_code(tgt_lang)
    
    # Try both direction files (en-XX format)
    config_file = None
    reverse = False
    
    # Try source->target direction
    config_path = os.path.join(cfg.wmt24pp_path, f"en-{tgt_wmt}_XX.jsonl")
    if os.path.exists(config_path.replace("_XX", "_" + tgt_wmt.upper())):
        config_file = config_path.replace("_XX", "_" + tgt_wmt.upper())
    elif os.path.exists(config_path.replace("_XX", "")):
        config_file = config_path.replace("_XX", "")
    
    if config_file is None:
        # Try reverse direction
        config_path = os.path.join(cfg.wmt24pp_path, f"en-{src_wmt}_XX.jsonl")
        if os.path.exists(config_path.replace("_XX", "_" + src_wmt.upper())):
            config_file = config_path.replace("_XX", "_" + src_wmt.upper())
            reverse = True
        elif os.path.exists(config_path.replace("_XX", "")):
            config_file = config_path.replace("_XX", "")
            reverse = True
    
    if config_file is None:
        raise ValueError(f"No WMT24++ data found for {src_lang}-{tgt_lang} pair")
    
    # Load the JSONL file
    import json
    with open(config_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Extract texts (WMT24++ has 'en' and target language fields)
    if reverse:
        # Use target language as source and English as intermediate
        src_texts = [ex[tgt_wmt] for ex in data if tgt_wmt in ex]
        # We need to translate to English first, then to target - this is complex
        # For now, skip this complex case
        raise ValueError(f"Reverse direction not supported for WMT24++ evaluation")
    else:
        # Extract English and target language
        src_texts = [ex['en'] for ex in data if 'en' in ex]
        ref_texts = [ex[tgt_wmt] for ex in data if tgt_wmt in ex]
        
        if len(src_texts) != len(ref_texts):
            raise ValueError(f"Mismatched data lengths in {config_file}")
    
    # Since WMT24++ is English-centric, we need to handle non-English pairs differently
    # For now, we'll simulate by using English as intermediate
    # This is a limitation - ideally we'd have direct non-English pairs
    raise ValueError("Direct non-English evaluation not supported with WMT24++ dataset format")


def run(cfg: EvalConfig) -> int:
    """Run multi-language evaluation."""
    
    _print_env()
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Create separate output file for each source language and mode
    if cfg.src_lang:
        csv_filename = f"multilang_scores_{cfg.src_lang}_{cfg.mode}.csv"
        print(f"Running evaluation for source language: {cfg.src_lang}, mode: {cfg.mode}")
    else:
        csv_filename = f"multilang_scores_{cfg.mode}.csv"
        print(f"Running evaluation for all language pairs, mode: {cfg.mode}")
    
    csv_path = os.path.join(cfg.output_dir, csv_filename)
    
    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_lang", "tgt_lang", "dataset", "mode", "split", 
                        "bleu", "chrf", "comet22", "kiwi23", "n_samples"])
    
    print("Dataset paths:")
    print(f"  FLORES: {cfg.flores_path}")
    print(f"  WMT24++: {cfg.wmt24pp_path}")
    print(f"Languages: {cfg.langs}")
    print(f"Mode: {cfg.mode}")

    model, tokenizer = _prepare_model_and_tokenizer(cfg.model_id, cfg.mode)

    # Generate language pairs based on configuration
    if cfg.src_lang:
        # Parallel mode: evaluate only from the specified source language
        target_langs = [lang for lang in cfg.langs if lang != cfg.src_lang]
        lang_pairs = [(cfg.src_lang, tgt_lang) for tgt_lang in target_langs]
        print(f"\nEvaluating {len(lang_pairs)} language pairs from {cfg.src_lang}...")
    else:
        # Original mode: evaluate all language pairs in both directions
        lang_pairs = list(combinations(cfg.langs, 2))
        print(f"\nEvaluating {len(lang_pairs)} language pairs in both directions...")
    
    for pair_idx, (src_lang, tgt_lang) in enumerate(lang_pairs):
        if cfg.src_lang:
            # Parallel mode: only evaluate src_lang -> tgt_lang
            directions = [(src_lang, tgt_lang)]
        else:
            # Original mode: evaluate both directions
            directions = [(src_lang, tgt_lang), (tgt_lang, src_lang)]
        
        for direction_src, direction_tgt in directions:
            
            # Evaluate FLORES
            print("\n" + "=" * 80)
            print(f"FLORES EVAL: {direction_src} -> {direction_tgt} [{cfg.split}]")
            print("=" * 80)
            t0 = time.time()
            try:
                bleu, chrf, comet22, kiwi23, n = _eval_flores_pair(
                    cfg, model, tokenizer, direction_src, direction_tgt
                )
                dt = time.time() - t0
                
                # Format metric scores for display and CSV
                comet22_str = f"{comet22:.4f}" if comet22 is not None else "N/A"
                kiwi23_str = f"{kiwi23:.4f}" if kiwi23 is not None else "N/A"
                
                print(f"BLEU: {bleu:.2f}  chrF: {chrf:.2f}  COMET-22: {comet22_str}  kiwi-23: {kiwi23_str}  N: {n}  time: {dt:.1f}s")
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([direction_src, direction_tgt, "flores", cfg.mode, cfg.split, 
                                   f"{bleu:.2f}", f"{chrf:.2f}", comet22_str, kiwi23_str, n])
            except Exception as e:
                print(f"FAILED FLORES {direction_src}->{direction_tgt}: {e}")

    print("\nResults saved to:", csv_path)
    return 0


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores_path", required=True, 
                       help="Path to local FLORES snapshot directory containing flores.py")
    parser.add_argument("--wmt24pp_path", required=True,
                       help="Path to WMT24++ dataset directory")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    parser.add_argument("--mode", choices=["baseline", "int8", "int4"], default="int4")
    parser.add_argument("--langs", default="deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn")
    parser.add_argument("--src_lang", default=None, 
                       help="Specific source language to evaluate from (for parallel processing)")
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", default="outputs/multilang_eval")
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    
    # If src_lang is specified, validate it's in the language list
    src_lang = args.src_lang
    if src_lang and src_lang not in langs:
        raise ValueError(f"Source language '{src_lang}' not found in language list: {langs}")
    
    return EvalConfig(
        flores_path=args.flores_path,
        wmt24pp_path=args.wmt24pp_path,
        model_id=args.model,
        mode=args.mode,
        langs=langs,
        src_lang=src_lang,
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