#!/usr/bin/env python3
"""
WMT24++ Non-English Language Pair Evaluation for NLLB-200-3.3B

- Evaluates non-English language pairs among the 9 specified languages
- Uses WMT24++ dataset with English pivoting (source→English→target)
- Supports baseline, int8, and int4 quantization modes
- Metrics: BLEU, chrF, COMET-22, kiwi-23
- Parallel evaluation across language pairs without English pairs
"""

import argparse
import csv
import itertools
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    print("Warning: evaluate library not available. Install with: pip install unbabel-comet")

# Use shared utilities from eval_utils
import eval_utils

# The 9 languages to evaluate (same as multilang_eval.py)
TARGET_LANGUAGES = [
    "deu_Latn", "rus_Cyrl", "fra_Latn", "nld_Latn", "pol_Latn", 
    "lvs_Latn", "zul_Latn", "tel_Telu", "swh_Latn"
]

@dataclass
class EvalConfig:
    wmt24pp_path: str
    source_lang: str  # One of the 9 target languages
    target_langs: List[str]  # Other languages to translate to
    model_id: str
    mode: str
    batch_size: int
    max_new_tokens: int
    output_dir: str
    max_examples: int = 1000
    comet_model_path: Optional[str] = None
    kiwi_model_path: Optional[str] = None,
    sparsity: float = 0.5  # For pruned mode


# Map NLLB language codes to WMT24++ 2-letter codes
NLLB_TO_WMT24PP = eval_utils.NLLB_TO_WMT24PP

def _load_wmt24pp_language_pair(wmt24pp_path: str, src_lang: str, tgt_lang: str, max_examples: int = 1000) -> Tuple[List[str], List[str]]:
    """Load WMT24++ data and create direct source→target pairs using English alignment."""
    import json
    import glob
    
    # Get WMT language codes
    src_wmt = NLLB_TO_WMT24PP.get(src_lang)
    tgt_wmt = NLLB_TO_WMT24PP.get(tgt_lang)
    
    if not src_wmt or not tgt_wmt:
        raise ValueError(f"Language mapping not found for {src_lang} or {tgt_lang}")
    
    # Load source language data (EN-SRC pairs)
    src_files = glob.glob(os.path.join(wmt24pp_path, f"en-{src_wmt}_*.jsonl"))
    if not src_files:
        src_files = glob.glob(os.path.join(wmt24pp_path, f"en-{src_wmt}.jsonl"))
    
    # Load target language data (EN-TGT pairs) 
    tgt_files = glob.glob(os.path.join(wmt24pp_path, f"en-{tgt_wmt}_*.jsonl"))
    if not tgt_files:
        tgt_files = glob.glob(os.path.join(wmt24pp_path, f"en-{tgt_wmt}.jsonl"))
    
    if not src_files or not tgt_files:
        raise ValueError(f"WMT24++ files not found for {src_lang} ({src_files}) or {tgt_lang} ({tgt_files})")
    
    print(f"Loading {src_lang} data from: {src_files[0]}")
    print(f"Loading {tgt_lang} data from: {tgt_files[0]}")
    
    # Load source language sentences with English alignment
    src_data = {}
    with open(src_files[0], 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(src_data) >= max_examples:
                break
            item = json.loads(line.strip())
            if item.get('is_bad_source', False):
                continue
            
            # For en-XX_YY.jsonl files: source=English, target=XX language
            en_text = item.get('source', '')  # English text
            lang_text = item.get('target', '')  # Non-English language text
            
            if en_text and lang_text:
                # Use line number as alignment key
                src_data[line_num] = (en_text, lang_text)
    
    # Load target language sentences with English alignment
    tgt_data = {}
    with open(tgt_files[0], 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if len(tgt_data) >= max_examples:
                break
            item = json.loads(line.strip())
            if item.get('is_bad_source', False):
                continue
            
            # For en-XX_YY.jsonl files: source=English, target=XX language  
            en_text = item.get('source', '')  # English text
            lang_text = item.get('target', '')  # Non-English language text
            
            if en_text and lang_text:
                # Use line number as alignment key
                tgt_data[line_num] = (en_text, lang_text)
    
    # Align source and target using English as pivot
    aligned_sources = []
    aligned_targets = []
    english_mismatches = 0
    total_checked = 0
    
    for line_num in src_data:
        if line_num in tgt_data:
            src_en, src_lang_text = src_data[line_num]
            tgt_en, tgt_lang_text = tgt_data[line_num]
            
            # Verify English pivot alignment
            total_checked += 1
            if src_en.strip() != tgt_en.strip():
                english_mismatches += 1
                if english_mismatches <= 3:  # Show first few mismatches
                    print(f"WARNING: English mismatch at line {line_num}")
                    print(f"  Source EN: '{src_en[:100]}...'")
                    print(f"  Target EN: '{tgt_en[:100]}...'")
            
            # Add the non-English language texts as source→target pair
            aligned_sources.append(src_lang_text)
            aligned_targets.append(tgt_lang_text)
    
    print(f"Created {len(aligned_sources)} aligned {src_lang}→{tgt_lang} pairs")
    print(f"English pivot alignment: {total_checked - english_mismatches}/{total_checked} matches ({english_mismatches} mismatches)")
    
    # Show first few aligned pairs for verification
    if len(aligned_sources) >= 3:
        print("Sample alignments:")
        for i in range(min(3, len(aligned_sources))):
            print(f"  {i+1}. SRC: {aligned_sources[i][:80]}...")
            print(f"     TGT: {aligned_targets[i][:80]}...")
            print()
    return aligned_sources, aligned_targets


def _evaluate_language_pair(
    model,
    tokenizer,
    cfg: EvalConfig,
    src_lang: str,
    tgt_lang: str,
) -> Dict[str, float]:
    """Evaluate direct translation between two non-English languages using WMT24++ aligned data."""
    print(f"\n--- Evaluating: {src_lang} → {tgt_lang} (direct translation) ---")
    
    # QUICK TEST FIRST: Verify evaluation pipeline works with 3 sentences
    print("🔍 RUNNING QUICK TEST WITH 3 SENTENCES...")
    test_sources, test_references = _load_wmt24pp_language_pair(
        cfg.wmt24pp_path, src_lang, tgt_lang, max_examples=3
    )
    
    if test_sources and test_references:
        # Test translation
        test_predictions = eval_utils.batch_translate(
            model, tokenizer, test_sources, src_lang, tgt_lang,
            cfg.batch_size, cfg.max_new_tokens
        )
        
        # Test metrics computation
        test_metrics = eval_utils.compute_advanced_metrics(test_sources, test_predictions, test_references)
        print(f"✅ TEST PASSED: BLEU={corpus_bleu(test_predictions, [[ref] for ref in test_references]).score:.2f}, COMET={test_metrics['comet22']:.4f}, kiwi={test_metrics['kiwi23']:.4f}")
    
    print("🚀 TEST SUCCESSFUL - PROCEEDING WITH FULL EVALUATION...")
    
    # NOW RUN FULL EVALUATION
    sources, references = _load_wmt24pp_language_pair(
        cfg.wmt24pp_path, src_lang, tgt_lang, max_examples=1000
    )
    
    if not sources or not references:
        print(f"No aligned data found for {src_lang}→{tgt_lang}")
        return {}
    
    print(f"Translating {len(sources)} sentences from {src_lang} to {tgt_lang}...")
    
    # Perform direct translation
    predictions = eval_utils.batch_translate(
        model, tokenizer, sources, src_lang, tgt_lang,
        cfg.batch_size, cfg.max_new_tokens
    )
    
    # Compute metrics - fix sacrebleu format: references should be list of lists
    # Each reference should be wrapped in a list for sacrebleu format
    references_formatted = [[ref] for ref in references]
    bleu_score = corpus_bleu(predictions, references_formatted).score
    chrf_score = corpus_chrf(predictions, references_formatted).score
    
    # Debug: Show first few prediction vs reference pairs
    print("First 3 prediction vs reference pairs:")
    for i in range(min(3, len(predictions))):
        print(f"  {i+1}. PRED: {predictions[i]}")
        print(f"     REF:  {references[i]}")
        print()
    
    # Compute advanced metrics
    advanced_metrics = eval_utils.compute_advanced_metrics(sources, predictions, references)
    
    results = {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "comet22": advanced_metrics["comet22"],
        "kiwi23": advanced_metrics["kiwi23"],
        "num_sentences": len(sources)
    }
    
    print(f"Results: BLEU={bleu_score:.2f}, chrF={chrf_score:.2f}, "
          f"COMET-22={results['comet22']:.4f}, kiwi-23={results['kiwi23']:.4f}")
    
    return results


def run(cfg: EvalConfig):
    """Main evaluation function for non-English language pairs."""
    print("Starting WMT24++ non-English language pair evaluation...")
    print(f"Source language: {cfg.source_lang}")
    print(f"Target languages: {cfg.target_langs}")
    print(f"Mode: {cfg.mode}")
    print(f"WMT24++ path: {cfg.wmt24pp_path}")
    
    # Load model
    print(f"\nLoading model: {cfg.model_id}")
    model, tokenizer = eval_utils.prepare_model_and_tokenizer(cfg.model_id, cfg.mode, sparsity=cfg.sparsity)


    print("Model loaded successfully")
    
    # Evaluate source language to all target languages
    all_results = []
    
    for tgt_lang in cfg.target_langs:
        if tgt_lang == cfg.source_lang:
            continue  # Skip self-translation
            
        results = _evaluate_language_pair(model, tokenizer, cfg, cfg.source_lang, tgt_lang)
        if results:
            # Add metadata
            results.update({
                "src_lang": cfg.source_lang,
                "tgt_lang": tgt_lang,
                "mode": cfg.mode,
                "language_pair": f"{cfg.source_lang}-{tgt_lang}"
            })
            all_results.append(results)
    
    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.mode == "pruned":
        output_file = f"{cfg.output_dir}/multilang_scores_{cfg.source_lang}_{cfg.mode}_s{int(cfg.sparsity*100)}.csv"
    else:
        output_file = f"{cfg.output_dir}/multilang_scores_{cfg.source_lang}_{cfg.mode}.csv"

    if all_results:
        fieldnames = ['src_lang', 'tgt_lang', 'language_pair', 'mode', 'bleu', 'chrf', 'comet22', 'kiwi23', 'num_sentences']
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        avg_bleu = sum(r['bleu'] for r in all_results) / len(all_results)
        avg_chrf = sum(r['chrf'] for r in all_results) / len(all_results)
        avg_comet = sum(r['comet22'] for r in all_results) / len(all_results)
        avg_kiwi = sum(r['kiwi23'] for r in all_results) / len(all_results)
        
        print(f"Summary for {cfg.source_lang}:")
        print(f"  Average BLEU: {avg_bleu:.2f}")
        print(f"  Average chrF: {avg_chrf:.2f}")
        print(f"  Average COMET-22: {avg_comet:.4f}")
        print(f"  Average kiwi-23: {avg_kiwi:.4f}")
        print(f"  Language pairs evaluated: {len(all_results)}")
    else:
        print("No results to save")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Multilingual non-English evaluation using WMT24++")
    parser.add_argument("--wmt24pp_path", required=True, help="Path to WMT24++ dataset directory")
    parser.add_argument("--model_id", default="facebook/nllb-200-3.3B", help="Model identifier")
    parser.add_argument("--mode", choices=["baseline", "int8", "int4", "pruned"], default="int4")
    parser.add_argument("--source_lang", required=True, help="Source language (one of the 9 target languages)")
    parser.add_argument("--target_langs", required=True, help="Comma-separated target languages")
    parser.add_argument("--batch_size", type=int, default=8, help="Translation batch size")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--max_examples", type=int, default=1000, help="Max examples to evaluate")
    parser.add_argument("--output_dir", default="outputs/wmt24pp_multilang_eval", help="Output directory")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level for pruned mode (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Parse target languages
    target_langs = [lang.strip() for lang in args.target_langs.split(',')]
    
    cfg = EvalConfig(
        wmt24pp_path=args.wmt24pp_path,
        source_lang=args.source_lang,
        target_langs=target_langs,
        model_id=args.model_id,
        mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_examples=args.max_examples,
        output_dir=args.output_dir,
        sparsity=args.sparsity,
    )
    
    eval_utils.print_env()
    print("🚀 About to start run() function...")
    result = run(cfg)
    print(f"🏁 run() function completed with result: {result}")
    return result


if __name__ == "__main__":
    sys.exit(main())