#!/usr/bin/env python3
"""
Shared utilities for NLLB evaluation scripts.

This module contains common functions used across:
- wmt24pp_eval.py
- wmt24pp_multilang_eval.py  
- flores_eval_parallel.py
- multilang_eval.py

Functions:
- Model loading (baseline, int8, int4, pruned)
- Batch translation
- Metrics computation (COMET-22, kiwi-23)
- Environment utilities
"""

import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================================
# Environment and Device Utilities
# ============================================================================

def supports_bf16() -> bool:
    """Check if BF16 is supported (Ampere/Hopper GPUs)."""
    if not torch.cuda.is_available():
        return False
    cap_major, _ = torch.cuda.get_device_capability()
    return cap_major >= 8  # Ampere+


def get_dtype_baseline():
    """Get optimal dtype for baseline model."""
    return torch.bfloat16 if supports_bf16() else torch.float16


def print_env():
    """Print environment information."""
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    try:
        import transformers, bitsandbytes, accelerate  # noqa
        print(f"Transformers: {transformers.__version__}")
        print(f"Accelerate: {accelerate.__version__}")
        print(f"bitsandbytes: {bitsandbytes.__version__}")
    except Exception as e:
        print(f"Note: could not import dependencies: {e}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        try:
            print(f"Capability: {torch.cuda.get_device_capability(0)}")
            print(f"BF16 supported: {supports_bf16()}")
        except Exception:
            pass
    print(f"HF_HOME: {os.environ.get('HF_HOME', '')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '')}")
    print("=" * 60)


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_tokenizer(model_id: str):
    """Load NLLB tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return tok


def load_model_baseline(model_id: str, dtype=None):
    """Load baseline model with fp16/bf16."""
    from transformers import AutoModelForSeq2SeqLM
    if dtype is None:
        dtype = get_dtype_baseline()
    print(f"Loading baseline model in dtype={dtype} ...")
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def load_model_int8(model_id: str):
    """Load 8-bit quantized model."""
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
    print("Loading 8-bit quantized model (LLM.int8) ...")
    qconf = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=qconf,
        torch_dtype=torch.float16,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def load_model_int4(model_id: str):
    """Load 4-bit quantized model."""
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
    print("Loading 4-bit quantized model (NF4 + double quant) ...")
    compute_dtype = torch.bfloat16 if supports_bf16() else torch.float16
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=qconf,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def apply_magnitude_pruning(model, sparsity: float = 0.5):
    """Apply magnitude-based unstructured pruning."""
    import torch.nn.utils.prune as prune
    
    print(f"Applying magnitude pruning with sparsity={sparsity:.2f}")
    pruned_count = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")
            pruned_count += 1
            total_params += module.weight.numel()
    
    print(f"Pruned {pruned_count} Linear layers ({total_params:,} parameters)")
    return model


def load_model_pruned(model_id: str, sparsity: float = 0.5):
    """Load model and apply magnitude pruning."""
    print(f"Loading pruned model (sparsity={sparsity:.2f}) ...")
    dtype = get_dtype_baseline()
    t0 = time.time()
    
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    model = apply_magnitude_pruning(model, sparsity=sparsity)
    print(f"Pruned model loaded in {time.time() - t0:.1f}s")
    return model


def prepare_model_and_tokenizer(model_id: str, mode: str, sparsity: float = 0.5):
    """Load model and tokenizer based on mode."""
    tokenizer = load_tokenizer(model_id)
    
    if mode == "baseline":
        model = load_model_baseline(model_id)
    elif mode == "int8":
        model = load_model_int8(model_id)
    elif mode == "int4":
        model = load_model_int4(model_id)
    elif mode == "pruned":
        model = load_model_pruned(model_id, sparsity=sparsity)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model, tokenizer


# ============================================================================
# Translation Functions
# ============================================================================

def batch_translate(
    model,
    tokenizer,
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_new_tokens: int,
    show_progress: bool = True,
) -> List[str]:
    """Translate texts in batches.
    
    Args:
        model: Translation model
        tokenizer: NLLB tokenizer
        texts: Input texts to translate
        src_lang: Source language code (e.g., 'eng_Latn')
        tgt_lang: Target language code (e.g., 'deu_Latn')
        batch_size: Batch size for translation
        max_new_tokens: Maximum tokens to generate
        show_progress: Show progress bar
        
    Returns:
        List of translated texts
    """
    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Handle case where convert_tokens_to_ids returns a list
    if isinstance(forced_bos_id, list):
        if len(forced_bos_id) > 0:
            forced_bos_id = forced_bos_id[0]
        else:
            raise ValueError(f"Could not resolve target language token id for {tgt_lang}")
    
    if forced_bos_id is None or forced_bos_id < 0:
        raise ValueError(f"Could not resolve target language token id for {tgt_lang}")
    
    results = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{src_lang}->{tgt_lang}")
    
    model.eval()
    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        
        batch_results = tokenizer.batch_decode(gen, skip_special_tokens=True)
        results.extend(batch_results)
        
        # Print sample on first batch
        if i == 0 and batch_texts:
            print(f"\nSample translation ({src_lang} -> {tgt_lang}):")
            print(f"  Input:  {batch_texts[0][:100]}...")
            print(f"  Output: {batch_results[0][:100]}...")
    
    return results


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_advanced_metrics(
    predictions: List[str],
    sources: List[str],
    references: List[str],
    batch_size: int = 8,
) -> Dict[str, Optional[float]]:
    """Compute COMET-22 and kiwi-23 metrics.
    
    Args:
        predictions: Model predictions
        sources: Source sentences
        references: Reference translations
        batch_size: Batch size for metric computation
        
    Returns:
        Dictionary with 'comet22' and 'kiwi23' scores (None if unavailable)
    """
    results = {"comet22": None, "kiwi23": None}
    
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("Warning: COMET not available. Install with: pip install unbabel-comet")
        return results
    
    # COMET-22 (reference-based)
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        comet_data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        comet_scores = model.predict(
            comet_data, 
            batch_size=batch_size, 
            gpus=1 if torch.cuda.is_available() else 0
        )
        results["comet22"] = comet_scores.system_score
        print(f"COMET-22 score: {results['comet22']:.4f}")
        
        # Clear GPU memory
        del model
        del comet_data
        del comet_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to compute COMET-22: {e}")
    
    # kiwi-23 (reference-free quality estimation)
    try:
        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
        model = load_from_checkpoint(model_path)
        kiwi_data = [
            {"src": src, "mt": pred}
            for src, pred in zip(sources, predictions)
        ]
        kiwi_scores = model.predict(
            kiwi_data,
            batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0
        )
        results["kiwi23"] = kiwi_scores.system_score
        print(f"kiwi-23 score: {results['kiwi23']:.4f}")
        
        # Clear GPU memory
        del model
        del kiwi_data
        del kiwi_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to compute kiwi-23: {e}")
    
    return results


# ============================================================================
# Language Code Mappings
# ============================================================================

# NLLB language codes to WMT24++ 2-letter codes
NLLB_TO_WMT24PP = {
    "deu_Latn": "de",
    "rus_Cyrl": "ru",
    "fra_Latn": "fr",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "lvs_Latn": "lv",
    "zul_Latn": "zu",
    "tel_Telu": "te",
    "swh_Latn": "sw",
    "eng_Latn": "en",
}

# WMT24++ 2-letter codes to NLLB language codes
WMT24PP_TO_NLLB = {v: k for k, v in NLLB_TO_WMT24PP.items()}


# ============================================================================
# CSV Filename Helpers
# ============================================================================

def get_scores_filename(mode: str, sparsity: Optional[float] = None) -> str:
    """Generate scores CSV filename based on mode and sparsity.
    
    Args:
        mode: Evaluation mode (baseline, int8, int4, pruned)
        sparsity: Sparsity level (only for pruned mode)
        
    Returns:
        Filename like 'scores_int4.csv' or 'scores_pruned_s0p50.csv'
    """
    if mode == "pruned" and sparsity is not None:
        # Format sparsity as s0p50 for 0.5
        sparsity_str = f"s{sparsity:.2f}".replace(".", "p")
        return f"scores_{mode}_{sparsity_str}.csv"
    return f"scores_{mode}.csv"


def format_metric(value: Optional[float], decimals: int = 2) -> str:
    """Format metric value for CSV output.
    
    Args:
        value: Metric value (or None)
        decimals: Number of decimal places
        
    Returns:
        Formatted string or 'N/A' if None
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"
