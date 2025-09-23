#!/usr/bin/env python3
"""
NLLB-200-3.3B Translation Test
- Modes: baseline (fp16/bf16), int8 (8-bit), int4 (4-bit)
- PyTorch 2.5+/CUDA 12.1 wheels, Transformers 4.46+, bitsandbytes 0.43+

Usage:
  python nllb_test.py --mode all
  python nllb_test.py --mode baseline --src eng_Latn --tgt pol_Latn --text "Hello!"
  python nllb_test.py --mode int4 --src eng_Latn --tgt deu_Latn --text "AI will change the world."
"""

import os
import sys
import argparse
import time
import torch

def _supports_bf16() -> bool:
    # BF16 widely supported on Ampere/Hopper (e.g., A100)
    if not torch.cuda.is_available():
        return False
    cap_major, _ = torch.cuda.get_device_capability()
    return cap_major >= 8  # Ampere+

def _dtype_baseline():
    return torch.bfloat16 if _supports_bf16() else torch.float16

def _print_env():
    print("="*60)
    print("ENVIRONMENT")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    try:
        import transformers, bitsandbytes, accelerate  # noqa
        print(f"Transformers: {transformers.__version__}")
        print(f"Accelerate: {accelerate.__version__}")
        print(f"bitsandbytes: {bitsandbytes.__version__}")
    except Exception as e:
        print(f"Note: could not import one of transformers/accelerate/bitsandbytes: {e}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Device 0: {torch.cuda.get_device_name(0)}")
        try:
            print(f"Capability: {torch.cuda.get_device_capability(0)}")
            print(f"BF16 supported: {_supports_bf16()}")
        except Exception:
            pass
    print(f"HF_HOME: {os.environ.get('HF_HOME','')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE','')}")
    print("="*60)

def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return tok

def _load_model_baseline(model_id: str, dtype):
    from transformers import AutoModelForSeq2SeqLM
    print(f"Loading baseline model in dtype={dtype} ...")
    t0 = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model

def _load_model_int8(model_id: str):
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
        torch_dtype=torch.float16,  # compute dtype
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model

def _load_model_int4(model_id: str):
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
    print("Loading 4-bit quantized model (NF4 + double quant) ...")
    compute_dtype = torch.bfloat16 if _supports_bf16() else torch.float16
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
        torch_dtype=compute_dtype,  # compute dtype
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")
    return model

def _report_memory(label: str):
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[{label}] Max CUDA memory allocated: {mem:.2f} GB")

def translate(model, tokenizer, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 128):
    # NLLB requires setting tokenizer.src_lang and forcing BOS of target
    tokenizer.src_lang = src_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_id is None or forced_bos_id < 0:
        raise ValueError(f"Could not resolve target language token id for {tgt_lang}")

    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    gen = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

def run_mode(mode: str, text: str, src: str, tgt: str, model_id: str):
    ok = True
    try:
        tokenizer = _load_tokenizer(model_id)

        if mode == "baseline":
            dtype = _dtype_baseline()
            model = _load_model_baseline(model_id, dtype)
            print("✅ Baseline model loaded.")
        elif mode == "int8":
            model = _load_model_int8(model_id)
            print("✅ 8-bit model loaded.")
        elif mode == "int4":
            model = _load_model_int4(model_id)
            print("✅ 4-bit model loaded.")
        else:
            raise ValueError("Unknown mode")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"Translating [{src} -> {tgt}]: {text!r}")
        out = translate(model, tokenizer, text, src, tgt)
        print(f"🟩 Translation ({mode}): {out}")
        _report_memory(mode.upper())
    except Exception as e:
        ok = False
        print(f"🟥 {mode} FAILED: {e}")
    return ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline","int8","int4","all"], default="all")
    parser.add_argument("--src", default="eng_Latn")
    parser.add_argument("--tgt", default="spa_Latn")
    parser.add_argument("--text", default="Hello, how are you today?")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B")
    args = parser.parse_args()

    _print_env()
    if torch.cuda.is_available():
        print(f"Using device: cuda -> {torch.cuda.get_device_name(0)}")
    else:
        print("Using device: cpu (performance will be limited)")

    modes = [args.mode] if args.mode != "all" else ["baseline","int8","int4"]
    passed = 0
    for m in modes:
        print("\n" + "="*60)
        print(f"MODE: {m.upper()}")
        print("="*60)
        if run_mode(m, args.text, args.src, args.tgt, args.model):
            passed += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Passed {passed}/{len(modes)}")
    if passed == len(modes):
        print("🎉 ALL SELECTED MODES PASSED!")
        return 0
    else:
        print("❌ Some modes failed. See logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
