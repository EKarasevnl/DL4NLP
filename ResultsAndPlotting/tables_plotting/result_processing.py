import numpy as np
import pandas as pd


LANG_MAP = {
    "deu": "German",
    "de": "German",
    "rus": "Russian",
    "ru": "Russian",
    "fra": "French",
    "fr": "French",
    "nld": "Dutch",
    "nl": "Dutch",
    "pol": "Polish",
    "pl": "Polish",
    "lvs": "Latvian",
    "lv": "Latvian",
    "zul": "Zulu",
    "zu": "Zulu",
    "tel": "Telugu",
    "te": "Telugu",
    "swh": "Swahili",
    "sw": "Swahili",
}
QUANT_MAP = {
    "baseline": "Baseline",
    "int8": "8-Bit",
    "int4": "4-Bit"
}
SPARSITY_MAP = {
    "10% Sparsity": "10% Sparsity",
    "30% Sparsity": "30% Sparsity", 
    "50% Sparsity": "50% Sparsity"
}


def flores_res_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing to standardize the results from the flores_eval directory.
    """

    # First, clean up the languages
    df['base_lang_code'] = df['lang'].str.split('_', n=1).str[0]
    df['mapped_name'] = df['base_lang_code'].map(LANG_MAP)
    df["quant_level"] = df["mode"].map(QUANT_MAP)
    english_name = 'English'
    is_en_to_x = df['direction'] == 'en2x'
    df['source_lang'] = np.where(
        is_en_to_x,
        english_name,
        df['mapped_name']
    )

    df['target_lang'] = np.where(
        is_en_to_x,
        df['mapped_name'],
        english_name
    )

    df = df.drop(columns=['lang', 'direction', 'base_lang_code', 'mapped_name', 'mode'])
    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'quant_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'quant_level']]
    return df[cols]


def multilang_res_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing to standardize the results from the multilang_eval directory.
    """

    # First, clean up the languages
    df['source_lang_code'] = df['src_lang'].str.split('_', n=1).str[0]
    df['target_lang_code'] = df['tgt_lang'].str.split('_', n=1).str[0]
    df['source_lang'] = df['source_lang_code'].map(LANG_MAP)
    df['target_lang'] = df['target_lang_code'].map(LANG_MAP)
    df["quant_level"] = df["mode"].map(QUANT_MAP)
    df = df.drop(columns=['src_lang', 'tgt_lang', 'source_lang_code', 'target_lang_code', 'dataset', 'mode'])

    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'quant_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'quant_level']]
    return df[cols]


def wmt24pp_res_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing to standardize the results from the wmt24pp_eval directory.
    """

    # First, clean up the languages
    df['base_lang_code'] = df['lang'].str.split('_', n=1).str[0]
    df['mapped_name'] = df['base_lang_code'].map(LANG_MAP)
    df["quant_level"] = df["mode"].map(QUANT_MAP)
    english_name = 'English'
    is_en_to_x = df['direction'] == 'en2x'
    df['source_lang'] = np.where(
        is_en_to_x,
        english_name,
        df['mapped_name']
    )

    df['target_lang'] = np.where(
        is_en_to_x,
        df['mapped_name'],
        english_name
    )

    df = df.drop(columns=['lang', 'direction', 'base_lang_code', 'mapped_name', 'mode'])
    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'quant_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'quant_level']]
    return df[cols]


def wmt_multilang_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing to standardize the results from the WMT_multiling_eval directory.
    """
    # First, clean up the languages
    df['source_lang_code'] = df['src_lang'].str.split('_', n=1).str[0]
    df['target_lang_code'] = df['tgt_lang'].str.split('_', n=1).str[0]
    df['source_lang'] = df['source_lang_code'].map(LANG_MAP)
    df['target_lang'] = df['target_lang_code'].map(LANG_MAP)
    df["quant_level"] = df["mode"].map(QUANT_MAP)
    df = df.drop(columns=['src_lang', 'tgt_lang', 'source_lang_code', 'target_lang_code', 'mode', 'language_pair'], errors='ignore')
    
    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'quant_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'quant_level']]
    return df[cols]


def pruning_flores_res_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing for pruning data in FLORES format (English↔X translations).
    """
    # First, clean up the languages
    df['base_lang_code'] = df['lang'].str.split('_', n=1).str[0]
    df['mapped_name'] = df['base_lang_code'].map(LANG_MAP)
    df["sparsity_level"] = df["mode"]  # Keep sparsity level as is (don't map through QUANT_MAP)
    english_name = 'English'
    is_en_to_x = df['direction'] == 'en2x'
    df['source_lang'] = np.where(
        is_en_to_x,
        english_name,
        df['mapped_name']
    )

    df['target_lang'] = np.where(
        is_en_to_x,
        df['mapped_name'],
        english_name
    )

    df = df.drop(columns=['lang', 'direction', 'base_lang_code', 'mapped_name', 'mode', 'split'], errors='ignore')
    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'sparsity_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'sparsity_level']]
    return df[cols]


def pruning_res_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom processing to standardize the results from the pruning_WMT directory.
    """
    # First, clean up the languages
    df['source_lang_code'] = df['src_lang'].str.split('_', n=1).str[0]
    df['target_lang_code'] = df['tgt_lang'].str.split('_', n=1).str[0]
    df['source_lang'] = df['source_lang_code'].map(LANG_MAP)
    df['target_lang'] = df['target_lang_code'].map(LANG_MAP)
    df["sparsity_level"] = df["mode"]  # Keep the sparsity level as is
    df = df.drop(columns=['src_lang', 'tgt_lang', 'source_lang_code', 'target_lang_code', 'mode', 'language_pair'], errors='ignore')
    
    # Re-ordering cols for better readability.
    cols = ['source_lang', 'target_lang', 'sparsity_level'] + [col for col in df.columns if col not in ['source_lang', 'target_lang', 'sparsity_level']]
    return df[cols]
