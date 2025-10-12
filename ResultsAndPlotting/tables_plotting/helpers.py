import os
from pathlib import Path
import pandas as pd
from result_processing import flores_res_processing, multilang_res_processing, wmt24pp_res_processing, wmt_multilang_processing


DATA_DIRS = ["flores_eval", "multilang_eval", "wmt24pp_eval", "WMT_multiling_eval", "pruning_WMT"]
PROCESSOR_MAP = {
    "flores": flores_res_processing,
    "multilang": multilang_res_processing,
    "wmt24pp": wmt24pp_res_processing,
    "WMT_multiling": wmt_multilang_processing,
    "pruning_WMT": multilang_res_processing,  # reuse multilang processing for pruning
}
RESOURCE_LVL_MAP = {
    "German": "High Resource",
    "Russian": "High Resource",
    "French": "High Resource",
    "English": "High Resource",
    "Dutch": "Mid Resource",
    "Polish": "Mid Resource",
    "Latvian": "Mid Resource",
    "Zulu": "Low Resource",
    "Telugu": "Low Resource",
    "Swahili": "Low Resource"
}


def load_all_data(data_path: str = "./outputs/", data_dirs: list[str] = DATA_DIRS) -> pd.DataFrame:
    """
    Load all results data automatically.
    Standardize the format and return them all as a single big DataFrame.
    """

    result_df = pd.DataFrame()
    extractions = []

    for data_dir in data_dirs:
        full_path = Path(os.path.join(data_path, data_dir))
        if not os.path.isdir(full_path):
            raise ValueError(f"The data directory {data_dir} does not exist!")
        
        benchmark_name = data_dir.replace("_eval", "")
        file_pattern = f'{benchmark_name.replace("WMT_multiling", "multilang")}_scores_*.csv'
        file_paths = list(full_path.glob(file_pattern))
        for file_path in file_paths:
            raw_df = pd.read_csv(file_path, index_col=None)
            raw_df["benchmark"] = benchmark_name.replace("multilang", "flores").replace("WMT_multiling", "WMT24++")
            raw_df["benchmark"] = raw_df["benchmark"].replace({
                "flores": "FloRes",
                "wmt24pp": "WMT24++"
            })
            raw_df.rename({"n_samples": "num_sentences"}, inplace=True, axis=1)
            raw_df = raw_df.drop(columns=["split"], errors="ignore")
            if benchmark_name == "multilang":
                quant_level = str(file_path).split("_")[-1].replace(".csv", "")
                if quant_level not in {"baseline", "int8", "int4"}:
                    quant_level = "int4"
                raw_df["mode"] = quant_level
            processor_func = PROCESSOR_MAP[benchmark_name]
            processed_df = processor_func(raw_df)
            extractions.append(processed_df)

    result_df = pd.concat(extractions, ignore_index=True)

    # Finally, add the resource level for source and target
    result_df["source_lang_resource_level"] = result_df["source_lang"].map(RESOURCE_LVL_MAP)
    result_df["target_lang_resource_level"] = result_df["target_lang"].map(RESOURCE_LVL_MAP)

    return result_df


def load_pruning_data(data_path: str = "./outputs/") -> pd.DataFrame:
    """
    Load pruning results data specifically.
    Process pruning sparsity levels and return standardized DataFrame.
    Handles both multilang format (X↔Y) and FLORES format (English↔X).
    """
    result_df = pd.DataFrame()
    extractions = []
    
    pruning_dir = "pruning_WMT"
    full_path = Path(os.path.join(data_path, pruning_dir))
    if not os.path.isdir(full_path):
        raise ValueError(f"The pruning directory {pruning_dir} does not exist!")
    
    # Find multilang pruning CSV files (X↔Y translations)
    multilang_pattern = 'multilang_scores_*_pruned_s*.csv'
    multilang_files = list(full_path.glob(multilang_pattern))
    
    for file_path in multilang_files:
        raw_df = pd.read_csv(file_path, index_col=None)
        raw_df["benchmark"] = "Pruning-WMT"
        
        # Extract sparsity level from filename (e.g., s10 -> 10% Sparsity)
        filename = str(file_path.name)
        sparsity_match = filename.split('_s')[-1].replace('.csv', '')
        sparsity_level = f"{sparsity_match}% Sparsity"
        raw_df["mode"] = sparsity_level
        
        # Process using pruning processing function
        from result_processing import pruning_res_processing
        processed_df = pruning_res_processing(raw_df)
        extractions.append(processed_df)
    
    # Find FLORES-style pruning CSV files (English↔X translations)
    flores_pattern = 'scores_pruned_s*.csv'
    flores_files = list(full_path.glob(flores_pattern))
    
    for file_path in flores_files:
        raw_df = pd.read_csv(file_path, index_col=None)
        raw_df["benchmark"] = "Pruning-WMT"
        
        # Extract sparsity level from filename (e.g., s0p10 -> 10% Sparsity)
        filename = str(file_path.name)
        if 's0p' in filename:
            sparsity_match = filename.split('s0p')[-1].replace('.csv', '')
            sparsity_level = f"{sparsity_match}% Sparsity"
        else:
            # Fallback for other patterns
            sparsity_match = filename.split('_s')[-1].replace('.csv', '')
            sparsity_level = f"{sparsity_match}% Sparsity"
        raw_df["mode"] = sparsity_level
        
        # Process using specialized pruning FLORES processing function
        from result_processing import pruning_flores_res_processing
        processed_df = pruning_flores_res_processing(raw_df)
        extractions.append(processed_df)
    
    result_df = pd.concat(extractions, ignore_index=True)
    
    # Add resource levels
    result_df["source_lang_resource_level"] = result_df["source_lang"].map(RESOURCE_LVL_MAP)
    result_df["target_lang_resource_level"] = result_df["target_lang"].map(RESOURCE_LVL_MAP)
    
    return result_df


def load_pruning_data(data_path: str = "./outputs/") -> pd.DataFrame:
    """
    Load pruning results data specifically.
    Process pruning sparsity levels and return standardized DataFrame.
    """
    result_df = pd.DataFrame()
    extractions = []
    
    pruning_dir = "pruning_WMT"
    full_path = Path(os.path.join(data_path, pruning_dir))
    if not os.path.isdir(full_path):
        raise ValueError(f"The pruning directory {pruning_dir} does not exist!")
    
    # Find all pruning CSV files
    file_pattern = 'multilang_scores_*_pruned_s*.csv'
    file_paths = list(full_path.glob(file_pattern))
    
    for file_path in file_paths:
        raw_df = pd.read_csv(file_path, index_col=None)
        raw_df["benchmark"] = "Pruning-WMT"  # New benchmark category for pruning
        
        # Extract sparsity level from filename (e.g., s10 -> 10% Sparsity)
        filename = str(file_path.name)
        sparsity_match = filename.split('_s')[-1].replace('.csv', '')
        sparsity_level = f"{sparsity_match}% Sparsity"
        raw_df["mode"] = sparsity_level
        
        # Process using pruning processing function
        from result_processing import pruning_res_processing
        processed_df = pruning_res_processing(raw_df)
        extractions.append(processed_df)
    
    result_df = pd.concat(extractions, ignore_index=True)
    
    # Add resource levels
    from helpers import RESOURCE_LVL_MAP
    result_df["source_lang_resource_level"] = result_df["source_lang"].map(RESOURCE_LVL_MAP)
    result_df["target_lang_resource_level"] = result_df["target_lang"].map(RESOURCE_LVL_MAP)
    
    return result_df
