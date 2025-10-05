import os
from pathlib import Path
import pandas as pd
from .result_processing import flores_res_processing, multilang_res_processing, wmt24pp_res_processing, wmt_multilang_processing


DATA_DIRS = ["flores_eval", "multilang_eval", "wmt24pp_eval", "WMT_multiling_eval"]
PROCESSOR_MAP = {
    "flores": flores_res_processing,
    "multilang": multilang_res_processing,
    "wmt24pp": wmt24pp_res_processing,
    "WMT_multiling": wmt_multilang_processing,
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


def load_all_data(data_path: str = "./DL4NLP/outputs/", data_dirs: list[str] = DATA_DIRS) -> pd.DataFrame:
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
        file_pattern = f"{benchmark_name.replace("WMT_multiling", "multilang")}_scores_*.csv"
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
