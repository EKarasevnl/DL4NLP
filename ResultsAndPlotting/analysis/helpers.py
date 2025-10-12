

# Convert the columns with suffixes (K, M) into numeric
def convert_to_number(x: str) -> float:
    """Convert human-readable counts with K/M into float values."""
    if isinstance(x, str):
        x = x.strip()
        if x.endswith("K"):
            return float(x[:-1]) * 1e3
        elif x.endswith("M"):
            return float(x[:-1]) * 1e6
        elif x.endswith("B"):
            return float(x[:-1]) * 1e9
        elif x == "___" or x == "—":
            return None
    return float(x)