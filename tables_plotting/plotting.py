from .helpers import load_all_data



if __name__ == "__main__":
    df = load_all_data()
    df.to_csv("DL4NLP/outputs/collected_results/results.csv", index=False)
