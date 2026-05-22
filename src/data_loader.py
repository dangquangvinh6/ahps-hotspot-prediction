import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    print(f"Shape ban dau: {df.shape}")
    return df
