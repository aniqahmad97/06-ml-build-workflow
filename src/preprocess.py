import pandas as pd

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Loads a CSV file and drops rows with missing values.
    """
    df = pd.read_csv(csv_path)
    return df.dropna()
