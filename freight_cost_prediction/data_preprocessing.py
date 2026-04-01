import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd

def load_vendor_invoice_data(db_path: str):
    """Load vendor invoice data from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features(df: pd.DataFrame, use_full_features=False):
    """Select features and target variable"""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    if use_full_features:
        X = df[["Dollars", "Quantity", "unit_cost", "is_bulk"]]
    else:
        X = df[["Dollars", "Quantity"]]
    y = df["Freight"]
    return X, y

def split_data(X:pd.DataFrame, y:pd.Series, test_size:float=0.2, random_state:int=42):
    """Split dataset into train and test sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)