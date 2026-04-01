import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """Load trained freight cost prediction model"""
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    return data["model"], data["features"]

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices,
    Parameters:
    ----------------
    input_data: dict
    Returns
    ----------------
    pd.DataFrame with predicted freight cost
    """
    model, features = load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[features]
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    #Example inference run(local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200],
        "Quantity": [2000, 900, 250, 10]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)