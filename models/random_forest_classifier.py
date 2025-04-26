import joblib

def load_rf_model():
    return joblib.load("random_forest_model.pkl")
