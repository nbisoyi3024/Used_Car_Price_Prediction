import joblib
import os

def save_model(model, filepath):
    """
    Save a trained model to a file.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load a trained model from a file.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model