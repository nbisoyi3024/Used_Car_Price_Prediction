from google.cloud import storage
import joblib

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


def load_model_from_gcs(bucket_name, model_filename):
    # Step 1 — connect to GCS
    client = storage.Client(project="gcplatform-495621")

    # Step 2 — get the bucket
    bucket = client.bucket(bucket_name)

    # Step 3 — get the specific file (blob)
    blob = bucket.blob(model_filename)

    # Step 4 — download it to a local temp file
    local_path = "/tmp/model.pkl"
    blob.download_to_filename(local_path)

    # Step 5 — load using joblib (matches how it was saved)
    model = joblib.load(local_path)

    return model
