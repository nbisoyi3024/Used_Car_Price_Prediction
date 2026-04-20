import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# --- Load and clean data ---
def load_and_clean_data(path):
    df = pd.read_csv(path)
    return df

# --- Split into features and target ---
def split_features_target(df, features, target='Price_log'):
    X = df[features]  # select only the features you want
    y = df[target]# select target
    return X, y


# --- Full preprocessing pipeline ---
def preprocess_pipeline(path):
    # Load & clean
    df = load_and_clean_data(path)

    # Define features
    features = [
        'Manufacturer', 'Model', 'Mileage', 'Drivetrain', 'Fuel Type',
        'Accidents Or Damage', 'One Owner', 'Personal Use Only',
        'Seller Rating', 'Driver Rating', 'Engine_Size', 'Transmission_clean',
        'Mpg_Clean'
    ]

    # Split into X and y
    X = df[features]
    y = df['Price_log']     #raw price

    print(y.head())

    # apply log transform
    #y_log = np.log1p(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test