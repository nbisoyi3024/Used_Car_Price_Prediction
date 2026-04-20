import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def get_model_pipelines(categorical_cols):
        preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
        )

        # --- CatBoost ---
        cb_pipeline = Pipeline([
            ("cb_model",CatBoostRegressor(
             cat_features = categorical_cols,  # list of column names
             iterations = 300,
             learning_rate = 0.1,
             verbose = 50,
             random_state = 42
            ))
        ])
        # --- LightGBM ---
        lgb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("lgb_model",lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42
        ))
        ])

        # Return as dictionary for train_models
        return { "CB": cb_pipeline, "LightGBM": lgb_pipeline}

def train_models(models, X_train, y_train, categorical_cols):
    trained_models = {}
    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save feature columns once
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    joblib.dump(categorical_cols, "models/categorical_cols.pkl")

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def cross_validate_models(models, X, y,cv=5):
    results = {}
    #encode categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ],
        remainder="passthrough"
    )

    for name, model in models.items():
        print(f"Cross-validating {name}...")

        # LightGBM needs categorical dtype
        if isinstance(model, lgb.LGBMRegressor):
            pipeline = Pipeline([
               ("preprocess", preprocessor),
               ("model", model)
           ])
        else:
           pipeline = model  # CatBoost handles raw categorical internally

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        results[name] = {
            "scores": scores,
            "mean_r2": np.mean(scores),
            "std_r2": np.std(scores)
        }

    return results

def predict_models(models, X_test):

        predictions = {}

        for name, model in models.items():
              print(f"Predicting with {name}...")

              predictions[name] = model.predict(X_test)

        return predictions

def evaluate_models(models, X_test, y_test):

            results = {}

            for name, model in models.items():
                print(f"Evaluating {name}...")

                # predictions
                y_pred_log = model.predict(X_test)

                # convert log price of y → real price as it was log tranformed)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_test)

                # metrics
                results[name] = {
                    "R2": r2_score(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": root_mean_squared_error(y_true, y_pred)
                }

            return results
