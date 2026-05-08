#Importing required libraries we need for our pipeline
from src.preprocessing import preprocess_pipeline
from src.modeling import get_model_pipelines,cross_validate_models, train_models, evaluate_models
from src.utils import save_model, load_model
import os
import warnings
warnings.filterwarnings("ignore", message=".*feature names.*")

DATA_PATH = "/Users/niharikabisoyi/PyCharmMiscProject/UsedCarPricePrediction/data/sample_cars.csv"
MODEL_PATH = "/Users/niharikabisoyi/PyCharmMiscProject/UsedCarPricePrediction/models"
os.makedirs(MODEL_PATH, exist_ok=True)

def run_pipeline():

        # Preprocess and split data
        print("Preprocessing and splitting data...")
        X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH)

        # Categorical columns once (on training set)
        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
        print("Categorical columns:", categorical_cols)

        # Get models
        print("Getting models...")
        models = get_model_pipelines(categorical_cols)

        # cross-validation
        print("Cross-validation...")
        cv_results = cross_validate_models(
            models=models,
            X=X_train,
            y=y_train,
            cv=5
        )

        # print results
        for name, res in cv_results.items():
            print(f"\n{name}")
            print("Mean R2:", res["mean_r2"])
            print("Std:", res["std_r2"])

        # Train models
        print("Training model...")
        trained_models = train_models(models, X_train, y_train,categorical_cols)

        # Evaluate models
        print("Evaluating model...")
        results = evaluate_models(trained_models, X_test, y_test)

        # Print results
        print("\nModel Accuracy:")

        for model_name, metrics in results.items():
            print(f"\n{model_name}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

        # Find model with highest R2
        best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean_r2'])

        best_model = trained_models[best_model_name]

        # Save the best  model
        print("Saving the best model....")
        save_path = os.path.join(MODEL_PATH, f"{best_model_name}_best_model.pkl")
        save_model(best_model, save_path)

        print(f"Best model saved: {best_model_name} with R2 = {results[best_model_name]['R2']:.4f}")
        print("Pipeline completed successfully!")

        return results, trained_models,cv_results, best_model


if __name__ == "__main__":
       run_pipeline()

