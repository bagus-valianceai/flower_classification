import os
import mlflow
import dotenv
from sklearn import datasets
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Flower Classification")

    # Data preparation
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.92
    )

    # EDA
    # Preprocessing
    # Feature Engineering

    # Training
    params = {
        "max_iter": 100,
        "random_state": 2,
    }
    lr = LogisticRegression(**params)
    lr.fit(
        X_train,
        y_train
    )

    # Evalutaion
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Unit test

    # Logging MLflow
    with mlflow.start_run(run_name = "Third run"):
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        signature = infer_signature(X_train, lr.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model = lr,
            artifact_path = "iris_model",
            signature = signature,
            input_example = X_train,
            registered_model_name = "Untouch Logistic Regression",
        )
    
    # DON'T FORGET TO SET ALIAS "STAGE" TO ONE OF LOGGED MODEL IN MLFLOW WEB UI BEFORE CREATE PULL REQUEST TO MASTER BRANCH