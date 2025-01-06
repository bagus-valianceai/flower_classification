import os
import dotenv
import mlflow
import requests

dotenv.load_dotenv(".env")

def test_model_availability():
    # Arrange
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    CHOSEN_MODEL = os.getenv("MODEL_ALIAS")

    mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Flower Classification")
    
    # Act
    chosen_model = mlflow.pyfunc.load_model(f"models:/Untouch Logistic Regression@{CHOSEN_MODEL}")

    # Assert
    assert type(chosen_model) == mlflow.pyfunc.PyFuncModel

def test_predict_on_ci():
    # Arrange
    MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
    
    data = {"x1": 1, "x2": 1, "x3": 1, "x4": 1}

    # Act
    res = requests.post(MODEL_SERVER_IP, json = data)

    # Assert
    assert res.status_code == 200
