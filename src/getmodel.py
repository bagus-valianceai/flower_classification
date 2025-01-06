import os
import mlflow
import dotenv

dotenv.load_dotenv(".env")
mlflow.set_tracking_uri(uri = "http://43.218.118.11:5000")
mlflow.set_experiment("Flower Classification")

model = mlflow.pyfunc.load_model(f"models:/Untouch Logistic Regression@stage")
print(model)
