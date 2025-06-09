import polars as pl
import mlflow
import mlflow.sklearn
import mlflow.spacy
import sklearn
from dotenv import load_dotenv
import os
from app.data_handling.preprocessing import lemmatize_line

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))

BOW_Model: sklearn.feature_extraction.text.TfidfVectorizer = None
BOW_Model_info: mlflow.models.model.ModelInfo = None


def run_BOW_on_line(line: str | list, model_uri: str):
    input_is_str = False
    global BOW_Model, BOW_Model_info
    if BOW_Model_info is None or BOW_Model_info.model_uri != model_uri:
        BOW_Model = mlflow.sklearn.load_model(model_uri)
        BOW_Model_info = mlflow.models.get_model_info(model_uri=model_uri)

    if type(line) is str:
        input_is_str = True
        line = [line]

    line = [lemmatize_line(ln) for ln in line]

    embs = BOW_Model.transform(line)
    if input_is_str:
        return embs[0]
    return embs
