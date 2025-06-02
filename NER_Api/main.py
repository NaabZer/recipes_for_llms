import mlflow
import mlflow.spacy
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))

model_url = "models:/recipe_NER@prod"
nlp = mlflow.spacy.load_model(model_uri=model_url)
model_info = mlflow.models.get_model_info(model_uri=model_url)

app = FastAPI()


class ApiText(BaseModel):
    text: str


class ApiData(BaseModel):
    data: ApiText


class ApiInput(BaseModel):
    tasks: List[ApiData]


@app.post("/reload")
def reload():
    global nlp, model_info
    nlp = mlflow.spacy.load_model(model_uri=model_url)
    model_info = mlflow.models.get_model_info(model_uri=model_url)
    return {}


@app.get("/metrics")
def metrics():
    return {}


@app.post("/setup")
def setup():
    return {"model_version": model_info.run_id}


@app.post("/webhook")
def webhook():
    return {"result": "test", "status": "ok"}


@app.get("/")
@app.get("/health")
def health():
    return {"status": "UP"}


@app.post("/predict")
def tag_NER(api_input: ApiInput):
    results = []
    for task in api_input.tasks:
        text = task.data.text
        out = nlp(text)
        result = []
        for t in out.ents:
            value = {
                "start": t.start_char,
                "end": t.end_char,
                "labels": [t.label_],
                "text": str(t)
                }
            print(value)
            result.append({
                "value": value,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                })
        out_json = {
                "score": 0.5,  # TODO: Can we get some score from spacy?
                "model_version": model_info.run_id,
                "result": result
                }
        results.append(out_json)
    return {"results": results}
