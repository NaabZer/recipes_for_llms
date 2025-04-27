import mlflow
import mlflow.spacy
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))

model_url = "models:/recipe_NER@prod"
nlp = mlflow.spacy.load_model(model_uri=model_url)

app = FastAPI()


class ApiInput(BaseModel):
    text: str


@app.post("/")
def tag_NER(api_input: ApiInput):
    out = nlp(api_input.text)
    label = []
    for t in out:
        if t.ent_type_:
            label_inner = [t.idx, t.idx + len(t), t.ent_type_]
            label.append(label_inner)
    out_jsonl = {"text": api_input.text, "label": label}
    return out_jsonl
