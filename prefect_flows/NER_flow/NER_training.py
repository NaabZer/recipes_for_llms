import os
from pathlib import Path
import spacy
from spacy.cli.train import train
from spacy.tokens import DocBin
import mlflow
import mlflow.spacy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json


labels = ['Quantity', 'Unit', 'Food', 'Variety', 'Preparation', 'Alteration',
          'Brand', 'Optional', 'State']


def plot_conf_mat(model_uri: str, val_path: Path):
    nlp = mlflow.spacy.load_model(model_uri=model_uri)
    test_set = list(DocBin().from_disk(val_path).get_docs(nlp.vocab))
    pred_ents = []
    true_ents = []

    for recipe in test_set:
        true_ents += [tok.ent_type_ for tok in recipe]
        pred_ents += [tok.ent_type_ for tok in nlp(recipe.text)]
    # create and display the confusion matrix
    cm = confusion_matrix(true_ents, pred_ents, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    f = disp.plot(xticks_rotation=70).figure_
    mlflow.log_figure(f, 'plots/confusion_matrix.png')


def log_metrics(model_uri: str, train_log_path: Path, val_path: Path):
    mlflow.log_artifact(train_log_path)
    with open(train_log_path) as f:
        for line in f:
            line_data = json.loads(line)
            if line_data['step'] == 0:
                continue
            step = line_data['step']
            for key, value in line_data['losses'].items():
                mlflow.log_metric(f"loss_{key}", value, step)
            for key, value in line_data['scores'].items():
                mlflow.log_metric(f"score_{key}", value, step)
            mlflow.log_metric("score", line_data['score'], step)
    plot_conf_mat(model_uri, val_path)


def train_ner(model_dir: Path, config_path: Path):
    train(config_path, model_dir)
    nlp = spacy.load(Path(model_dir) / 'model-best')

    mlflow.log_artifact(config_path)
    mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model")
    mlflow.set_tag('model_flavor', 'spacy')
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    return model_uri


def run_mlflow_pipe(train_path: Path, val_path: Path, model_dir: Path,
                    config_path: Path, train_log_path: Path):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))
    mlflow.set_experiment("jupyter_NER")
    if os.path.exists(train_log_path):
        os.remove(train_log_path)
    with mlflow.start_run(run_name="jupyer_test") as _:
        model_uri = train_ner(model_dir, config_path)
        log_metrics(model_uri, train_log_path, val_path)
    return model_uri
