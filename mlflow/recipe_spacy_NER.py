import random
import spacy
import mlflow
import mlflow.spacy
from spacy.util import minibatch, compounding


def train_ner(train_data, val_data, labels, iterations, drop_rate):
    nlp = spacy.blank('en')
    nlp.add_pipe("ner", last=True)

    for itn in range(iterations):
        print(f"Starting iteration {itn}")
        random.shuffle(train_data)

        losses = {}

        nlp.initialize(lambda: train_data)
        for batch in minibatch(train_data, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(
                    batch,
                    drop=drop_rate,
                    losses=losses
                    )
        print(f"Losses: {losses}")
        mlflow.log_metrics(losses)

    mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model")
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
