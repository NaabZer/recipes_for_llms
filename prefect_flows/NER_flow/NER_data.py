import os
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from doccano_api import Doccano_API
from prefect import flow, task
import spacy
import subprocess
from spacy.tokens import DocBin

load_dotenv()

labels = ['Quantity', 'Unit', 'Food', 'Variety', 'Preparation', 'Alteration',
          'Brand', 'Optional', 'State']


def doc_from_ds(ds):
    db = DocBin()
    nlp = spacy.blank("en")
    for i, [text, annotations] in ds[['text', 'label']].iterrows():
        try:
            doc = nlp.make_doc(text)
            entities = [doc.char_span(*annotation) for annotation
                        in annotations]
            doc.ents = entities
            db.add(doc)
        except TypeError:
            continue
    return db


def gen_config(train_path: str, dev_path: str, train_log_path: str,
               out_path: str):
    BASE_CONFIG = f"""[paths]
    train = {train_path}
    dev = {dev_path}
    vectors = null
    [system]
    gpu_allocator = null

    [nlp]
    lang = "en"
    pipeline = ["tok2vec","ner"]
    batch_size = 1000

    [components]

    [components.tok2vec]
    factory = "tok2vec"

    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"

    [components.tok2vec.model.embed]
    @architectures = "spacy.MultiHashEmbed.v2"
    width = ${{components.tok2vec.model.encode.width}}
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    rows = [5000, 1000, 2500, 2500]
    include_static_vectors = false

    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 96
    depth = 4
    window_size = 1
    maxout_pieces = 3

    [components.ner]
    factory = "ner"

    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = true
    nO = null

    [components.ner.model.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = ${{components.tok2vec.model.encode.width}}

    [corpora]

    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = ${{paths.train}}
    max_length = 0

    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = ${{paths.dev}}
    max_length = 0

    [training]
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"

    [training.optimizer]
    @optimizers = "Adam.v1"

    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2

    [training.batcher.size]
    @schedules = "compounding.v1"
    start = 100
    stop = 1000
    compound = 1.001

    [initialize]
    vectors = ${{paths.vectors}}

    [training.logger]
    @loggers = "spacy.ConsoleLogger.v3"
    progress_bar = "eval"
    console_output = true
    output_file = {train_log_path}
    """

    with open("temp.cfg", 'w') as f:
        f.write(BASE_CONFIG)
    subprocess.call(
            ['python', '-m', 'spacy', 'init',
             'fill-config', 'temp.cfg', out_path]
            )
    os.remove('temp.cfg')


@task
def download_doccano_data(project_id: int):
    """ Task: Download data from doccano, unzip the JSONL file,
    and return the path to it """
    api_base = os.getenv('DOCCANO_API_URL')
    docc = Doccano_API(api_base)
    docc.login(os.getenv('DOCCANO_API_USERNAME'),
               os.getenv('DOCCANO_API_PASSWORD'))
    fp = docc.download(project_id)
    now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    zipped_path = f"./temp_{now}/"
    with zipfile.ZipFile(fp, 'r') as zip_file:
        zip_file.extractall(zipped_path)
    # remove zip file
    os.remove(fp)
    return zipped_path


@flow
def prepare_ner_data():
    return None
