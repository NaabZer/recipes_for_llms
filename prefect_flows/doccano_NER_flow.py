import os
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from doccano_api import Doccano_API
from prefect import flow, task

load_dotenv()


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
