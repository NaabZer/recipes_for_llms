import os
from dotenv import load_dotenv
from doccano_api import Doccano_API

load_dotenv()

api_base = os.getenv('DOCCANO_API_URL')
docc = Doccano_API(api_base)
docc.login(os.getenv('DOCCANO_API_USERNAME'),
           os.getenv('DOCCANO_API_PASSWORD'))
docc.download(1)
