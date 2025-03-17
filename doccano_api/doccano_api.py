import time
import pathlib
from datetime import datetime
import requests


class Doccano_API:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.api_url = self.base_url + '/v1'
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "referer": base_url,
        }
        self.session.headers.update(headers)

    def login(self, username: str, password: str) -> None:
        """Login to a session with the Doccano instance related to the base url

        Args:
            username (str): The username of the user
            password (str): The password of the user
        """
        login_url = self.api_url + '/auth/login/'
        _ = self.session.post(login_url, json={"username": username,
                                               "password": password})
        self.session.headers.update({"X-CSRFToken":
                                     self.session.cookies.get("csrftoken")})

    def get(self, url, **kwargs):
        if not url.startswith(self.api_url):
            url = self.api_url + url
        response = self.session.get(url, **kwargs)
        return response

    def post(self, url, **kwargs):
        if not url.startswith(self.api_url):
            url = self.api_url + url
        response = self.session.post(url, **kwargs)
        return response

    def download(self, project_id, only_approved=True):
        # Prep download
        url = f"/projects/{project_id}/download"
        data = {'format': 'JSONL', 'exportApproved': only_approved}
        response = self.post(url, json=data)
        print(response.json())
        task_id = response.json()["task_id"]

        time.sleep(10)

        # Get downloaded file
        data = {"taskId": task_id}
        response = self.get(url, params=data, stream=True)
        dir_path = pathlib.Path(".")
        now = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
        file_name = f"doccano_{project_id}_{now}.zip"
        file_path = dir_path / file_name
        with file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
