FROM python:3.10

WORKDIR /NER_Api
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["fastapi", "run", "main.py"]
CMD ["--port", "9000"]
