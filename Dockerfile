FROM python:3.10-slim

WORKDIR /serve

COPY app.py /serve/app.py
COPY utils.py /serve/utils.py
COPY model.pkl /serve/model.pkl
COPY data/ /serve/data
COPY requirements.txt /serve/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD [ "python", "/serve/app.py" ]
