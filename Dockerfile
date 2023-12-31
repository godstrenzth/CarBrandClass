FROM python:3.11

WORKDIR /work

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /work/requirements.txt

COPY ./app /work/app
COPY ./model /work/model

ENV PYTHONPATH "${PYTHONPATH}:/work"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]