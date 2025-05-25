FROM python:3.10-slim

WORKDIR /app
COPY client.py model_tf.py . 

RUN pip install flwr[dashboard]==1.8.0 tensorflow numpy

CMD ["python", "client.py"]
