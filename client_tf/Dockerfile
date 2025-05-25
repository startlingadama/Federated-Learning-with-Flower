FROM python:3.10-slim

WORKDIR /app
COPY client.py model_tf.py . 

RUN pip install flwr tensorflow numpy

CMD ["python", "client.py"]
