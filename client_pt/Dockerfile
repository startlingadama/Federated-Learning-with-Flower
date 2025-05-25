FROM python:3.10-slim

WORKDIR /app
COPY client.py model_pt.py . 

RUN pip install flwr torch numpy

CMD ["python", "client.py"]
