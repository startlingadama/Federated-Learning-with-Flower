FROM python:3.10-slim

WORKDIR /app
COPY client.py model_pt.py . 

RUN pip install flwr[dashboard]==1.8.0 torch numpy

CMD ["python", "client.py"]
