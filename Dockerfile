FROM python:3.12-slim
WORKDIR /srv
COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
CMD ["python", "server.py"]
