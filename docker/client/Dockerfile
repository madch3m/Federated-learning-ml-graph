FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install torch
EXPOSE 5000
CMD ["python", "server.py"]