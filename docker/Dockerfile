FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -o Acquire::Retries=3 \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY requirements-ml.txt /app/requirements-ml.txt
RUN pip install --no-cache-dir -r /app/requirements-ml.txt
# musicnn'i pip yerine github'dan kur (pip paketindeki eski numpy constraint yüzünden)
RUN pip install --no-cache-dir --no-deps musicnn==0.1.0


COPY app /app/app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

